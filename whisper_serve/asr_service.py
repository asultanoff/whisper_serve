import argparse, asyncio, websockets
import time, struct
import numpy as np

from faster_whisper import WhisperModel


class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=np.float32)
        self.index = 0

    def append(self, array):
        incoming_length = len(array)
        if incoming_length > self.capacity:
            #trim the array
            array = array[-self.capacity:]

        if self.index + incoming_length < self.capacity:
            self.data[self.index:self.index + incoming_length] = array
            #clamp the index to the capacity
            self.index = max(self.index + incoming_length, self.capacity)
        else:
            self.data = np.roll(self.data, -incoming_length)
            self.data[-incoming_length:] = array

    def get_data(self):
        return self.data
    
    def get_index(self):
        return self.index

parser = argparse.ArgumentParser(description='Whisper ASR online server')
parser.add_argument('--model_dir', type=str, help='Directory with models')
parser.add_argument('--lan', type=str, help='Language code', default='uz')
parser.add_argument('--chunk-length', type=int, help='Chunk length in seconds', default=3)
parser.add_argument('--port', )

args = parser.parse_args()
SAMPLE_RATE=16000
BUFFER_SIZE=SAMPLE_RATE*5 # 15 seconds of audio buffer

buffer = RingBuffer(BUFFER_SIZE)

asr = WhisperModel(args.model_dir, 'cuda', compute_type='float16')



#warm up the model
audio = np.random.rand(16000)
asr.transcribe(audio)

last_message_length = 0

async def asr_processor(websocket, path):
    global last_message_length
    try:
        async for message in websocket:
            server_ts = time.time()
            timestamp_bytes = message[:8]
            audio_bytes = message[8:]
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append(audio)
            client_ts = struct.unpack('d', timestamp_bytes)[0]
            latency = server_ts-client_ts
            segments, info = asr.transcribe(buffer.get_data(), language=args.lan, task='transcribe', word_timestamps=True, temperature=0.0, no_repeat_ngram_size=2, repetition_penalty=3, vad_filter=True) 

            text = ' '.join([seg.text for seg in segments])
            
            padding = ' ' * (last_message_length-len(text))
            print(f"\r{text}{padding}", flush=True, end='')
            last_message_length = len(text)
    except Exception as e:
        print(f"Exception {e}")

start_server = websockets.serve(asr_processor, "0.0.0.0", args.port,
                                ping_interval=10000,
                                ping_timeout=5000,
                                max_size=2**30)
asyncio.get_event_loop().run_until_complete(start_server)
print(f"ASR server started on port {args.port}")
asyncio.get_event_loop().run_forever()


