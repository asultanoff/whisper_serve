# Use the specified image as the base
FROM ghcr.io/opennmt/ctranslate2:4.1.1-ubuntu20.04-cuda12.2

# Run apt-get commands to install neovim, ffmpeg, and libsndfile1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    neovim ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*



# Set the working directory to /workspace
WORKDIR /tmp_workdir

COPY requirements.txt /tmp_workdir

#Install Python dependencies from requirement.txt
RUN pip install --no-cache-dir -r /tmp_workdir/requirements.txt

WORKDIR /workspace

ENTRYPOINT ["/bin/bash"]

