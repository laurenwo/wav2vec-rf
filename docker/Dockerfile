
ARG REPO=amr-registry.caas.intel.com/aipg
ARG OWNER=ai-lab
ARG CUDA_VERSION=11.3.0-cudnn8-runtime-ubuntu20.04
ARG PACKAGER=pip
FROM ${REPO}/${OWNER}-${PACKAGER}:${CUDA_VERSION}

USER root
RUN apt-get --fix-broken install

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN useradd -u 11959762 -g 100 laurenwo && \
    rm -rf /var/log/lastlog && \
    rm -rf /var/log/faillog && \
    usermod -aG sudo laurenwo
                       
