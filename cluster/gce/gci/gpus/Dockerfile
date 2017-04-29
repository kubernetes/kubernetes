FROM ubuntu:16.04

# Disable prompts from apt
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -qq update
RUN apt-get install -qq -y pciutils gcc g++ git make dpkg-dev bc module-init-tools curl 

RUN mkdir /lakitu-kernel
RUN git clone https://chromium.googlesource.com/chromiumos/third_party/kernel /lakitu-kernel

ADD installer.sh /usr/bin/nvidia-installer.sh
RUN chmod a+x /usr/bin/nvidia-installer.sh
CMD ["/usr/bin/nvidia-installer.sh"]