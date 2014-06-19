FROM google/golang-runtime
MAINTAINER dengnan@google.com vmarmol@google.com proppy@google.com

# TODO(vmarmol): Build from source.
# Get lmctfy and its dependencies.
RUN apt-get update -y --force-yes &&  apt-get install -y --no-install-recommends --force-yes pkg-config libapparmor1
ADD http://storage.googleapis.com/cadvisor-bin/lmctfy/libre2.so.0.0.0 /usr/lib/libre2.so.0
ADD http://storage.googleapis.com/cadvisor-bin/lmctfy/lmctfy /usr/bin/lmctfy
RUN chmod +x /usr/bin/lmctfy

# Install libprotobuf8.
ADD http://storage.googleapis.com/cadvisor-bin/lmctfy/libprotobuf8_2.5.0-9_amd64.deb /tmp/libprotobuf8_2.5.0-9_amd64.deb
ADD http://storage.googleapis.com/cadvisor-bin/lmctfy/libc6_2.19-1_amd64.deb /tmp/libc6_2.19-1_amd64.deb
RUN dpkg -i /tmp/libc6_2.19-1_amd64.deb /tmp/libprotobuf8_2.5.0-9_amd64.deb

# The image builds the app and exposes it on 8080.
