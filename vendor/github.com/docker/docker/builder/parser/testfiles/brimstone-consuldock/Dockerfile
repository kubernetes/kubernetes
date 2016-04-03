FROM brimstone/ubuntu:14.04

MAINTAINER brimstone@the.narro.ws

# TORUN -v /var/run/docker.sock:/var/run/docker.sock

ENV GOPATH /go

# Set our command
ENTRYPOINT ["/usr/local/bin/consuldock"]

# Install the packages we need, clean up after them and us
RUN apt-get update \
	&& dpkg -l | awk '/^ii/ {print $2}' > /tmp/dpkg.clean \
    && apt-get install -y --no-install-recommends git golang ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists \

	&& go get -v github.com/brimstone/consuldock \
    && mv $GOPATH/bin/consuldock /usr/local/bin/consuldock \

	&& dpkg -l | awk '/^ii/ {print $2}' > /tmp/dpkg.dirty \
	&& apt-get remove --purge -y $(diff /tmp/dpkg.clean /tmp/dpkg.dirty | awk '/^>/ {print $2}') \
	&& rm /tmp/dpkg.* \
	&& rm -rf $GOPATH
