# set author and base
FROM fedora
MAINTAINER Luis Pabón <lpabon@redhat.com>

LABEL version="1.3.1"
LABEL description="Development build"

# let's setup all the necessary environment variables
ENV BUILD_HOME=/build
ENV GOPATH=$BUILD_HOME/golang
ENV PATH=$GOPATH/bin:$PATH
ENV HEKETI_BRANCH="master"

RUN curl https://glide.sh/get | sh

# install dependencies, build and cleanup
RUN mkdir $BUILD_HOME $GOPATH && \
    dnf -q -y install golang git && \
    dnf -q -y install make && \
    dnf -q -y clean all && \
    mkdir -p $GOPATH/src/github.com/heketi && \
    cd $GOPATH/src/github.com/heketi && \
    git clone -b $HEKETI_BRANCH https://github.com/heketi/heketi.git && \
    cd $GOPATH/src/github.com/heketi/heketi && \
    glide install -v && make && \
    cp heketi /usr/bin/heketi && \
    cp client/cli/go/heketi-cli /usr/bin/heketi-cli && \
    cd && rm -rf $BUILD_HOME && \
    dnf -q -y remove git golang make && \
    dnf -q -y autoremove && \
    dnf -q -y clean all

# post install config and volume setup
ADD ./heketi.json /etc/heketi/heketi.json
ADD ./heketi-start.sh /usr/bin/heketi-start.sh
VOLUME /etc/heketi

RUN mkdir /var/lib/heketi
VOLUME /var/lib/heketi

# expose port, set user and set entrypoint with config option
ENTRYPOINT ["/usr/bin/heketi-start.sh"]
EXPOSE 8080
