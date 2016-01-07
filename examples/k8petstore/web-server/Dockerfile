FROM golang:latest

# Add source to gopath.  This is defacto required for go apps.
ADD ./src /gopath/src/k8petstore
RUN mkdir /gopath/bin/
ADD ./static /tmp/static
ADD ./test.sh /opt/test.sh
RUN chmod 777 /opt/test.sh

# So that we can easily run and install
WORKDIR /gopath/src

# Install the code (the executables are in the main dir)  This will get the deps also.
RUN export GOPATH=/gopath/ && go get k8petstore
RUN export GOPATH=/gopath/ && go install k8petstore


# Expected that you will override this in production kubernetes.
ENV STATIC_FILES /tmp/static
CMD /gopath/bin/k8petstore
