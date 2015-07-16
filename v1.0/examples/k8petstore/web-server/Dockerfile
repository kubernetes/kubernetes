FROM google/golang:latest

# Add source to gopath.  This is defacto required for go apps.
ADD ./src /gopath/src/
ADD ./static /tmp/static
ADD ./test.sh /opt/test.sh
RUN chmod 777 /opt/test.sh
# $GOPATH/[src/a/b/c]
# go build a/b/c
# go run main

# So that we can easily run and install
WORKDIR /gopath/src/

# Install the code (the executables are in the main dir)  This will get the deps also.
RUN go get main
#RUN go build main

# Expected that you will override this in production kubernetes.
ENV STATIC_FILES /tmp/static
CMD /gopath/bin/main
