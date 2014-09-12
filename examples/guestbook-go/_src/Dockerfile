FROM google/golang:latest

RUN mkdir -p /gopath/src/github.com/GoogleCloudPlatform/ && cd /gopath/src/github.com/GoogleCloudPlatform/ && \
    git clone http://github.com/GoogleCloudPlatform/kubernetes && \
    cd /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/guestbook-go/src/ && \
    go get && go build -o ../bin/guestbook && \
    cp ./guestbook/Dockerfile /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/guestbook-go/

CMD docker build --rm --force-rm -t kubernetes/guestbook /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/guestbook-go/
