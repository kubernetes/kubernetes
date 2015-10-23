FROM google/golang:latest

ADD . /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/guestbook-go/_src

WORKDIR /gopath/src/github.com/GoogleCloudPlatform/kubernetes/examples/guestbook-go/
RUN cd _src/ && go get && go build -o ../bin/guestbook
RUN cp _src/guestbook/Dockerfile .

CMD tar cvzf - .
