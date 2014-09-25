FROM golang:1.3

RUN go get github.com/tools/godep

WORKDIR /go/src/github.com/GoogleCloudPlatform/kubernetes
ADD Godeps/Godeps.json $GOPATH/src/github.com/GoogleCloudPlatform/kubernetes/Godeps/Godeps.json
RUN godep restore
ADD . $GOPATH/src/github.com/GoogleCloudPlatform/kubernetes
RUN CGO_ENABLED=0 godep go install -a -ldflags '-s' -tags netgo github.com/GoogleCloudPlatform/kubernetes/cmd/...
RUN echo \
'FROM scratch\n'\
'ADD cmd /bin/cmd\n'\
'ENTRYPOINT ["/bin/cmd"]' > Dockerfile.cmd
CMD tar cvzhf - $GOPATH/bin/$cmd Dockerfile.cmd --transform="s,Dockerfile.cmd,Dockerfile,;s,go/bin/$cmd,cmd," --show-transformed-names
