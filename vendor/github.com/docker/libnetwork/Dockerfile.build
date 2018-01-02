FROM golang:1.8.3
RUN apt-get update && apt-get -y install iptables

RUN go get github.com/tools/godep \
		github.com/golang/lint/golint \
		golang.org/x/tools/cmd/cover \
		github.com/mattn/goveralls \
		github.com/gordonklaus/ineffassign \
		github.com/client9/misspell/cmd/misspell \
		honnef.co/go/tools/cmd/gosimple
