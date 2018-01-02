FROM ubuntu:12.04
FROM golang:1.5

ADD . /go/src/github.com/aws/aws-sdk-go

WORKDIR /go/src/github.com/aws/aws-sdk-go
CMD ["make", "get-deps", "unit"]
