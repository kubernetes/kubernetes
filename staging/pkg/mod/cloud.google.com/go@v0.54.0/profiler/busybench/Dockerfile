FROM golang:alpine

WORKDIR /go/src/busybench

COPY *.go .

EXPOSE 8080

RUN apk update \
    && apk add --no-cache git \
    && go get -d -v ./... \
    && apk del git

RUN go install -v ./...

CMD ["busybench", "--service", "busybench", "--service_version", "1.0.2", "--duration", "0", "--mutex_profiling", "true"]
