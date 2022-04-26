ARG GO_VERSION=1.16

FROM golang:${GO_VERSION}-alpine AS builder
RUN apk add --update --no-cache make git curl gcc libc-dev
RUN mkdir -p /build
WORKDIR /build
COPY . /build/
RUN go mod download
RUN go build -o go-mnd cmd/mnd/main.go

FROM golang:${GO_VERSION}-alpine
RUN apk add --update --no-cache bash git gcc libc-dev
COPY --from=builder /build/go-mnd /bin/go-mnd
COPY entrypoint.sh /bin/entrypoint.sh
VOLUME /app
WORKDIR /app
ENTRYPOINT ["/bin/entrypoint.sh"]
