ARG GO_VERSION
FROM golang:${GO_VERSION}-alpine AS builder
RUN apk add --update --no-cache ca-certificates make git curl gcc libc-dev
RUN mkdir -p /build
WORKDIR /build
COPY . /build/
RUN go mod download
RUN make build-linux

FROM golang:${GO_VERSION}-alpine 
RUN apk add --update --no-cache ca-certificates bash git gcc libc-dev openssh
ENV GO111MODULE on
COPY --from=builder /build/gosec /bin/gosec
COPY entrypoint.sh /bin/entrypoint.sh
ENTRYPOINT ["/bin/entrypoint.sh"]
