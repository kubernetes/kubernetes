ARG GO_VERSION=1.14.2
ARG ALPINE_VERSION=3.11
ARG gomodguard_VERSION=

# ---- Build container
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION} AS builder
WORKDIR /gomodguard
COPY . .
RUN apk add --no-cache git
RUN go build -o gomodguard cmd/gomodguard/main.go

# ---- App container
FROM golang:${GO_VERSION}-alpine${ALPINE_VERSION}
WORKDIR /
RUN apk --no-cache add ca-certificates
COPY --from=builder gomodguard/gomodguard /
ENTRYPOINT ./gomodguard
