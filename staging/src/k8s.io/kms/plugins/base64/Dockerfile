FROM golang:1.20.1-bullseye as builder

WORKDIR /workspace

# Copy the source
COPY apimachinery/ apimachinery/
COPY kms/ kms/
WORKDIR /workspace/kms/plugins/base64

ARG TARGETARCH
ARG TARGETPLATFORM
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} GO111MODULE=on go build -a -o base64-plugin plugin.go
RUN chmod +x base64-plugin

# Use distroless as minimal base image to package the manager binary
# Refer to https://github.com/GoogleContainerTools/distroless for more details
FROM --platform=${TARGETPLATFORM:-linux/amd64} gcr.io/distroless/static:nonroot
WORKDIR /
COPY --from=builder /workspace/kms/plugins/base64/base64-plugin .

ENTRYPOINT [ "/base64-plugin" ]
