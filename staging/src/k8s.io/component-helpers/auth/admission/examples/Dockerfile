# Minimal multi-stage Dockerfile for examples/main.go.
# Deployment env vars (such as WEBHOOK_LOVE_KITTENS_ISSUER and
# WEBHOOK_LOVE_KITTENS_AUDIENCE in out-of-cluster.yaml) and projected
# ServiceAccount files (in-cluster.yaml) are injected at runtime by Kubernetes.
FROM golang:1.27 AS build
WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /out/kittens-webhook ./examples

FROM gcr.io/distroless/static:nonroot
USER nonroot:nonroot
COPY --from=build /out/kittens-webhook /kittens-webhook
EXPOSE 8443
ENTRYPOINT ["/kittens-webhook"]
