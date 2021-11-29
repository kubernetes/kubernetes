# OpenTelemetry Collector Go Exporter

[![PkgGoDev](https://pkg.go.dev/badge/go.opentelemetry.io/otel/exporters/otlp)](https://pkg.go.dev/go.opentelemetry.io/otel/exporters/otlp)

This exporter exports OpenTelemetry spans and metrics to the OpenTelemetry Collector.


## Installation and Setup

The exporter can be installed using standard `go` functionality.

```bash
$ go get -u go.opentelemetry.io/otel/exporters/otlp
```

A new exporter can be created using the `NewExporter` function.

## Retries

The exporter will not, by default, retry failed requests to the collector.
However, it is configured in a way that it can be easily enabled.

To enable retries, the `GRPC_GO_RETRY` environment variable needs to be set to `on`. For example,

```
GRPC_GO_RETRY=on go run .
```

The [default service config](https://github.com/grpc/proposal/blob/master/A6-client-retries.md) used by default is defined to retry failed requests with exponential backoff (`0.3seconds * (2)^retry`) with [a max of `5` retries](https://github.com/open-telemetry/oteps/blob/be2a3fcbaa417ebbf5845cd485d34fdf0ab4a2a4/text/0035-opentelemetry-protocol.md#export-response)).

These retries are only attempted for reponses that are [deemed "retry-able" by the collector](https://github.com/grpc/proposal/blob/master/A6-client-retries.md#validation-of-retrypolicy).
