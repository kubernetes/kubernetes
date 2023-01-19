# OpenTelemetry-Go OTLP Span Exporter

[![Go Reference](https://pkg.go.dev/badge/go.opentelemetry.io/otel/exporters/otlp/otlptrace.svg)](https://pkg.go.dev/go.opentelemetry.io/otel/exporters/otlp/otlptrace)

[OpenTelemetry Protocol Exporter](https://github.com/open-telemetry/opentelemetry-specification/blob/v1.5.0/specification/protocol/exporter.md) implementation.

## Installation

```
go get -u go.opentelemetry.io/otel/exporters/otlp/otlptrace
```

## Examples

- [HTTP Exporter setup and examples](./otlptracehttp/example_test.go)
- [Full example of gRPC Exporter sending telemetry to a local collector](../../../example/otel-collector)

## [`otlptrace`](https://pkg.go.dev/go.opentelemetry.io/otel/exporters/otlp/otlptrace)

The `otlptrace` package provides an exporter implementing the OTel span exporter interface.
This exporter is configured using a client satisfying the `otlptrace.Client` interface.
This client handles the transformation of data into wire format and the transmission of that data to the collector.

## [`otlptracegrpc`](https://pkg.go.dev/go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc)

The `otlptracegrpc` package implements a client for the span exporter that sends trace telemetry data to the collector using gRPC with protobuf-encoded payloads.

## [`otlptracehttp`](https://pkg.go.dev/go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp)

The `otlptracehttp` package implements a client for the span exporter that sends trace telemetry data to the collector using HTTP with protobuf-encoded payloads.

## Configuration

### Environment Variables

The following environment variables can be used (instead of options objects) to
override the default configuration. For more information about how each of
these environment variables is interpreted, see [the OpenTelemetry
specification](https://github.com/open-telemetry/opentelemetry-specification/blob/v1.8.0/specification/protocol/exporter.md).

| Environment variable                                                     | Option                        | Default value                                            |
| ------------------------------------------------------------------------ |------------------------------ | -------------------------------------------------------- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`       | `WithEndpoint` `WithInsecure` | `https://localhost:4317` or `https://localhost:4318`[^1] |
| `OTEL_EXPORTER_OTLP_CERTIFICATE` `OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE` | `WithTLSClientConfig`         |                                                          |
| `OTEL_EXPORTER_OTLP_HEADERS` `OTEL_EXPORTER_OTLP_TRACES_HEADERS`         | `WithHeaders`                 |                                                          |
| `OTEL_EXPORTER_OTLP_COMPRESSION` `OTEL_EXPORTER_OTLP_TRACES_COMPRESSION` | `WithCompression`             |                                                          |
| `OTEL_EXPORTER_OTLP_TIMEOUT` `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT`         | `WithTimeout`                 | `10s`                                                    |

[^1]: The gRPC client defaults to `https://localhost:4317` and the HTTP client `https://localhost:4318`.

Configuration using options have precedence over the environment variables.
