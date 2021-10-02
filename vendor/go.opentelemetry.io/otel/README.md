# OpenTelemetry-Go

[![CI](https://github.com/open-telemetry/opentelemetry-go/workflows/ci/badge.svg)](https://github.com/open-telemetry/opentelemetry-go/actions?query=workflow%3Aci+branch%3Amain)
[![PkgGoDev](https://pkg.go.dev/badge/go.opentelemetry.io/otel)](https://pkg.go.dev/go.opentelemetry.io/otel)
[![Go Report Card](https://goreportcard.com/badge/go.opentelemetry.io/otel)](https://goreportcard.com/report/go.opentelemetry.io/otel)
[![Slack](https://img.shields.io/badge/slack-@cncf/otel--go-brightgreen.svg?logo=slack)](https://cloud-native.slack.com/archives/C01NPAXACKT)


The Go [OpenTelemetry](https://opentelemetry.io/) implementation.

## Project Status

**Warning**: this project is currently in a pre-GA phase. Backwards
incompatible changes may be introduced in subsequent minor version releases as
we work to track the evolving OpenTelemetry specification and user feedback.

Our progress towards a GA release candidate is tracked in [this project
board](https://github.com/orgs/open-telemetry/projects/5). This release
candidate will follow semantic versioning and will be released with a major
version greater than zero.

Progress and status specific to this repository is tracked in our local
[project boards](https://github.com/open-telemetry/opentelemetry-go/projects)
and
[milestones](https://github.com/open-telemetry/opentelemetry-go/milestones).

Project versioning information and stability guarantees can be found in the
[versioning documentation](./VERSIONING.md).

### Compatibility

This project is tested on the following systems.

| OS      | Go Version | Architecture |
| ------- | ---------- | ------------ |
| Ubuntu  | 1.15       | amd64        |
| Ubuntu  | 1.14       | amd64        |
| Ubuntu  | 1.15       | 386          |
| Ubuntu  | 1.14       | 386          |
| MacOS   | 1.15       | amd64        |
| MacOS   | 1.14       | amd64        |
| Windows | 1.15       | amd64        |
| Windows | 1.14       | amd64        |
| Windows | 1.15       | 386          |
| Windows | 1.14       | 386          |

While this project should work for other systems, no compatibility guarantees
are made for those systems currently.

## Getting Started

You can find a getting started guide on [opentelemetry.io](https://opentelemetry.io/docs/go/getting-started/).

OpenTelemetry's goal is to provide a single set of APIs to capture distributed
traces and metrics from your application and send them to an observability
platform. This project allows you to do just that for applications written in
Go. There are two steps to this process: instrument your application, and
configure an exporter.

### Instrumentation

To start capturing distributed traces and metric events from your application
it first needs to be instrumented. The easiest way to do this is by using an
instrumentation library for your code. Be sure to check out [the officially
supported instrumentation
libraries](https://github.com/open-telemetry/opentelemetry-go-contrib/tree/main/instrumentation).

If you need to extend the telemetry an instrumentation library provides or want
to build your own instrumentation for your application directly you will need
to use the
[go.opentelemetry.io/otel/api](https://pkg.go.dev/go.opentelemetry.io/otel/api)
package. The included [examples](./example/) are a good way to see some
practical uses of this process.

### Export

Now that your application is instrumented to collect telemetry, it needs an
export pipeline to send that telemetry to an observability platform.

You can find officially supported exporters [here](./exporters/) and in the
companion [contrib
repository](https://github.com/open-telemetry/opentelemetry-go-contrib/tree/main/exporters/metric).
Additionally, there are many vendor specific or 3rd party exporters for
OpenTelemetry. These exporters are broken down by
[trace](https://pkg.go.dev/go.opentelemetry.io/otel/sdk/export/trace?tab=importedby)
and
[metric](https://pkg.go.dev/go.opentelemetry.io/otel/sdk/export/metric?tab=importedby)
support.

## Contributing

See the [contributing documentation](CONTRIBUTING.md).
