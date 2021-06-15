# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.20.0] - 2021-04-23

### Changed

- The `go.opentelemetry.io/contrib/instrumentation/go.mongodb.org/mongo-driver/mongo/otelmongo` instrumentation now accepts a `WithCommandAttributeDisabled`, 
   so the caller can specify whether to opt-out of tracing the mongo command. (#712)
- Upgrade to v0.20.0 of `go.opentelemetry.io/otel`. (#758)
- The B3 and Jaeger propagators now store their debug or deferred state in the context.Context instead of the SpanContext. (#758)

## [0.19.0] - 2021-03-19

### Changed

- Upgrade to v0.19.0 of `go.opentelemetry.io/otel`.

## [0.18.0] - 2021-03-04

### Fixed

- `otelmemcache` no longer sets span status to OK instead of leaving it unset. (#477)
- Fix goroutine leak in gRPC `StreamClientInterceptor`. (#581)

### Removed

- Remove service name from `otelmemcache` configuration and span attributes. (#477)

## [0.17.0] - 2021-02-15

### Added

- Add `ot-tracer` propagator (#562)

### Changed

- Rename project default branch from `master` to `main`.

### Fixed

- Added failure message for AWS ECS resource detector for better debugging (#568)
- Goroutine leak in gRPC StreamClientInterceptor while streamer returns an error. (#581)

## [0.16.0] - 2021-01-13

### Fixed

- Fix module path for AWS ECS resource detector (#517)

## [0.15.1] - 2020-12-14

### Added

- Add registry link check to `Makefile` and pre-release script. (#446)
- A new AWS X-Ray ID Generator (#459)
- Migrate CircleCI jobs to GitHub Actions (#476)
- Add CodeQL GitHub Action (#506)
- Add gosec workflow to GitHub Actions (#507)

### Fixed

- Fixes the body replacement in otelhttp to not to mutate a nil body. (#484)

## [0.15.0] - 2020-12-11

### Added

- A new Amazon EKS resource detector. (#465)
- A new `gcp.CloudRun` detector for detecting resource from a Cloud Run instance. (#455)

## [0.14.0] - 2020-11-20

### Added

- `otelhttp.{Get,Head,Post,PostForm}` convenience wrappers for their `http` counterparts. (#390)
- The AWS detector now adds the cloud zone, host image ID, host type, and host name to the returned `Resource`. (#410)
- Add Amazon ECS Resource Detector for AWS X-Ray. (#466)
- Add propagator for AWS X-Ray (#462)

### Changed

- Add semantic version to `Tracer` / `Meter` created by instrumentation packages `otelsaram`, `otelrestful`, `otelmongo`, `otelhttp` and `otelhttptrace`. (#412)
- Update instrumentation guidelines about tracer / meter semantic version. (#412)
- Replace internal tracer and meter helpers by helpers from `go.opentelemetry.io/otel`. (#414)
- gRPC instrumentation sets span attribute `rpc.grpc.status_code`. (#453)

## Fixed

- `/detectors/aws` no longer fails if instance metadata is not available (e.g. not running in AWS) (#401)
- The AWS detector now returns a partial resource and an appropriate error if it encounters an error part way through determining a `Resource` identity. (#410)
- The `host` instrumentation unit test has been updated to not depend on the system it runs on. (#426)

## [0.13.0] - 2020-10-09

## Added

- A Jaeger propagator. (#375)

## Changed

- The `go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc` package instrumentation no longer accepts a `Tracer` as an argument to the interceptor function.
   Instead, a new `WithTracerProvider` option is added to configure the `TracerProvider` used when creating the `Tracer` for the instrumentation. (#373)
- The `go.opentelemetry.io/contrib/instrumentation/gopkg.in/macaron.v1/otelmacaron` instrumentation now accepts a `TracerProvider` rather than a `Tracer`. (#374)
- Remove `go.opentelemetry.io/otel/sdk` dependency from instrumentation. (#381)
- Use `httpsnoop` in `go.opentelemetry.io/contrib/instrumentation/github.com/gorilla/mux/otelmux` to ensure `http.ResponseWriter` additional interfaces are preserved. (#388)

### Fixed

- The `go.opentelemetry.io/contrib/instrumentation/github.com/labstack/echo/otelecho.Middleware` no longer sends duplicate errors to the global `ErrorHandler`. (#377, #364)
- The import comment in `go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp` is now correctly quoted. (#379)
- The B3 propagator sets the sample bitmask when the sampling decision is `debug`. (#369)

## [0.12.0] - 2020-09-25

### Changed

- Replace `WithTracer` with `WithTracerProvider` in the `go.opentelemetry.io/contrib/instrumentation/gopkg.in/macaron.v1/otelmacaron` instrumentation. (#374)

### Added

- Benchmark tests for the gRPC instrumentation. (#296)
- Integration testing for the gRPC instrumentation. (#297)
- Allow custom labels to be added to net/http metrics. (#306)
- Added B3 propagator, moving it out of open.telemetry.io/otel repo. (#344)

### Changed

- Unify instrumentation about provider options for `go.mongodb.org/mongo-driver`, `gin-gonic/gin`, `gorilla/mux`,
  `labstack/echo`, `emicklei/go-restful`, `bradfitz/gomemcache`, `Shopify/sarama`, `net/http` and `beego`. (#303)
- Update instrumentation guidelines about uniform provider options. Also, update style guide. (#303)
- Make config struct of instrumentation unexported. (#303)
- Instrumentations have been updated to adhere to the [configuration style guide's](https://github.com/open-telemetry/opentelemetry-go/blob/master/CONTRIBUTING.md#config)
   updated recommendation to use `newConfig()` instead of `configure()`. (#336)
- A new instrumentation naming scheme is implemented to avoid package name conflicts for instrumented packages while still remaining discoverable. (#359)
  - `google.golang.org/grpc` -> `google.golang.org/grpc/otelgrpc`
  - `go.mongodb.org/mongo-driver` -> `go.mongodb.org/mongo-driver/mongo/otelmongo`
  - `net/http` -> `net/http/otelhttp`
  - `net/http/httptrace` -> `net/http/httptrace/otelhttptrace`
  - `github.com/labstack/echo` -> `github.com/labstack/echo/otelecho`
  - `github.com/bradfitz/gomemcache` -> `github.com/bradfitz/gomemcache/memcache/otelmemcache`
  - `github.com/gin-gonic/gin` -> `github.com/gin-gonic/gin/otelgin`
  - `github.com/gocql/gocql` -> `github.com/gocql/gocql/otelgocql`
  - `github.com/emicklei/go-restful` -> `github.com/emicklei/go-restful/otelrestful`
  - `github.com/Shopify/sarama` -> `github.com/Shopify/sarama/otelsarama`
  - `github.com/gorilla/mux` -> `github.com/gorilla/mux/otelmux`
  - `github.com/astaxie/beego` -> `github.com/astaxie/beego/otelbeego`
  - `gopkg.in/macaron.v1` -> `gopkg.in/macaron.v1/otelmacaron`
- Rename `OTelBeegoHandler` to `Handler` in the `go.opentelemetry.io/contrib/instrumentation/github.com/astaxie/beego/otelbeego` package. (#359)

## [0.11.0] - 2020-08-25

### Added

- Top-level `Version()` and `SemVersion()` functions defining the current version of the contrib package. (#225)
- Instrumentation for the `github.com/astaxie/beego` package. (#200)
- Instrumentation for the `github.com/bradfitz/gomemcache` package. (#204)
- Host metrics instrumentation. (#231)
- Cortex histogram and distribution support. (#237)
- Cortex example project. (#238)
- Cortex HTTP authentication. (#246)

### Changed

- Remove service name as a parameter of Sarama instrumentation. (#221)
- Replace `WithTracer` with `WithTracerProvider` in Sarama instrumentation. (#221)
- Switch to use common top-level module `SemVersion()` when creating versioned tracer in `bradfitz/gomemcache`. (#226)
- Use `IntegrationShouldRun` in `gomemcache_test`. (#254)
- Use Go 1.15 for CI builds. (#236)
- Improved configuration for `runtime` instrumentation. (#224)

### Fixed

- Update dependabot configuration to include newly added `bradfitz/gomemcache` package. (#226)
- Correct `runtime` instrumentation name. (#241)

## [0.10.1] - 2020-08-13

### Added

- The `go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc` module has been added to replace the instrumentation that had previoiusly existed in the `go.opentelemetry.io/otel/instrumentation/grpctrace` package. (#189)
- Instrumentation for the stdlib `net/http` and `net/http/httptrace` packages. (#190)
- Initial Cortex exporter. (#202, #205, #210, #211, #215)

### Fixed

- Bump google.golang.org/grpc from 1.30.0 to 1.31.0. (#166)
- Bump go.mongodb.org/mongo-driver from 1.3.5 to 1.4.0 in /instrumentation/go.mongodb.org/mongo-driver. (#170)
- Bump google.golang.org/grpc in /instrumentation/github.com/gin-gonic/gin. (#173)
- Bump google.golang.org/grpc in /instrumentation/github.com/labstack/echo. (#176)
- Bump google.golang.org/grpc from 1.30.0 to 1.31.0 in /instrumentation/github.com/Shopify/sarama. (#179)
- Bump cloud.google.com/go from 0.61.0 to 0.63.0 in /detectors/gcp. (#181, #199)
- Bump github.com/aws/aws-sdk-go from 1.33.15 to 1.34.1 in /detectors/aws. (#184, #192, #193, #198, #201, #203)
- Bump github.com/golangci/golangci-lint from 1.29.0 to 1.30.0 in /tools. (#186)
- Setup CI to run tests that require external resources (Cassandra and MongoDB). (#191)
- Bump github.com/Shopify/sarama from 1.26.4 to 1.27.0 in /instrumentation/github.com/Shopify/sarama. (#206)

## [0.10.0] - 2020-07-31

This release upgrades its [go.opentelemetry.io/otel](https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.10.0) dependency to v0.10.0 and includes new instrumentation for popular Kafka and Cassandra clients.

### Added

- A detector that generate resources from GCE instance. (#132)
- A detector that generate resources from AWS instances. (#139)
- Instrumentation for the Kafka client github.com/Shopify/sarama. (#134, #153)
- Links and status message for mock span in the internal testing library. (#134)
- Instrumentation for the Cassandra client github.com/gocql/gocql. (#137)
- A detector that generate resources from GKE clusters. (#154)

### Fixed

- Bump github.com/aws/aws-sdk-go from 1.33.8 to 1.33.15 in /detectors/aws. (#155, #157, #159, #162)
- Bump github.com/golangci/golangci-lint from 1.28.3 to 1.29.0 in /tools. (#146)

## [0.9.0] - 2020-07-20

This release upgrades its [go.opentelemetry.io/otel](https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.9.0) dependency to v0.9.0.

### Fixed

- Bump github.com/emicklei/go-restful/v3 from 3.0.0 to 3.2.0 in /instrumentation/github.com/emicklei/go-restful. (#133)
- Update dependabot configuration to correctly check all included packages. (#131)
- Update `RELEASING.md` with correct `tag.sh` command. (#130)

## [0.8.0] - 2020-07-10

This release upgrades its [go.opentelemetry.io/otel](https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.8.0) dependency to v0.8.0, includes minor fixes, and new instrumentation.

### Added

- Create this `CHANGELOG.md`. (#114)
- Add `emicklei/go-restful/v3` trace instrumentation. (#115)

### Changed

- Update `CONTRIBUTING.md` to ask for updates to `CHANGELOG.md` with each pull request. (#114)
- Move all `github.com` package instrumentation under a `github.com` directory. (#118)

### Fixed

- Update README to include information about external instrumentation.
   To start, this includes native instrumentation found in the `go-redis/redis` package. (#117)
- Bump github.com/golangci/golangci-lint from 1.27.0 to 1.28.2 in /tools. (#122, #123, #125)
- Bump go.mongodb.org/mongo-driver from 1.3.4 to 1.3.5 in /instrumentation/go.mongodb.org/mongo-driver. (#124)

## [0.7.0] - 2020-06-29

This release upgrades its [go.opentelemetry.io/otel](https://github.com/open-telemetry/opentelemetry-go/releases/tag/v0.7.0) dependency to v0.7.0.

### Added

- Create `RELEASING.md` instructions. (#101)
- Apply transitive dependabot go.mod updates as part of a new automatic Github workflow. (#94)
- New dependabot integration to automate package upgrades. (#61)
- Add automatic tag generation script for release. (#60)

### Changed

- Upgrade Datadog metrics exporter to include Resource tags. (#46)
- Added output validation to Datadog example. (#96)
- Move Macaron package to match layout guidelines. (#92)
- Update top-level README and instrumentation README. (#92)
- Bump google.golang.org/grpc from 1.29.1 to 1.30.0. (#99)
- Bump github.com/golangci/golangci-lint from 1.21.0 to 1.27.0 in /tools. (#77)
- Bump go.mongodb.org/mongo-driver from 1.3.2 to 1.3.4 in /instrumentation/go.mongodb.org/mongo-driver. (#76)
- Bump github.com/stretchr/testify from 1.5.1 to 1.6.1. (#74)
- Bump gopkg.in/macaron.v1 from 1.3.5 to 1.3.9 in /instrumentation/macaron. (#68)
- Bump github.com/gin-gonic/gin from 1.6.2 to 1.6.3 in /instrumentation/gin-gonic/gin. (#73)
- Bump github.com/DataDog/datadog-go from 3.5.0+incompatible to 3.7.2+incompatible in /exporters/metric/datadog. (#78)
- Replaced `internal/trace/http.go` helpers with `api/standard` helpers from otel-go repo. (#112)

## [0.6.1] - 2020-06-08

First official tagged release of `contrib` repository.

### Added

- `labstack/echo` trace instrumentation (#42)
- `mongodb` trace instrumentation (#26)
- Go Runtime metrics (#9)
- `gorilla/mux` trace instrumentation (#19)
- `gin-gonic` trace instrumentation (#15)
- `macaron` trace instrumentation (#20)
- `dogstatsd` metrics exporter (#10)
- `datadog` metrics exporter (#22)
- Tags to all modules in repository
- Repository folder structure and automated build (#3)

### Changes

- Prefix support for dogstatsd (#34)
- Update Go Runtime package to use batch observer (#44)

[Unreleased]: https://github.com/open-telemetry/opentelemetry-go-contrib/compare/v0.20.0...HEAD
[0.20.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.20.0
[0.19.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.19.0
[0.18.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.18.0
[0.17.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.17.0
[0.16.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.16.0
[0.15.1]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.15.1
[0.15.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.15.0
[0.14.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.14.0
[0.13.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.13.0
[0.12.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.12.0
[0.11.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.11.0
[0.10.1]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.10.1
[0.10.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.10.0
[0.9.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.9.0
[0.8.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.8.0
[0.7.0]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.7.0
[0.6.1]: https://github.com/open-telemetry/opentelemetry-go-contrib/releases/tag/v0.6.1
