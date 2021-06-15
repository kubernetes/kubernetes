# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

Types of changes:
- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

## [Unreleased]

### Added

- [#223](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/223) Add go-kit logging middleware - [adrien-f](https://github.com/adrien-f)

## [v1.1.0] - 2019-09-12
### Added
- [#226](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/226) Support for go modules.
- [#221](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/221) logging/zap add support for gRPC LoggerV2  - [kush-patel-hs](https://github.com/kush-patel-hs)
- [#181](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/181) Rate Limit support - [ceshihao](https://github.com/ceshihao)
- [#161](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/161) Retry on server stream call - [lonnblad](https://github.com/lonnblad)
- [#152](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/152) Exponential backoff functions - [polyfloyd](https://github.com/polyfloyd)
- [#147](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/147) Jaeger support for ctxtags extraction - [vporoshok](https://github.com/vporoshok)
- [#184](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/184) ctxTags identifies if the call was sampled

### Deprecated
- [#201](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/201) `golang.org/x/net/context` - [houz42](https://github.com/houz42)
- [#183](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/183) Documentation Generation in favour of <godoc.org>.

### Fixed
- [172](https://github.com/grpc-ecosystem/go-grpc-middleware/pull/172) Passing ctx into retry and recover - [johanbrandhorst](https://github.com/johanbrandhorst)
- Numerious documentation fixes.

## v1.0.0 - 2018-05-08
### Added
- grpc_auth 
- grpc_ctxtags
- grpc_zap
- grpc_logrus
- grpc_opentracing
- grpc_retry
- grpc_validator
- grpc_recovery

[Unreleased]: https://github.com/grpc-ecosystem/go-grpc-middleware/compare/v1.1.0...HEAD 
[v1.1.0]: https://github.com/grpc-ecosystem/go-grpc-middleware/compare/v1.0.0...v1.1.0 
