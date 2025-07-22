# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0](https://github.com/grpc-ecosystem/go-grpc-prometheus/releases/tag/v1.2.0) - 2018-06-04

### Added

* Provide metrics object as `prometheus.Collector`, for conventional metric registration.
* Support non-default/global Prometheus registry.
* Allow configuring counters with `prometheus.CounterOpts`.

### Changed

* Remove usage of deprecated `grpc.Code()`.
* Remove usage of deprecated `grpc.Errorf` and replace with `status.Errorf`.

---

This changelog was started with version `v1.2.0`, for earlier versions refer to the respective [GitHub releases](https://github.com/grpc-ecosystem/go-grpc-prometheus/releases).
