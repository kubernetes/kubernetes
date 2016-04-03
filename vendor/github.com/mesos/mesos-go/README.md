# Go bindings for Apache Mesos

Very early version of a pure Go language bindings for Apache Mesos. As with other pure implementation, mesos-go uses the HTTP wire protocol to communicate directly with  a running Mesos master and its slave instances. One of the objectives of this project is to provide an idiomatic Go API that makes it super easy to create Mesos frameworks using Go. 

[![Build Status](https://travis-ci.org/mesos/mesos-go.svg)](https://travis-ci.org/mesos/mesos-go) [![GoDoc] (https://godoc.org/github.com/mesos/mesos-go?status.png)](https://godoc.org/github.com/mesos/mesos-go)

## Status
This project is undergoing a *complete* rewrite happening in the
[next](https://github.com/mesos/mesos-go/tree/next) branch. These bindings will
integrate exclusively with the new public Mesos HTTP API.

The current version of the bindings are considered **alpha** and won't
see any major development besides critical compatibility and bug fixes.

We use [semantic versioning](http://semver.org/).

### Compatibility
`mesos-N` tags mark the start of support for a specific Mesos version while
maintaining backwards compatibility with the previous major version.

### Features
- The SchedulerDriver API implemented
- The ExecutorDriver API implemented
- Stable API (based on the core Mesos code)
- Plenty of unit and integrative of tests
- Modular design for easy readability/extensibility
- Example programs on how to use the API
- Leading master detection
- Authentication via SASL/CRAM-MD5

### Pre-Requisites
- Go 1.3 or higher
- A standard and working Go workspace setup
- Apache Mesos 0.19 or newer

## Installing
Users of this library are encouraged to vendor it. API stability isn't guaranteed
at this stage.
```shell
$ go get github.com/mesos/mesos-go
```

## Testing
```shell
$ go test -race ./...
```

## Contributing
Contributions are welcome. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines.

## License
This project is [Apache License 2.0](LICENSE).
