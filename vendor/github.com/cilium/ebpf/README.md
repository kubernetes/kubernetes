# eBPF

[![PkgGoDev](https://pkg.go.dev/badge/github.com/cilium/ebpf)](https://pkg.go.dev/github.com/cilium/ebpf)

![HoneyGopher](.github/images/cilium-ebpf.png)

ebpf-go is a pure Go library that provides utilities for loading, compiling, and
debugging eBPF programs. It has minimal external dependencies and is intended to
be used in long running processes.

See [ebpf.io](https://ebpf.io) for complementary projects from the wider eBPF
ecosystem.

## Getting Started

A small collection of Go and eBPF programs that serve as examples for building
your own tools can be found under [examples/](examples/).

[Contributions](CONTRIBUTING.md) are highly encouraged, as they highlight certain use cases of
eBPF and the library, and help shape the future of the project.

## Getting Help

The community actively monitors our [GitHub Discussions](https://github.com/cilium/ebpf/discussions) page.
Please search for existing threads before starting a new one. Refrain from
opening issues on the bug tracker if you're just starting out or if you're not
sure if something is a bug in the library code.

Alternatively, [join](https://ebpf.io/slack) the
[#ebpf-go](https://cilium.slack.com/messages/ebpf-go) channel on Slack if you
have other questions regarding the project. Note that this channel is ephemeral
and has its history erased past a certain point, which is less helpful for
others running into the same problem later.

## Packages

This library includes the following packages:

* [asm](https://pkg.go.dev/github.com/cilium/ebpf/asm) contains a basic
  assembler, allowing you to write eBPF assembly instructions directly
  within your Go code. (You don't need to use this if you prefer to write your eBPF program in C.)
* [cmd/bpf2go](https://pkg.go.dev/github.com/cilium/ebpf/cmd/bpf2go) allows
  compiling and embedding eBPF programs written in C within Go code. As well as
  compiling the C code, it auto-generates Go code for loading and manipulating
  the eBPF program and map objects.
* [link](https://pkg.go.dev/github.com/cilium/ebpf/link) allows attaching eBPF
  to various hooks
* [perf](https://pkg.go.dev/github.com/cilium/ebpf/perf) allows reading from a
  `PERF_EVENT_ARRAY`
* [ringbuf](https://pkg.go.dev/github.com/cilium/ebpf/ringbuf) allows reading from a
  `BPF_MAP_TYPE_RINGBUF` map
* [features](https://pkg.go.dev/github.com/cilium/ebpf/features) implements the equivalent
  of `bpftool feature probe` for discovering BPF-related kernel features using native Go.
* [rlimit](https://pkg.go.dev/github.com/cilium/ebpf/rlimit) provides a convenient API to lift
  the `RLIMIT_MEMLOCK` constraint on kernels before 5.11.
* [btf](https://pkg.go.dev/github.com/cilium/ebpf/btf) allows reading the BPF Type Format.

## Requirements

* A version of Go that is [supported by
  upstream](https://golang.org/doc/devel/release.html#policy)
* Linux >= 4.9. CI is run against kernel.org LTS releases. 4.4 should work but is
  not tested against.

## Regenerating Testdata

Run `make` in the root of this repository to rebuild testdata in all
subpackages. This requires Docker, as it relies on a standardized build
environment to keep the build output stable.

It is possible to regenerate data using Podman by overriding the `CONTAINER_*`
variables: `CONTAINER_ENGINE=podman CONTAINER_RUN_ARGS= make`.

The toolchain image build files are kept in [testdata/docker/](testdata/docker/).

## License

MIT

### eBPF Gopher

The eBPF honeygopher is based on the Go gopher designed by Renee French.
