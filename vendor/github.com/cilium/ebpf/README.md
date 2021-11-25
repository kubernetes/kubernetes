# eBPF

[![PkgGoDev](https://pkg.go.dev/badge/github.com/cilium/ebpf)](https://pkg.go.dev/github.com/cilium/ebpf)

eBPF is a pure Go library that provides utilities for loading, compiling, and
debugging eBPF programs. It has minimal external dependencies and is intended to
be used in long running processes.

* [asm](https://pkg.go.dev/github.com/cilium/ebpf/asm) contains a basic
  assembler
* [link](https://pkg.go.dev/github.com/cilium/ebpf/link) allows attaching eBPF
  to various hooks
* [perf](https://pkg.go.dev/github.com/cilium/ebpf/perf) allows reading from a
  `PERF_EVENT_ARRAY`
* [cmd/bpf2go](https://pkg.go.dev/github.com/cilium/ebpf/cmd/bpf2go) allows
  compiling and embedding eBPF programs in Go code

The library is maintained by [Cloudflare](https://www.cloudflare.com) and
[Cilium](https://www.cilium.io). Feel free to
[join](https://cilium.herokuapp.com/) the
[#libbpf-go](https://cilium.slack.com/messages/libbpf-go) channel on Slack.

## Current status

The package is production ready, but **the API is explicitly unstable right
now**. Expect to update your code if you want to follow along.

## Getting Started

A small collection of Go and eBPF programs that serve as examples for building
your own tools can be found under [examples/](examples/).

Contributions are highly encouraged, as they highlight certain use cases of
eBPF and the library, and help shape the future of the project.

## Requirements

* A version of Go that is [supported by
  upstream](https://golang.org/doc/devel/release.html#policy)
* Linux 4.9, 4.19 or 5.4 (versions in-between should work, but are not tested)

## Useful resources

* [eBPF.io](https://ebpf.io) (recommended)
* [Cilium eBPF documentation](https://docs.cilium.io/en/latest/bpf/#bpf-guide)
  (recommended)
* [Linux documentation on
  BPF](https://www.kernel.org/doc/html/latest/networking/filter.html)
* [eBPF features by Linux
  version](https://github.com/iovisor/bcc/blob/master/docs/kernel-versions.md)

## Regenerating Testdata

Run `make` in the root of this repository to rebuild testdata in all
subpackages. This requires Docker, as it relies on a standardized build
environment to keep the build output stable.

The toolchain image build files are kept in [testdata/docker/](testdata/docker/).

## License

MIT
