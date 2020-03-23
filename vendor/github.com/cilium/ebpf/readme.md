eBPF
-------
[![](https://godoc.org/github.com/cilium/ebpf?status.svg)](https://godoc.org/github.com/cilium/ebpf)

eBPF is a pure Go library that provides utilities for loading, compiling, and debugging eBPF programs. It has minimal external dependencies and is intended to be used in long running processes.

[ebpf/asm](https://godoc.org/github.com/cilium/ebpf/asm) contains a basic assembler.

The library is maintained by [Cloudflare](https://www.cloudflare.com) and [Cilium](https://www.cilium.io). Feel free to [join](https://cilium.herokuapp.com/) the [libbpf-go](https://cilium.slack.com/messages/libbpf-go) channel on Slack.

## Current status

The package is production ready, but **the API is explicitly unstable
right now**. Expect to update your code if you want to follow along.

## Useful resources

* [Cilium eBPF documentation](https://cilium.readthedocs.io/en/latest/bpf/#bpf-guide) (recommended)
* [Linux documentation on BPF](http://elixir.free-electrons.com/linux/latest/source/Documentation/networking/filter.txt)
* [eBPF features by Linux version](https://github.com/iovisor/bcc/blob/master/docs/kernel-versions.md)
