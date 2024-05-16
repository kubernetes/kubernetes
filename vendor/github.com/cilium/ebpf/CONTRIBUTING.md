# How to contribute

Development is on [GitHub](https://github.com/cilium/ebpf) and contributions in
the form of pull requests and issues reporting bugs or suggesting new features
are welcome. Please take a look at [the architecture](ARCHITECTURE.md) to get
a better understanding for the high-level goals.

New features must be accompanied by tests. Before starting work on any large
feature, please [join](https://ebpf.io/slack) the
[#ebpf-go](https://cilium.slack.com/messages/ebpf-go) channel on Slack to
discuss the design first.

When submitting pull requests, consider writing details about what problem you
are solving and why the proposed approach solves that problem in commit messages
and/or pull request description to help future library users and maintainers to
reason about the proposed changes.

## Running the tests

Many of the tests require privileges to set resource limits and load eBPF code.
The easiest way to obtain these is to run the tests with `sudo`.

To test the current package with your local kernel you can simply run:
```
go test -exec sudo  ./...
```

To test the current package with a different kernel version you can use the [run-tests.sh](run-tests.sh) script.
It requires [virtme](https://github.com/amluto/virtme) and qemu to be installed.

Examples:

```bash
# Run all tests on a 5.4 kernel
./run-tests.sh 5.4

# Run a subset of tests:
./run-tests.sh 5.4 go test ./link
```

