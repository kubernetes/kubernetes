# How to contribute

Development is on [GitHub](https://github.com/cilium/ebpf) and contributions in
the form of pull requests and issues reporting bugs or suggesting new features
are welcome. Please take a look at [the architecture](ARCHITECTURE.md) to get
a better understanding for the high-level goals.

New features must be accompanied by tests. Before starting work on any large
feature, please [join](https://cilium.herokuapp.com/) the
[#libbpf-go](https://cilium.slack.com/messages/libbpf-go) channel on Slack to
discuss the design first.

When submitting pull requests, consider writing details about what problem you
are solving and why the proposed approach solves that problem in commit messages
and/or pull request description to help future library users and maintainers to
reason about the proposed changes.

## Running the tests

Many of the tests require privileges to set resource limits and load eBPF code.
The easiest way to obtain these is to run the tests with `sudo`:

    sudo go test ./...