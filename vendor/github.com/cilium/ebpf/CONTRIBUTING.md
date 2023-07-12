# How to contribute

Development is on [GitHub](https://github.com/cilium/ebpf) and contributions in
the form of pull requests and issues reporting bugs or suggesting new features
are welcome. Please take a look at [the architecture](ARCHITECTURE.md) to get
a better understanding for the high-level goals.

## Adding a new feature

1. [Join](https://ebpf.io/slack) the
[#ebpf-go](https://cilium.slack.com/messages/ebpf-go) channel to discuss your requirements and how the feature can be implemented. The most important part is figuring out how much new exported API is necessary. **The less new API is required the easier it will be to land the feature.**
2. (*optional*) Create a draft PR if you want to discuss the implementation or have hit a problem. It's fine if this doesn't compile or contains debug statements.
3. Create a PR that is ready to merge. This must pass CI and have tests.

### API stability

The library doesn't guarantee the stability of its API at the moment.

1. If possible avoid breakage by introducing new API and deprecating the old one
   at the same time. If an API was deprecated in v0.x it can be removed in v0.x+1.
2. Breaking API in a way that causes compilation failures is acceptable but must
   have good reasons.
3. Changing the semantics of the API without causing compilation failures is
   heavily discouraged.

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
./run-tests.sh 5.4 ./link
```

