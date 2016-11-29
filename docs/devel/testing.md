# Testing guide

Updated: 5/21/2016

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Testing guide](#testing-guide)
  - [Unit tests](#unit-tests)
    - [Run all unit tests](#run-all-unit-tests)
    - [Set go flags during unit tests](#set-go-flags-during-unit-tests)
    - [Run unit tests from certain packages](#run-unit-tests-from-certain-packages)
    - [Run specific unit test cases in a package](#run-specific-unit-test-cases-in-a-package)
    - [Stress running unit tests](#stress-running-unit-tests)
    - [Unit test coverage](#unit-test-coverage)
    - [Benchmark unit tests](#benchmark-unit-tests)
  - [Integration tests](#integration-tests)
    - [Install etcd dependency](#install-etcd-dependency)
    - [Etcd test data](#etcd-test-data)
    - [Run integration tests](#run-integration-tests)
    - [Run a specific integration test](#run-a-specific-integration-test)
  - [End-to-End tests](#end-to-end-tests)

<!-- END MUNGE: GENERATED_TOC -->

This assumes you already read the [development guide](development.md) to
install go, godeps, and configure your git client.  All command examples are
relative to the `kubernetes` root directory.

Before sending pull requests you should at least make sure your changes have
passed both unit and integration tests.

Kubernetes only merges pull requests when unit, integration, and e2e tests are
passing, so it is often a good idea to make sure the e2e tests work as well.

## Unit tests

* Unit tests should be fully hermetic
  - Only access resources in the test binary.
* All packages and any significant files require unit tests.
* The preferred method of testing multiple scenarios or input is
  [table driven testing](https://github.com/golang/go/wiki/TableDrivenTests)
  - Example: [TestNamespaceAuthorization](../../test/integration/auth/auth_test.go)
* Unit tests must pass on OS X and Windows platforms.
  - Tests using linux-specific features must be skipped or compiled out.
  - Skipped is better, compiled out is required when it won't compile.
* Concurrent unit test runs must pass.
* See [coding conventions](coding-conventions.md).

### Run all unit tests

`make test` is the entrypoint for running the unit tests that ensures that
`GOPATH` is set up correctly.  If you have `GOPATH` set up correctly, you can
also just use `go test` directly.

```sh
cd kubernetes
make test  # Run all unit tests.
```

### Set go flags during unit tests

You can set [go flags](https://golang.org/cmd/go/) by setting the
`KUBE_GOFLAGS` environment variable.

### Run unit tests from certain packages

`make test` accepts packages as arguments; the `k8s.io/kubernetes` prefix is
added automatically to these:

```sh
make test WHAT=pkg/api                # run tests for pkg/api
```

To run multiple targets you need quotes:

```sh
make test WHAT="pkg/api pkg/kubelet"  # run tests for pkg/api and pkg/kubelet
```

In a shell, it's often handy to use brace expansion:

```sh
make test WHAT=pkg/{api,kubelet}  # run tests for pkg/api and pkg/kubelet
```

### Run specific unit test cases in a package

You can set the test args using the `KUBE_TEST_ARGS` environment variable.
You can use this to pass the `-run` argument to `go test`, which accepts a
regular expression for the name of the test that should be run.

```sh
# Runs TestValidatePod in pkg/api/validation with the verbose flag set
make test WHAT=pkg/api/validation KUBE_GOFLAGS="-v" KUBE_TEST_ARGS='-run ^TestValidatePod$'

# Runs tests that match the regex ValidatePod|ValidateConfigMap in pkg/api/validation
make test WHAT=pkg/api/validation KUBE_GOFLAGS="-v" KUBE_TEST_ARGS="-run ValidatePod\|ValidateConfigMap$"
```

For other supported test flags, see the [golang
documentation](https://golang.org/cmd/go/#hdr-Description_of_testing_flags).

### Stress running unit tests

Running the same tests repeatedly is one way to root out flakes.
You can do this efficiently.

```sh
# Have 2 workers run all tests 5 times each (10 total iterations).
make test PARALLEL=2 ITERATION=5
```

For more advanced ideas please see [flaky-tests.md](flaky-tests.md).

### Unit test coverage

Currently, collecting coverage is only supported for the Go unit tests.

To run all unit tests and generate an HTML coverage report, run the following:

```sh
make test KUBE_COVER=y
```

At the end of the run, an HTML report will be generated with the path
printed to stdout.

To run tests and collect coverage in only one package, pass its relative path
under the `kubernetes` directory as an argument, for example:

```sh
make test WHAT=pkg/kubectl KUBE_COVER=y
```

Multiple arguments can be passed, in which case the coverage results will be
combined for all tests run.

### Benchmark unit tests

To run benchmark tests, you'll typically use something like:

```sh
go test ./pkg/apiserver -benchmem -run=XXX -bench=BenchmarkWatch
```

This will do the following:

1. `-run=XXX` is a regular expression filter on the name of test cases to run
2. `-bench=BenchmarkWatch` will run test methods with BenchmarkWatch in the name
  * See `grep -nr BenchmarkWatch .` for examples
3. `-benchmem` enables memory allocation stats

See `go help test` and `go help testflag` for additional info.

## Integration tests

* Integration tests should only access other resources on the local machine
  - Most commonly etcd or a service listening on localhost.
* All significant features require integration tests.
  - This includes kubectl commands
* The preferred method of testing multiple scenarios or inputs
is [table driven testing](https://github.com/golang/go/wiki/TableDrivenTests)
  - Example: [TestNamespaceAuthorization](../../test/integration/auth/auth_test.go)
* Each test should create its own master, httpserver and config.
  - Example: [TestPodUpdateActiveDeadlineSeconds](../../test/integration/pods/pods_test.go)
* See [coding conventions](coding-conventions.md).

### Install etcd dependency

Kubernetes integration tests require your `PATH` to include an
[etcd](https://github.com/coreos/etcd/releases) installation. Kubernetes
includes a script to help install etcd on your machine.

```sh
# Install etcd and add to PATH

# Option a) install inside kubernetes root
hack/install-etcd.sh  # Installs in ./third_party/etcd
echo export PATH="\$PATH:$(pwd)/third_party/etcd" >> ~/.profile  # Add to PATH

# Option b) install manually
grep -E "image.*etcd" cluster/saltbase/etcd/etcd.manifest  # Find version
# Install that version using yum/apt-get/etc
echo export PATH="\$PATH:<LOCATION>" >> ~/.profile  # Add to PATH
```

### Etcd test data

Many tests start an etcd server internally, storing test data in the operating system's temporary directory.

If you see test failures because the temporary directory does not have sufficient space,
or is on a volume with unpredictable write latency, you can override the test data directory
for those internal etcd instances with the `TEST_ETCD_DIR` environment variable.

### Run integration tests

The integration tests are run using `make test-integration`.
The Kubernetes integration tests are writting using the normal golang testing
package but expect to have a running etcd instance to connect to.  The `test-
integration.sh` script wraps `make test` and sets up an etcd instance
for the integration tests to use.

```sh
make test-integration  # Run all integration tests.
```

This script runs the golang tests in package
[`test/integration`](../../test/integration/).

### Run a specific integration test

You can use also use the `KUBE_TEST_ARGS` environment variable with the `hack
/test-integration.sh` script to run a specific integration test case:

```sh
# Run integration test TestPodUpdateActiveDeadlineSeconds with the verbose flag set.
make test-integration KUBE_GOFLAGS="-v" KUBE_TEST_ARGS="-run ^TestPodUpdateActiveDeadlineSeconds$"
```

If you set `KUBE_TEST_ARGS`, the test case will be run with only the `v1` API
version and the watch cache test is skipped.

## End-to-End tests

Please refer to [End-to-End Testing in Kubernetes](e2e-tests.md).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/testing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
