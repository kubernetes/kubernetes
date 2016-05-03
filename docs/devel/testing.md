<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Testing guide

Updated: 5/3/2016

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Testing guide](#testing-guide)
  - [Unit tests](#unit-tests)
    - [Run all unit tests](#run-all-unit-tests)
    - [Run some unit tests](#run-some-unit-tests)
    - [Stress running unit tests](#stress-running-unit-tests)
    - [Unit test coverage](#unit-test-coverage)
    - [Benchmark unit tests](#benchmark-unit-tests)
  - [Integration tests](#integration-tests)
    - [Install etcd dependency](#install-etcd-dependency)
    - [Run integration tests](#run-integration-tests)
  - [End-to-End tests](#end-to-end-tests)

<!-- END MUNGE: GENERATED_TOC -->

This assumes you already read the [development guide](development.md) to
install go, godeps, and configure your git client.

Before sending pull requests you should at least make sure your changes have
passed both unit and integration tests.

Kubernetes only merges pull requests when unit, integration, and e2e tests are
passing, so it is often a good idea to make sure the e2e tests work as well.

## Unit tests

* Unit tests should be fully hermetic
  - Only access resources in the test binary.
* All packages and any significant files require unit tests.
* The preferred method of testing multiple scenarios or inputs
is [table driven testing](https://github.com/golang/go/wiki/TableDrivenTests)
  - Example: [TestNamespaceAuthorization](../../test/integration/auth_test.go)
* Unit tests must pass on OS X and Windows platforms.
  - Tests using linux-specific features must be skipped or compiled out.
  - Skipped is better, compiled out is required when it won't compile.
* Concurrent unit test runs must pass.
* See [coding conventions](coding-conventions.md).

### Run all unit tests

```sh
cd kubernetes
hack/test-go.sh  # Run all unit tests.
```

### Run some unit tests

```sh
cd kubernetes

# Run all tests under pkg (requires client to be in $GOPATH/src/k8s.io)
godep go test ./pkg/...

# Run all tests in the pkg/api (but not subpackages)
godep go test ./pkg/api
```

### Stress running unit tests

Running the same tests repeatedly is one way to root out flakes.
You can do this efficiently.


```sh
cd kubernetes

# Have 2 workers run all tests 5 times each (10 total iterations).
hack/test-go.sh -p 2 -i 5
```

For more advanced ideas please see [flaky-tests.md](flaky-tests.md).

### Unit test coverage

Currently, collecting coverage is only supported for the Go unit tests.

To run all unit tests and generate an HTML coverage report, run the following:

```sh
cd kubernetes
KUBE_COVER=y hack/test-go.sh
```

At the end of the run, an the HTML report will be generated with the path printed to stdout.

To run tests and collect coverage in only one package, pass its relative path under the `kubernetes` directory as an argument, for example:

```sh
cd kubernetes
KUBE_COVER=y hack/test-go.sh pkg/kubectl
```

Multiple arguments can be passed, in which case the coverage results will be combined for all tests run.

Coverage results for the project can also be viewed on [Coveralls](https://coveralls.io/r/kubernetes/kubernetes), and are continuously updated as commits are merged. Additionally, all pull requests which spawn a Travis build will report unit test coverage results to Coveralls. Coverage reports from before the Kubernetes Github organization was created can be found [here](https://coveralls.io/r/GoogleCloudPlatform/kubernetes).

### Benchmark unit tests

To run benchmark tests, you'll typically use something like:

```sh
cd kubernetes
godep go test ./pkg/apiserver -benchmem -run=XXX -bench=BenchmarkWatch
```

This will do the following:

1. `-run=XXX` will turn off regular unit tests
  * Technically it will run test methods with XXX in the name.
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
  - Example: [TestNamespaceAuthorization](../../test/integration/auth_test.go)
* Integration tests must run in parallel
  - Each test should create its own master, httpserver and config.
  - Example: [TestPodUpdateActiveDeadlineSeconds](../../test/integration/pods_test.go)
* See [coding conventions](coding-conventions.md).

### Install etcd dependency

Kubernetes integration tests require your PATH to include an [etcd](https://github.com/coreos/etcd/releases) installation.
Kubernetes includes a script to help install etcd on your machine.

```sh
# Install etcd and add to PATH

# Option a) install inside kubernetes root
cd kubernetes
hack/install-etcd.sh  # Installs in ./third_party/etcd
echo export PATH="$PATH:$(pwd)/third_party/etcd" >> ~/.profile  # Add to PATH

# Option b) install manually
cd kubernetes
grep -E "image.*etcd" cluster/saltbase/etcd/etcd.manifest  # Find version
# Install that version using yum/apt-get/etc
echo export PATH="$PATH:<LOCATION>" >> ~/.profile  # Add to PATH
```

### Run integration tests

```sh
cd kubernetes
hack/test-integration.sh  # Run all integration tests.
```


## End-to-End tests

Please refer to [End-to-End Testing in Kubernetes](e2e-tests.md).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/testing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
