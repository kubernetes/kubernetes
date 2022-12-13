# Overview

The Kubernetes E2E framework simplifies writing Ginkgo tests suites. It's main
usage is for these tests suites in the Kubernetes repository itself:
- test/e2e: runs as client for a Kubernetes cluster. The e2e.test binary is
  used for conformance testing.
- test/e2e_node: runs on the same node as a kublet instance. Used for testing
  kubelet.
- test/e2e_kubeadm: test suite for kubeadm.

Usage of the framework outside of Kubernetes is possible, but not encouraged.
Downstream users have to be prepared to deal with API changes.

# Code Organization

The core framework is the `k8s.io/kubernetes/test/e2e/framework` package. It
contains functionality that all E2E suites are expected to need:
- connecting to the apiserver
- managing per-test namespaces
- logging (`Logf`)
- failure handling (`Fail`, `Failf`)
- writing concise JUnit test results

It also contains a `TestContext` with settings that can be controlled via
command line flags. For historic reasons, this also contains settings for
individual tests or packages that are not part of the core framework.

Optional functionality is placed in sub packages like
`test/e2e/framework/pod`. The core framework does not depend on those. Sub
packages may depend on the core framework.

The advantages of splitting the code like this are:
- leaner go doc packages by grouping related functions together
- not forcing all E2E suites to import all functionality
- avoiding import cycles

# Execution Flow

When a test suite gets invoked, the top-level `Describe` calls register the
callbacks that define individual tests, but does not invoke them yet. After
that init phase, command line flags are parsed and the `Describe` callbacks are
invoked. Those then define the actual tests for the test suite. Command line
flags can be used to influence the test definitions.

Now `Context/BeforeEach/AfterEach/It` define code that will be called later
when executing a specific test. During this setup phase, `f :=
framework.NewDefaultFramework("some tests")` creates a `Framework` instance for
one or more tests. `NewDefaultFramework` initializes that instance anew for
each test with a `BeforeEach` callback. Starting with Kubernetes 1.26, that
instance gets cleaned up after all other code for a test has been invoked, so
the following code is correct:

```
f := framework.NewDefaultFramework("some tests")

ginkgo.AfterEach(func() {
    # Do something with f.ClientSet.
}

ginkgo.It("test something", func(ctx context.Context) {
    # The actual test.
})
```

Optional functionality can be injected into each test by adding a callback to
`NewFrameworkExtensions` in an init function. `NewDefaultFramework` will invoke
those callbacks as if the corresponding code had been added to each test like this:

```
f := framework.NewDefaultFramework("some tests")

optional.SomeCallback(f)
```

`SomeCallback` then can register additional `BeforeEach` or `AfterEach`
callbacks that use the test's `Framework` instance.

When a test runs, callbacks defined for it with `BeforeEach` and `AfterEach`
are called in first-in-first-out order. Since the migration to ginkgo v2 in
Kubernetes 1.25, the `AfterEach` callback is called also when there has been a
test failure. This can be used to run cleanup code for a test
reliably. However,
[`ginkgo.DeferCleanup`](https://onsi.github.io/ginkgo/#spec-cleanup-aftereach-and-defercleanup)
is often a better alternative. Its callbacks are executed in first-in-last-out
order.

`test/e2e/framework/internal/unittests/cleanup/cleanup.go` shows how these
different callbacks can be used and in which order they are going to run.
