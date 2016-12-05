# End-to-End Testing in Kubernetes

Updated: 5/3/2016

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [End-to-End Testing in Kubernetes](#end-to-end-testing-in-kubernetes)
  - [Overview](#overview)
  - [Building and Running the Tests](#building-and-running-the-tests)
    - [Cleaning up](#cleaning-up)
  - [Advanced testing](#advanced-testing)
    - [Bringing up a cluster for testing](#bringing-up-a-cluster-for-testing)
    - [Federation e2e tests](#federation-e2e-tests)
      - [Configuring federation e2e tests](#configuring-federation-e2e-tests)
      - [Image Push Repository](#image-push-repository)
      - [Build](#build)
      - [Deploy federation control plane](#deploy-federation-control-plane)
      - [Run the Tests](#run-the-tests)
      - [Teardown](#teardown)
      - [Shortcuts for test developers](#shortcuts-for-test-developers)
    - [Debugging clusters](#debugging-clusters)
    - [Local clusters](#local-clusters)
      - [Testing against local clusters](#testing-against-local-clusters)
    - [Version-skewed and upgrade testing](#version-skewed-and-upgrade-testing)
  - [Kinds of tests](#kinds-of-tests)
    - [Viper configuration and hierarchichal test parameters.](#viper-configuration-and-hierarchichal-test-parameters)
    - [Conformance tests](#conformance-tests)
    - [Defining Conformance Subset](#defining-conformance-subset)
  - [Continuous Integration](#continuous-integration)
    - [What is CI?](#what-is-ci)
    - [What runs in CI?](#what-runs-in-ci)
      - [Non-default tests](#non-default-tests)
    - [The PR-builder](#the-pr-builder)
    - [Adding a test to CI](#adding-a-test-to-ci)
    - [Moving a test out of CI](#moving-a-test-out-of-ci)
  - [Performance Evaluation](#performance-evaluation)
  - [One More Thing](#one-more-thing)

<!-- END MUNGE: GENERATED_TOC -->

## Overview

End-to-end (e2e) tests for Kubernetes provide a mechanism to test end-to-end
behavior of the system, and is the last signal to ensure end user operations
match developer specifications. Although unit and integration tests provide a
good signal, in a distributed system like Kubernetes it is not uncommon that a
minor change may pass all unit and integration tests, but cause unforeseen
changes at the system level.

The primary objectives of the e2e tests are to ensure a consistent and reliable
behavior of the kubernetes code base, and to catch hard-to-test bugs before
users do, when unit and integration tests are insufficient.

The e2e tests in kubernetes are built atop of
[Ginkgo](http://onsi.github.io/ginkgo/) and
[Gomega](http://onsi.github.io/gomega/). There are a host of features that this
Behavior-Driven Development (BDD) testing framework provides, and it is
recommended that the developer read the documentation prior to diving into the
 tests.

The purpose of *this* document is to serve as a primer for developers who are
looking to execute or add tests using a local development environment.

Before writing new tests or making substantive changes to existing tests, you
should also read [Writing Good e2e Tests](writing-good-e2e-tests.md)

## Building and Running the Tests

There are a variety of ways to run e2e tests, but we aim to decrease the number
of ways to run e2e tests to a canonical way: `hack/e2e.go`.

You can run an end-to-end test which will bring up a master and nodes, perform
some tests, and then tear everything down. Make sure you have followed the
getting started steps for your chosen cloud platform (which might involve
changing the `KUBERNETES_PROVIDER` environment variable to something other than
"gce").

To build Kubernetes, up a cluster, run tests, and tear everything down, use:

```sh
go run hack/e2e.go -v --build --up --test --down
```

If you'd like to just perform one of these steps, here are some examples:

```sh
# Build binaries for testing
go run hack/e2e.go -v --build

# Create a fresh cluster.  Deletes a cluster first, if it exists
go run hack/e2e.go -v --up

# Run all tests
go run hack/e2e.go -v --test

# Run tests matching the regex "\[Feature:Performance\]"
go run hack/e2e.go -v --test --test_args="--ginkgo.focus=\[Feature:Performance\]"

# Conversely, exclude tests that match the regex "Pods.*env"
go run hack/e2e.go -v --test --test_args="--ginkgo.skip=Pods.*env"

# Run tests in parallel, skip any that must be run serially
GINKGO_PARALLEL=y go run hack/e2e.go --v --test --test_args="--ginkgo.skip=\[Serial\]"

# Run tests in parallel, skip any that must be run serially and keep the test namespace if test failed
GINKGO_PARALLEL=y go run hack/e2e.go --v --test --test_args="--ginkgo.skip=\[Serial\] --delete-namespace-on-failure=false"

# Flags can be combined, and their actions will take place in this order:
# --build, --up, --test, --down
#
# You can also specify an alternative provider, such as 'aws'
#
# e.g.:
KUBERNETES_PROVIDER=aws go run hack/e2e.go -v --build --up --test --down

# -ctl can be used to quickly call kubectl against your e2e cluster. Useful for
# cleaning up after a failed test or viewing logs. Use -v to avoid suppressing
# kubectl output.
go run hack/e2e.go -v -ctl='get events'
go run hack/e2e.go -v -ctl='delete pod foobar'
```

The tests are built into a single binary which can be run used to deploy a
Kubernetes system or run tests against an already-deployed Kubernetes system.
See `go run hack/e2e.go --help` (or the flag definitions in `hack/e2e.go`) for
more options, such as reusing an existing cluster.

### Cleaning up

During a run, pressing `control-C` should result in an orderly shutdown, but if
something goes wrong and you still have some VMs running you can force a cleanup
with this command:

```sh
go run hack/e2e.go -v --down
```

## Advanced testing

### Bringing up a cluster for testing

If you want, you may bring up a cluster in some other manner and run tests
against it. To do so, or to do other non-standard test things, you can pass
arguments into Ginkgo using `--test_args` (e.g. see above). For the purposes of
brevity, we will look at a subset of the options, which are listed below:

```
--ginkgo.dryRun=false: If set, ginkgo will walk the test hierarchy without
actually running anything. Best paired with -v.

--ginkgo.failFast=false: If set, ginkgo will stop running a test suite after a
failure occurs.

--ginkgo.failOnPending=false: If set, ginkgo will mark the test suite as failed
if any specs are pending.

--ginkgo.focus="": If set, ginkgo will only run specs that match this regular
expression.

--ginkgo.skip="": If set, ginkgo will only run specs that do not match this
regular expression.

--ginkgo.trace=false: If set, default reporter prints out the full stack trace
when a failure occurs

--ginkgo.v=false: If set, default reporter print out all specs as they begin.

--host="": The host, or api-server, to connect to

--kubeconfig="": Path to kubeconfig containing embedded authinfo.

--prom-push-gateway="": The URL to prometheus gateway, so that metrics can be
pushed during e2es and scraped by prometheus. Typically something like
127.0.0.1:9091.

--provider="": The name of the Kubernetes provider (gce, gke, local, vagrant,
etc.)

--repo-root="../../": Root directory of kubernetes repository, for finding test
files.
```

Prior to running the tests, you may want to first create a simple auth file in
your home directory, e.g. `$HOME/.kube/config`, with the following:

```
{
  "User": "root",
  "Password": ""
}
```

As mentioned earlier there are a host of other options that are available, but
they are left to the developer.

**NOTE:** If you are running tests on a local cluster repeatedly, you may need
to periodically perform some manual cleanup:

  - `rm -rf /var/run/kubernetes`, clear kube generated credentials, sometimes
stale permissions can cause problems.

  - `sudo iptables -F`, clear ip tables rules left by the kube-proxy.

### Federation e2e tests

By default, `e2e.go` provisions a single Kubernetes cluster, and any `Feature:Federation` ginkgo tests will be skipped.

Federation e2e testing involve bringing up multiple "underlying" Kubernetes clusters,
and deploying the federation control plane as a Kubernetes application on the underlying clusters.

The federation e2e tests are still managed via `e2e.go`, but require some extra configuration items.

#### Configuring federation e2e tests

The following environment variables will enable federation e2e building, provisioning and testing.

```sh
$ export FEDERATION=true
$ export E2E_ZONES="us-central1-a us-central1-b us-central1-f"
```

A Kubernetes cluster will be provisioned in each zone listed in `E2E_ZONES`. A zone can only appear once in the `E2E_ZONES` list.

#### Image Push Repository

Next, specify the docker repository where your ci images will be pushed.

* **If `KUBERNETES_PROVIDER=gce` or `KUBERNETES_PROVIDER=gke`**:

  If you use the same GCP project where you to run the e2e tests as the container image repository,
  FEDERATION_PUSH_REPO_BASE environment variable will be defaulted to "gcr.io/${DEFAULT_GCP_PROJECT_NAME}".
  You can skip ahead to the **Build** section.

	You can simply set your push repo base based on your project name, and the necessary repositories will be
  auto-created when you first push your container images.

	```sh
	$ export FEDERATION_PUSH_REPO_BASE="gcr.io/${GCE_PROJECT_NAME}"
	```

	Skip ahead to the **Build** section.

* **For all other providers**:

	You'll be responsible for creating and managing access to the repositories manually.

	```sh
	$ export FEDERATION_PUSH_REPO_BASE="quay.io/colin_hom"
	```

	Given this example, the `federation-apiserver` container image will be pushed to the repository
	`quay.io/colin_hom/federation-apiserver`.

	The docker client on the machine running `e2e.go` must have push access for the following pre-existing repositories:

	* `${FEDERATION_PUSH_REPO_BASE}/federation-apiserver`
	* `${FEDERATION_PUSH_REPO_BASE}/federation-controller-manager`

	These repositories must allow public read access, as the e2e node docker daemons will not have any credentials. If you're using
	GCE/GKE as your provider, the repositories will have read-access by default.

#### Build

* Compile the binaries and build container images:

  ```sh
  $ KUBE_RELEASE_RUN_TESTS=n KUBE_FASTBUILD=true go run hack/e2e.go -v -build
  ```

* Push the federation container images

  ```sh
  $ build-tools/push-federation-images.sh
  ```

#### Deploy federation control plane

The following command will create the underlying Kubernetes clusters in each of `E2E_ZONES`, and then provision the
federation control plane in the cluster occupying the last zone in the `E2E_ZONES` list.

```sh
$ go run hack/e2e.go -v --up
```

#### Run the Tests

This will run only the `Feature:Federation` e2e tests. You can omit the `ginkgo.focus` argument to run the entire e2e suite.

```sh
$ go run hack/e2e.go -v --test --test_args="--ginkgo.focus=\[Feature:Federation\]"
```

#### Teardown

```sh
$ go run hack/e2e.go -v --down
```

#### Shortcuts for test developers

* To speed up `e2e.go -up`, provision a single-node kubernetes cluster in a single e2e zone:

  `NUM_NODES=1 E2E_ZONES="us-central1-f"`

  Keep in mind that some tests may require multiple underlying clusters and/or minimum compute resource availability.

* You can quickly recompile the e2e testing framework via `go install ./test/e2e`. This will not do anything besides
  allow you to verify that the go code compiles.

* If you want to run your e2e testing framework without re-provisioning the e2e setup, you can do so via
  `make WHAT=test/e2e/e2e.test` and then re-running the ginkgo tests.

* If you're hacking around with the federation control plane deployment itself,
  you can quickly re-deploy the federation control plane Kubernetes manifests without tearing any resources down.
  To re-deploy the federation control plane after running `-up` for the first time:

  ```sh
  $ federation/cluster/federation-up.sh
  ```

### Debugging clusters

If a cluster fails to initialize, or you'd like to better understand cluster
state to debug a failed e2e test, you can use the `cluster/log-dump.sh` script
to gather logs.

This script requires that the cluster provider supports ssh. Assuming it does,
running:

```
cluster/log-dump.sh <directory>
````

will ssh to the master and all nodes and download a variety of useful logs to
the provided directory (which should already exist).

The Google-run Jenkins builds automatically collected these logs for every
build, saving them in the `artifacts` directory uploaded to GCS.

### Local clusters

It can be much faster to iterate on a local cluster instead of a cloud-based
one. To start a local cluster, you can run:

```sh
# The PATH construction is needed because PATH is one of the special-cased
# environment variables not passed by sudo -E
sudo PATH=$PATH hack/local-up-cluster.sh
```

This will start a single-node Kubernetes cluster than runs pods using the local
docker daemon. Press Control-C to stop the cluster.

You can generate a valid kubeconfig file by following instructions printed at the
end of aforementioned script.

#### Testing against local clusters

In order to run an E2E test against a locally running cluster, point the tests
at a custom host directly:

```sh
export KUBECONFIG=/path/to/kubeconfig
export KUBE_MASTER_IP="http://127.0.0.1:<PORT>"
export KUBE_MASTER=local
go run hack/e2e.go -v --test
```

To control the tests that are run:

```sh
go run hack/e2e.go -v --test --test_args="--ginkgo.focus=\"Secrets\""
```

### Version-skewed and upgrade testing

We run version-skewed tests to check that newer versions of Kubernetes work
similarly enough to older versions.  The general strategy is to cover the following cases:

1. One version of `kubectl` with another version of the cluster and tests (e.g.
   that v1.2 and v1.4 `kubectl` doesn't break v1.3 tests running against a v1.3
   cluster).
1. A newer version of the Kubernetes master with older nodes and tests (e.g.
   that upgrading a master to v1.3 with nodes at v1.2 still passes v1.2 tests).
1. A newer version of the whole cluster with older tests (e.g. that a cluster
   upgraded---master and nodes---to v1.3 still passes v1.2 tests).
1. That an upgraded cluster functions the same as a brand-new cluster of the
   same version (e.g. a cluster upgraded to v1.3 passes the same v1.3 tests as
   a newly-created v1.3 cluster).

[hack/e2e-runner.sh](http://releases.k8s.io/HEAD/hack/jenkins/e2e-runner.sh) is
the authoritative source on how to run version-skewed tests, but below is a
quick-and-dirty tutorial.

```sh
# Assume you have two copies of the Kubernetes repository checked out, at
# ./kubernetes and ./kubernetes_old

# If using GKE:
export KUBERNETES_PROVIDER=gke
export CLUSTER_API_VERSION=${OLD_VERSION}

# Deploy a cluster at the old version; see above for more details
cd ./kubernetes_old
go run ./hack/e2e.go -v --up

# Upgrade the cluster to the new version
#
# If using GKE, add --upgrade-target=${NEW_VERSION}
#
# You can target Feature:MasterUpgrade or Feature:ClusterUpgrade
cd ../kubernetes
go run ./hack/e2e.go -v --test --check_version_skew=false --test_args="--ginkgo.focus=\[Feature:MasterUpgrade\]"

# Run old tests with new kubectl
cd ../kubernetes_old
go run ./hack/e2e.go -v --test --test_args="--kubectl-path=$(pwd)/../kubernetes/cluster/kubectl.sh"
```

If you are just testing version-skew, you may want to just deploy at one
version and then test at another version, instead of going through the whole
upgrade process:

```sh
# With the same setup as above

# Deploy a cluster at the new version
cd ./kubernetes
go run ./hack/e2e.go -v --up

# Run new tests with old kubectl
go run ./hack/e2e.go -v --test --test_args="--kubectl-path=$(pwd)/../kubernetes_old/cluster/kubectl.sh"

# Run old tests with new kubectl
cd ../kubernetes_old
go run ./hack/e2e.go -v --test --test_args="--kubectl-path=$(pwd)/../kubernetes/cluster/kubectl.sh"
```

## Kinds of tests

We are working on implementing clearer partitioning of our e2e tests to make
running a known set of tests easier (#10548). Tests can be labeled with any of
the following labels, in order of increasing precedence (that is, each label
listed below supersedes the previous ones):

  - If a test has no labels, it is expected to run fast (under five minutes), be
able to be run in parallel, and be consistent.

  - `[Slow]`: If a test takes more than five minutes to run (by itself or in
parallel with many other tests), it is labeled `[Slow]`. This partition allows
us to run almost all of our tests quickly in parallel, without waiting for the
stragglers to finish.

  - `[Serial]`: If a test cannot be run in parallel with other tests (e.g. it
takes too many resources or restarts nodes), it is labeled `[Serial]`, and
should be run in serial as part of a separate suite.

  - `[Disruptive]`: If a test restarts components that might cause other tests
to fail or break the cluster completely, it is labeled `[Disruptive]`. Any
`[Disruptive]` test is also assumed to qualify for the `[Serial]` label, but
need not be labeled as both. These tests are not run against soak clusters to
avoid restarting components.

  - `[Flaky]`: If a test is found to be flaky and we have decided that it's too
hard to fix in the short term (e.g. it's going to take a full engineer-week), it
receives the `[Flaky]` label until it is fixed. The `[Flaky]` label should be
used very sparingly, and should be accompanied with a reference to the issue for
de-flaking the test, because while a test remains labeled `[Flaky]`, it is not
monitored closely in CI. `[Flaky]` tests are by default not run, unless a
`focus` or `skip` argument is explicitly given.

  - `[Feature:.+]`: If a test has non-default requirements to run or targets
some non-core functionality, and thus should not be run as part of the standard
suite, it receives a `[Feature:.+]` label, e.g. `[Feature:Performance]` or
`[Feature:Ingress]`. `[Feature:.+]` tests are not run in our core suites,
instead running in custom suites. If a feature is experimental or alpha and is
not enabled by default due to being incomplete or potentially subject to
breaking changes, it does *not* block the merge-queue, and thus should run in
some separate test suites owned by the feature owner(s)
(see [Continuous Integration](#continuous-integration) below).

### Viper configuration and hierarchichal test parameters.

The future of e2e test configuration idioms will be increasingly defined using viper, and decreasingly via flags.

Flags in general fall apart once tests become sufficiently complicated.  So, even if we could use another flag library, it wouldn't be ideal.

To use viper, rather than flags, to configure your tests:

- Just add "e2e.json" to the current directory you are in, and define parameters in it... i.e. `"kubeconfig":"/tmp/x"`.

Note that advanced testing parameters, and hierarchichally defined parameters, are only defined in viper, to see what they are, you can dive into [TestContextType](../../test/e2e/framework/test_context.go).

In time, it is our intent to add or autogenerate a sample viper configuration that includes all e2e parameters, to ship with kubernetes.

### Conformance tests

Finally, `[Conformance]` tests represent a subset of the e2e-tests we expect to
pass on **any** Kubernetes cluster. The `[Conformance]` label does not supersede
any other labels.

As each new release of Kubernetes providers new functionality, the subset of
tests necessary to demonstrate conformance grows with each release. Conformance
is thus considered versioned, with the same backwards compatibility guarantees
as laid out in [our versioning policy](../design/versioning.md#supported-releases).
Conformance tests for a given version should be run off of the release branch
that corresponds to that version. Thus `v1.2` conformance tests would be run
from the head of the `release-1.2` branch. eg:

 - A v1.3 development cluster should pass v1.1, v1.2 conformance tests

 - A v1.2 cluster should pass v1.1, v1.2 conformance tests

 - A v1.1 cluster should pass v1.0, v1.1 conformance tests, and fail v1.2
conformance tests

Conformance tests are designed to be run with no cloud provider configured.
Conformance tests can be run against clusters that have not been created with
`hack/e2e.go`, just provide a kubeconfig with the appropriate endpoint and
credentials.

```sh
# setup for conformance tests
export KUBECONFIG=/path/to/kubeconfig
export KUBERNETES_CONFORMANCE_TEST=y
export KUBERNETES_PROVIDER=skeleton

# run all conformance tests
go run hack/e2e.go -v --test --test_args="--ginkgo.focus=\[Conformance\]"

# run all parallel-safe conformance tests in parallel
GINKGO_PARALLEL=y go run hack/e2e.go -v --test --test_args="--ginkgo.focus=\[Conformance\] --ginkgo.skip=\[Serial\]"

# ... and finish up with remaining tests in serial
go run hack/e2e.go -v --test --test_args="--ginkgo.focus=\[Serial\].*\[Conformance\]"
```

### Defining Conformance Subset

It is impossible to define the entire space of Conformance tests without knowing
the future, so instead, we define the compliment of conformance tests, below
(`Please update this with companion PRs as necessary`):

  - A conformance test cannot test cloud provider specific features (i.e. GCE
monitoring, S3 Bucketing, ...)

  - A conformance test cannot rely on any particular non-standard file system
permissions granted to containers or users (i.e. sharing writable host /tmp with
a container)

  - A conformance test cannot rely on any binaries that are not required for the
linux kernel or for a kubelet to run (i.e. git)

  - A conformance test cannot test a feature which obviously cannot be supported
on a broad range of platforms (i.e. testing of multiple disk mounts, GPUs, high
density)

## Continuous Integration

A quick overview of how we run e2e CI on Kubernetes.

### What is CI?

We run a battery of `e2e` tests against `HEAD` of the master branch on a
continuous basis, and block merges via the [submit
queue](http://submit-queue.k8s.io/) on a subset of those tests if they fail (the
subset is defined in the [munger config]
(https://github.com/kubernetes/contrib/blob/master/mungegithub/mungers/submit-queue.go)
via the `jenkins-jobs` flag; note we also block on	`kubernetes-build` and
`kubernetes-test-go` jobs for build and unit and integration tests).

CI results can be found at [ci-test.k8s.io](http://ci-test.k8s.io), e.g.
[ci-test.k8s.io/kubernetes-e2e-gce/10594](http://ci-test.k8s.io/kubernetes-e2e-gce/10594).

### What runs in CI?

We run all default tests (those that aren't marked `[Flaky]` or `[Feature:.+]`)
against GCE and GKE. To minimize the time from regression-to-green-run, we
partition tests across different jobs:

  - `kubernetes-e2e-<provider>` runs all non-`[Slow]`, non-`[Serial]`,
non-`[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in parallel.

  - `kubernetes-e2e-<provider>-slow` runs all `[Slow]`, non-`[Serial]`,
non-`[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in parallel.

  - `kubernetes-e2e-<provider>-serial` runs all `[Serial]` and `[Disruptive]`,
non-`[Flaky]`, non-`[Feature:.+]` tests in serial.

We also run non-default tests if the tests exercise general-availability ("GA")
features that require a special environment to run in, e.g.
`kubernetes-e2e-gce-scalability` and `kubernetes-kubemark-gce`, which test for
Kubernetes performance.

#### Non-default tests

Many `[Feature:.+]` tests we don't run in CI. These tests are for features that
are experimental (often in the `experimental` API), and aren't enabled by
default.

### The PR-builder

We also run a battery of tests against every PR before we merge it. These tests
are equivalent to `kubernetes-gce`: it runs all non-`[Slow]`, non-`[Serial]`,
non-`[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in parallel. These
tests are considered "smoke tests" to give a decent signal that the PR doesn't
break most functionality. Results for your PR can be found at
[pr-test.k8s.io](http://pr-test.k8s.io), e.g.
[pr-test.k8s.io/20354](http://pr-test.k8s.io/20354) for #20354.

### Adding a test to CI

As mentioned above, prior to adding a new test, it is a good idea to perform a
`-ginkgo.dryRun=true` on the system, in order to see if a behavior is already
being tested, or to determine if it may be possible to augment an existing set
of tests for a specific use case.

If a behavior does not currently have coverage and a developer wishes to add a
new e2e test, navigate to the ./test/e2e directory and create a new test using
the existing suite as a guide.

TODO(#20357): Create a self-documented example which has been disabled, but can
be copied to create new tests and outlines the capabilities and libraries used.

When writing a test, consult #kinds_of_tests above to determine how your test
should be marked, (e.g. `[Slow]`, `[Serial]`; remember, by default we assume a
test can run in parallel with other tests!).

When first adding a test it should *not* go straight into CI, because failures
block ordinary development. A test should only be added to CI after is has been
running in some non-CI suite long enough to establish a track record showing
that the test does not fail when run against *working* software. Note also that
tests running in CI are generally running on a well-loaded cluster, so must
contend for resources; see above about [kinds of tests](#kinds_of_tests).

Generally, a feature starts as `experimental`, and will be run in some suite
owned by the team developing the feature. If a feature is in beta or GA, it
*should* block the merge-queue. In moving from experimental to beta or GA, tests
that are expected to pass by default should simply remove the `[Feature:.+]`
label, and will be incorporated into our core suites. If tests are not expected
to pass by default, (e.g. they require a special environment such as added
quota,) they should remain with the `[Feature:.+]` label, and the suites that
run them should be incorporated into the
[munger config](https://github.com/kubernetes/contrib/blob/master/mungegithub/mungers/submit-queue.go)
via the `jenkins-jobs` flag.

Occasionally, we'll want to add tests to better exercise features that are
already GA. These tests also shouldn't go straight to CI. They should begin by
being marked as `[Flaky]` to be run outside of CI, and once a track-record for
them is established, they may be promoted out of `[Flaky]`.

### Moving a test out of CI

If we have determined that a test is known-flaky and cannot be fixed in the
short-term, we may move it out of CI indefinitely. This move should be used
sparingly, as it effectively means that we have no coverage of that test. When a
test is demoted, it should be marked `[Flaky]` with a comment accompanying the
label with a reference to an issue opened to fix the test.

## Performance Evaluation

Another benefit of the e2e tests is the ability to create reproducible loads on
the system, which can then be used to determine the responsiveness, or analyze
other characteristics of the system. For example, the density tests load the
system to 30,50,100 pods per/node and measures the different characteristics of
the system, such as throughput, api-latency, etc.

For a good overview of how we analyze performance data, please read the
following [post](http://blog.kubernetes.io/2015/09/kubernetes-performance-measurements-and.html)

For developers who are interested in doing their own performance analysis, we
recommend setting up [prometheus](http://prometheus.io/) for data collection,
and using [promdash](http://prometheus.io/docs/visualization/promdash/) to
visualize the data.  There also exists the option of pushing your own metrics in
from the tests using a
[prom-push-gateway](http://prometheus.io/docs/instrumenting/pushing/).
Containers for all of these components can be found
[here](https://hub.docker.com/u/prom/).

For more accurate measurements, you may wish to set up prometheus external to
kubernetes in an environment where it can access the major system components
(api-server, controller-manager, scheduler). This is especially useful when
attempting to gather metrics in a load-balanced api-server environment, because
all api-servers can be analyzed independently as well as collectively. On
startup, configuration file is passed to prometheus that specifies the endpoints
that prometheus will scrape, as well as the sampling interval.

```
#prometheus.conf
job: {
  name: "kubernetes"
  scrape_interval: "1s"
  target_group: {
    # apiserver(s)
    target: "http://localhost:8080/metrics"
    # scheduler
    target: "http://localhost:10251/metrics"
    # controller-manager
    target: "http://localhost:10252/metrics"
  }
}
```

Once prometheus is scraping the kubernetes endpoints, that data can then be
plotted using promdash, and alerts can be created against the assortment of
metrics that kubernetes provides.

## One More Thing

You should also know the [testing conventions](coding-conventions.md#testing-conventions).

**HAPPY TESTING!**



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/e2e-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
