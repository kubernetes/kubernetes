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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/devel/e2e-tests.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# End-2-End Testing in Kubernetes

## Overview

The end-2-end tests for kubernetes provide a mechanism to test behavior of the system, and to ensure end user operations match developer specifications.  In distributed systems it is not uncommon that a minor change may pass all unit tests, but cause unforseen changes at the system level.  Thus, the primary objectives of the end-2-end tests are to ensure a consistent and reliable behavior of the kubernetes code base, and to catch bugs early.

The end-2-end tests in kubernetes are built atop of [ginkgo] (http://onsi.github.io/ginkgo/) and [gomega] (http://onsi.github.io/gomega/).  There are a host of features that this BDD testing framework provides, and it is recommended that the developer read the documentation prior to diving into the tests.

The purpose of *this* document is to serve as a primer for developers who are looking to execute, or add tests, using a local development environment.

## Building and Running the Tests

**NOTE:** The tests have an array of options.  For simplicity, the examples will focus on leveraging the tests on a local cluster using `sudo ./hack/local-up-cluster.sh`

### Building the Tests

The tests are built into a single binary which can be run against any deployed kubernetes system.  To build the tests, navigate to your source directory and execute:

`$ make all`

The output for the end-2-end tests will be a single binary called `e2e.test` under the default output directory, which is typically `_output/local/bin/linux/amd64/`.  Within the repository there are scripts that are provided under the `./hack` directory that are helpful for automation, but may not apply for a local development purposes.  Instead, we recommend familiarizing yourself with the executable options.  To obtain the full list of options, run the following:

`$ ./e2e.test --help`

### Running the Tests

For the purposes of brevity, we will look at a subset of the options, which are listed below:

```
-ginkgo.dryRun=false: If set, ginkgo will walk the test hierarchy without actually running anything.  Best paired with -v.
-ginkgo.failFast=false: If set, ginkgo will stop running a test suite after a failure occurs.
-ginkgo.failOnPending=false: If set, ginkgo will mark the test suite as failed if any specs are pending.
-ginkgo.focus="": If set, ginkgo will only run specs that match this regular expression.
-ginkgo.skip="": If set, ginkgo will only run specs that do not match this regular expression.
-ginkgo.trace=false: If set, default reporter prints out the full stack trace when a failure occurs
-ginkgo.v=false: If set, default reporter print out all specs as they begin.
-host="": The host, or api-server, to connect to
-kubeconfig="": Path to kubeconfig containing embedded authinfo.
-prom-push-gateway="": The URL to prometheus gateway, so that metrics can be pushed during e2es and scraped by prometheus. Typically something like 127.0.0.1:9091.
-provider="": The name of the Kubernetes provider (gce, gke, local, vagrant, etc.)
-repo-root="../../": Root directory of kubernetes repository, for finding test files.
```

Prior to running the tests, it is recommended that you first create a simple auth file in your home directory, e.g. `$HOME/.kube/config` , with the following:

```
{
  "User": "root",
  "Password": ""
}
```

Next, you will need a cluster that you can test against.  As mentioned earlier, you will want to execute `sudo ./hack/local-up-cluster.sh`.  To get a sense of what tests exist, you may want to run:

`e2e.test --host="127.0.0.1:8080" --provider="local" --ginkgo.v=true -ginkgo.dryRun=true --kubeconfig="$HOME/.kube/config" --repo-root="$KUBERNETES_SRC_PATH"`

If you wish to execute a specific set of tests you can use the `-ginkgo.focus=` regex, e.g.:

`e2e.test ... --ginkgo.focus="DNS|(?i)nodeport(?-i)|kubectl guestbook"`

Conversely, if you wish to exclude a set of tests, you can run:

`e2e.test ... --ginkgo.skip="Density|Scale"`

As mentioned earlier there are a host of other options that are available, but are left to the developer

**NOTE:** If you are running tests on a local cluster repeatedly, you may need to periodically perform some manual cleanup.
- `rm -rf /var/run/kubernetes`, clear kube generated credentials, sometimes stale permissions can cause problems.
- `sudo iptables -F`, clear ip tables rules left by the kube-proxy.

## Kinds of tests

We are working on implementing clearer partitioning of our e2e tests to make running a known set of tests easier (#10548).  Tests can be labeled with any of the following labels, in order of increasing precedence (that is, each label listed below supersedes the previous ones):

- If a test has no labels, it is expected to run fast (under five minutes), be able to be run in parallel, and be consistent.
- `[Slow]`: If a test takes more than five minutes to run (by itself or in parallel with many other tests), it is labeled `[Slow]`.  This partition allows us to run almost all of our tests quickly in parallel, without waiting for the stragglers to finish.
- `[Serial]`: If a test cannot be run in parallel with other tests (e.g. it takes too many resources or restarts nodes), it is labeled `[Serial]`, and should be run in serial as part of a separate suite.
- `[Disruptive]`: If a test restarts components that might cause other tests to fail or break the cluster completely, it is labeled `[Disruptive]`.  Any `[Disruptive]` test is also assumed to qualify for the `[Serial]` label, but need not be labeled as both.  These tests are not run against soak clusters to avoid restarting components.
- `[Flaky]`: If a test is found to be flaky and we have decided that it's too hard to fix in the short term (e.g. it's going to take a full engineer-week), it receives the `[Flaky]` label until it is fixed.  The `[Flaky]` label should be used very sparingly, and should be accompanied with a reference to the issue for de-flaking the test, because while a test remains labeled `[Flaky]`, it is not monitored closely in CI. `[Flaky]` tests are by default not run, unless a `focus` or `skip` argument is explicitly given.
- `[Skipped]`: `[Skipped]` is a legacy label that we're phasing out.  If a test is marked `[Skipped]`, there should be an issue open to label it properly.  `[Skipped]` tests are by default not run, unless a `focus` or `skip` argument is explicitly given.
- `[Feature:.+]`: If a test has non-default requirements to run or targets some non-core functionality, and thus should not be run as part of the standard suite, it receives a `[Feature:.+]` label, e.g. `[Feature:Performance]` or `[Feature:Ingress]`.  `[Feature:.+]` tests are not run in our core suites, instead running in custom suites. If a feature is experimental or alpha and is not enabled by default due to being incomplete or potentially subject to breaking changes, it does *not* block the merge-queue, and thus should run in some separate test suites owned by the feature owner(s) (see #continuous_integration below).
- `[LocalNode]`: `[LocalNode]` indicates that a test performs operations on the host system it is running on, which it expects to be the same host that is running the node utilized by pods run by the test.  Example usage is setting up a host path that is used as a volume mount for a pod.

Finally, `[Conformance]` tests are tests we expect to pass on **any** Kubernetes cluster.  The `[Conformance]` label does not supersede any other labels.  `[Conformance]` test policies are a work-in-progress; see #18162.

## Continuous Integration

A quick overview of how we run e2e CI on Kubernetes.

### What is CI?

We run a battery of `e2e` tests against `HEAD` of the master branch on a continuous basis, and block merges via the [submit queue](http://submit-queue.k8s.io/) on a subset of those tests if they fail (the subset is defined in the [munger config](https://github.com/kubernetes/contrib/blob/master/mungegithub/mungers/submit-queue.go) via the `jenkins-jobs` flag; note we also block on	`kubernetes-build` and `kubernetes-test-go` jobs for build and unit and integration tests).

CI results can be found at [ci-test.k8s.io](http://ci-test.k8s.io), e.g. [ci-test.k8s.io/kubernetes-e2e-gce/10594](http://ci-test.k8s.io/kubernetes-e2e-gce/10594).

### What runs in CI?

We run all default tests (those that aren't marked `[Flaky]` or `[Feature:.+]`) against GCE and GKE.  To minimize the time from regression-to-green-run, we partition tests across different jobs:

- `kubernetes-e2e-<provider>` runs all non-`[Slow]`, non-`[Serial]`, non-`[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in parallel.
- `kubernetes-e2e-<provider>-slow` runs all `[Slow]`, non-`[Serial]`, non-`[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in parallel.
- `kubernetes-e2e-<provider>-serial` runs all `[Serial]` and `[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in serial.

We also run non-default tests if the tests exercise general-availability ("GA") features that require a special environment to run in, e.g. `kubernetes-e2e-gce-scalability` and `kubernetes-kubemark-gce`, which test for Kubernetes performance.

#### Non-default tests

Many `[Feature:.+]` tests we don't run in CI.  These tests are for features that are experimental (often in the `experimental` API), and aren't enabled by default.

### The PR-builder

We also run a battery of tests against every PR before we merge it.  These tests are equivalent to `kubernetes-gce`: it runs all non-`[Slow]`, non-`[Serial]`, non-`[Disruptive]`, non-`[Flaky]`, non-`[Feature:.+]` tests in parallel.  These tests are considered "smoke tests" to give a decent signal that the PR doesn't break most functionality.  Results for you PR can be found at [pr-test.k8s.io](http://pr-test.k8s.io), e.g. [pr-test.k8s.io/20354](http://pr-test.k8s.io/20354) for #20354.

### Adding a test to CI

As mentioned above, prior to adding a new test, it is a good idea to perform a `-ginkgo.dryRun=true` on the system, in order to see if a behavior is already being tested, or to determine if it may be possible to augment an existing set of tests for a specific use case.

If a behavior does not currently have coverage and a developer wishes to add a new e2e test, navigate to the ./test/e2e directory and create a new test using the existing suite as a guide.

TODO(#20357): Create a self-documented example which has been disabled, but can be copied to create new tests and outlines the capabilities and libraries used.

When writing a test, consult #kinds_of_tests above to determine how your test should be marked, (e.g. `[Slow]`, `[Serial]`; remember, by default we assume a test can run in parallel with other tests!).

When first adding a test it should *not* go straight into CI, because failures block ordinary development. A test should only be added to CI after is has been running in some non-CI suite long enough to establish a track record showing that the test does not fail when run against *working* software.  Note also that tests running in CI are generally running on a well-loaded cluster, so must contend for resources; see above about [kinds of tests](#kinds_of_tests).

Generally, a feature starts as `experimental`, and will be run in some suite owned by the team developing the feature.  If a feature is in beta or GA, it *should* block the merge-queue.  In moving from experimental to beta or GA, tests that are expected to pass by default should simply remove the `[Feature:.+]` label, and will be incorporated into our core suites.  If tests are not expected to pass by default, (e.g. they require a special environment such as added quota,) they should remain with the `[Feature:.+]` label, and the suites that run them should be incorporated into the [munger config](https://github.com/kubernetes/contrib/blob/master/mungegithub/mungers/submit-queue.go) via the `jenkins-jobs` flag.

Occasionally, we'll want to add tests to better exercise features that are already GA.  These tests also shouldn't go straight to CI.  They should begin by being marked as `[Flaky]` to be run outside of CI, and once a track-record for them is established, they may be promoted out of `[Flaky]`.

### Moving a test out of CI

If we have determined that a test is known-flaky and cannot be fixed in the short-term, we may move it out of CI indefinitely.  This move should be used sparingly, as it effectively means that we have no coverage of that test.  When a test if demoted, it should be marked `[Flaky]` with a comment accompanying the label with a reference to an issue opened to fix the test.

## Performance Evaluation

Another benefit of the end-2-end tests is the ability to create reproducible loads on the system, which can then be used to determine the responsiveness, or analyze other characteristics of the system.  For example, the density tests load the system to 30,50,100 pods per/node and measures the different characteristics of the system, such as throughput, api-latency, etc.

For a good overview of how we analyze performance data, please read the following [post](http://blog.kubernetes.io/2015/09/kubernetes-performance-measurements-and.html)

For developers who are interested in doing their own performance analysis, we recommend setting up [prometheus](http://prometheus.io/) for data collection, and using [promdash](http://prometheus.io/docs/visualization/promdash/) to visualize the data.  There also exists the option of pushing your own metrics in from the tests using a [prom-push-gateway](http://prometheus.io/docs/instrumenting/pushing/).  Containers for all of these components can be found [here](https://hub.docker.com/u/prom/).

For more accurate measurements, you may wish to set up prometheus external to kubernetes in an environment where it can access the major system components (api-server, controller-manager, scheduler).  This is especially useful when attempting to gather metrics in a load-balanced api-server environment, because all api-servers can be analyzed independently as well as collectively. On startup, configuration file is passed to prometheus that specifies the endpoints that prometheus will scrape, as well as the sampling interval.

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
```

Once prometheus is scraping the kubernetes endpoints, that data can then be plotted using promdash, and alerts can be created against the assortment of metrics that kubernetes provides.

**HAPPY TESTING!**



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/e2e-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
