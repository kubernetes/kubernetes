<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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

Prior to running the tests, it is recommended that you first create a simple auth file in your home directory, e.g. `$HOME/.kubernetes_auth` , with the following:

```
{
  "User": "root",
  "Password": ""
}
```

Next, you will need a cluster that you can test against.  As mentioned earlier, you will want to execute `sudo ./hack/local-up-cluster.sh`.  To get a sense of what tests exist, you may want to run:

`e2e.test --host="127.0.0.1:8080" --provider="local" --ginkgo.v=true -ginkgo.dryRun=true --kubeconfig="$HOME/.kubernetes_auth" --repo-root="$KUBERNETES_SRC_PATH"`

If you wish to execute a specific set of tests you can use the `-ginkgo.focus=` regex, e.g.:

`e2e.test ... --ginkgo.focus="DNS|(?i)nodeport(?-i)|kubectl guestbook"`

Conversely, if you wish to exclude a set of tests, you can run:

`e2e.test ... --ginkgo.skip="Density|Scale"`

As mentioned earlier there are a host of other options that are available, but are left to the developer

**NOTE:** If you are running tests on a local cluster repeatedly, you may need to periodically perform some manual cleanup.
- `rm -rf /var/run/kubernetes`, clear kube generated credentials, sometimes stale permissions can cause problems.
- `sudo iptables -F`, clear ip tables rules left by the kube-proxy.

## Adding a New Test

As mentioned above, prior to adding a new test, it is a good idea to perform a `-ginkgo.dryRun=true` on the system, in order to see if a behavior is already being tested, or to determine if it may be possible to augment an existing set of tests for a specific use case.

If a behavior does not currently have coverage and a developer wishes to add a new e2e test, navigate to the ./test/e2e directory and create a new test using the existing suite as a guide.

**TODO:** Create a self-documented example which has been disabled, but can be copied to create new tests and outlines the capabilities and libraries used.

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





<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/e2e-tests.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
