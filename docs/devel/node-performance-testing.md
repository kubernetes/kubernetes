<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/devel/node-performance-testing.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Measuring Node Performance

This document outlines the issues and pitfalls of measuring Node performance, as
well as the tools available.

## Cluster Set-up

There are lots of factors which can affect node performance numbers, so care
must be taken in setting up the cluster to make the intended measurements. In
addition to taking the following steps into consideration, it is important to
document precisely which setup was used. For example, performance can vary
wildly from commit-to-commit, so it is very important to **document which commit
or version** of Kubernetes was used, which Docker version was used, etc.

### Addon pods

Be aware of which addon pods are running on which nodes. By default Kubernetes
runs 8 addon pods, plus another 2 per node (`fluentd-elasticsearch` and
`kube-proxy`) in the `kube-system` namespace. The addon pods can be disabled for
more consistent results, but doing so can also have performance implications.

For example, Heapster polls each node regularly to collect stats data. Disabling
Heapster will hide the performance cost of serving those stats in the Kubelet.

#### Disabling Add-ons

Disabling addons is simple. Just ssh into the Kubernetes master and move the
addon from `/etc/kubernetes/addons/` to a backup location. More details
[here](../../cluster/addons/).

### Which / how many pods?

Performance will vary a lot between a node with 0 pods and a node with 100 pods.
In many cases you'll want to make measurements with several different amounts of
pods. On a single node cluster scaling a replication controller makes this easy,
just make sure the system reaches a steady-state before starting the
measurement. E.g. `kubectl scale replicationcontroller pause --replicas=100`

In most cases pause pods will yield the most consistent measurements since the
system will not be affected by pod load. However, in some special cases
Kubernetes has been tuned to optimize pods that are not doing anything, such as
the cAdvisor housekeeping (stats gathering). In these cases, performing a very
light task (such as a simple network ping) can make a difference.

Finally, you should also consider which features yours pods should be using. For
example, if you want to measure performance with probing, you should obviously
use pods with liveness or readiness probes configured. Likewise for volumes,
number of containers, etc.

### Other Tips

**Number of nodes** - On the one hand, it can be easier to manage logs, pods,
environment etc. with a single node to worry about. On the other hand, having
multiple nodes will let you gather more data in parallel for more robust
sampling.

## E2E Performance Test

There is an end-to-end test for collecting overall resource usage of node
components: [kubelet_perf.go](../../test/e2e/kubelet_perf.go). To
run the test, simply make sure you have an e2e cluster running (`go run
hack/e2e.go -up`) and [set up](#cluster-set-up) correctly.

Run the test with `go run hack/e2e.go -v -test
--test_args="--ginkgo.focus=resource\susage\stracking"`. You may also wish to
customise the number of pods or other parameters of the test (remember to rerun
`make WHAT=test/e2e/e2e.test` after you do).

## Profiling

Kubelet installs the [go pprof handlers]
(https://golang.org/pkg/net/http/pprof/), which can be queried for CPU profiles:

```console
$ kubectl proxy &
Starting to serve on 127.0.0.1:8001
$ curl -G "http://localhost:8001/api/v1/proxy/nodes/${NODE}:10250/debug/pprof/profile?seconds=${DURATION_SECONDS}" > $OUTPUT
$ KUBELET_BIN=_output/dockerized/bin/linux/amd64/kubelet
$ go tool pprof -web $KUBELET_BIN $OUTPUT
```

`pprof` can also provide heap usage, from the `/debug/pprof/heap` endpoint
(e.g. `http://localhost:8001/api/v1/proxy/nodes/${NODE}:10250/debug/pprof/heap`).

More information on go profiling can be found
[here](http://blog.golang.org/profiling-go-programs).

## Benchmarks

Before jumping through all the hoops to measure a live Kubernetes node in a real
cluster, it is worth considering whether the data you need can be gathered
through a Benchmark test. Go provides a really simple benchmarking mechanism,
just add a unit test of the form:

```go
// In foo_test.go
func BenchmarkFoo(b *testing.B) {
  b.StopTimer()
  setupFoo() // Perform any global setup
  b.StartTimer()
  for i := 0; i < b.N; i++ {
    foo() // Functionality to measure
  }
}
```

Then:

```console
$ go test -bench=. -benchtime=${SECONDS}s foo_test.go
```

More details on benchmarking [here](https://golang.org/pkg/testing/).

## TODO

- (taotao) Measuring docker performance
- Expand cluster set-up section
- (vishh) Measuring disk usage
- (yujuhong) Measuring memory usage
- Add section on monitoring kubelet metrics (e.g. with prometheus)



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/node-performance-testing.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
