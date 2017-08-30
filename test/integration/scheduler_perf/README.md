Scheduler Performance Test
======

Motivation
------
We already have a performance testing system -- Kubemark. However, Kubemark requires setting up and bootstrapping a whole cluster, which takes a lot of time.

We want to have a standard way to reproduce scheduling latency metrics result and benchmark scheduler as simple and fast as possible. We have the following goals:

- Save time on testing
  - The test and benchmark can be run in a single box.
    We only set up components necessary to scheduling without booting up a cluster.
- Profiling runtime metrics to find out bottleneck
  - Write scheduler integration test but focus on performance measurement.
    Take advantage of go profiling tools and collect fine-grained metrics,
    like cpu-profiling, memory-profiling and block-profiling.
- Reproduce test result easily
  - We want to have a known place to do the performance related test for scheduler.
    Developers should just run one script to collect all the information they need.

Currently the test suite has the following:

- density test (by adding a new Go test)
  - schedule 30k pods on 1000 (fake) nodes and 3k pods on 100 (fake) nodes
  - print out scheduling rate every second
  - let you learn the rate changes vs number of scheduled pods
- benchmark
  - make use of `go test -bench` and report nanosecond/op.
  - schedule b.N pods when the cluster has N nodes and P scheduled pods. Since it takes relatively long time to finish one round, b.N is small: 10 - 100.


How To Run
------
```shell
# In Kubernetes root path
make generated_files

cd test/integration/scheduler_perf
./test-performance.sh
```

Configure Scheduler
------

The scheduling policy in this Scheduler Performance Test is also configurable.

Be default, `kube-scheduler` will always use default algorithm provider regardless of `config/scheduler-config.json` is present or not.

You can specify you want to use customized scheduler by set `DEFAULT_PROVIDER` to `false`:

```shell
export DEFAULT_PROVIDER=false

make generated_files

cd test/integration/scheduler_perf
./test-performance.sh
```

In this case, `kube-scheduler` will use predicates and priorities specified in `config/scheduler-config.json`.

Also, the performance tests will configure nodes and pods based on the contents in the configure file.

For example, I only want to enable `MatchNodeSelector` predicate during the performance test:

```json
{
"kind" : "Policy",
"apiVersion" : "v1",
"predicates" : [
  {"name" : "MatchNodeSelector"}
  ],
"priorities" : [
  {"name" : "LeastRequestedPriority", "weight" : 1},
  {"name" : "BalancedResourceAllocation", "weight" : 1},
  {"name" : "ServiceSpreadingPriority", "weight" : 1},
  {"name" : "EqualPriority", "weight" : 1}
  ],
"hardPodAffinitySymmetricWeight" : 10
}
```
Done!

The test case will configure corresponding `NodeAffinity` to make sure this predicate is respected during test process.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/component/scheduler/perf/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/test/integration/scheduler_perf/README.md?pixel)]()
