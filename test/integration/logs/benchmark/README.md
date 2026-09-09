# Benchmarking logging

Any major changes to the logging code, whether it is in Kubernetes or in klog,
must be benchmarked before and after the change.

## Running the benchmark

```
go test -v -bench=. -benchmem -benchtime=10s .
```

## Real log data

The files under `data` define test cases for specific aspects of formatting. To
test with a log file that represents output under some kind of real load, copy
the log file into `data/<file name>.log` and run benchmarking as described
above.  `-bench=BenchmarkLogging/<file name without .log suffix>` can be used
to benchmark just the new file.

When using `data/v<some number>/<file name>.log`, formatting will be done at
that log level. Symlinks can be created to simulating writing of the same log
data at different levels.

No such real data is included in the Kubernetes repo because of their size.
They can be found in the "artifacts" of this
https://testgrid.kubernetes.io/sig-instrumentation-tests#kind-json-logging-master
Prow job:
- `artifacts/logs/kind-control-plane/containers`
- `artifacts/logs/kind-*/kubelet.log`

With sufficient credentials, `gsutil` can be used to download everything for a job directly
into a directory that then will be used by the benchmarks automatically:

```
kubernetes$ test/integration/logs/benchmark/get-logs.sh
++ dirname test/integration/logs/benchmark/get-logs.sh
+ cd test/integration/logs/benchmark
++ latest_job
++ gsutil cat gs://kubernetes-jenkins/logs/ci-kubernetes-kind-e2e-json-logging/latest-build.txt
+ job=1618864842834186240
+ rm -rf ci-kubernetes-kind-e2e-json-logging
+ mkdir ci-kubernetes-kind-e2e-json-logging
...
```

This sets up the `data` directory so that additional test cases are available
(`BenchmarkEncoding/v3/kind-worker-kubelet/`,
`BenchmarkEncoding/kube-scheduler/`, etc.).


To clean up, use
```
git clean -fx test/integration/logs/benchmark
```

## Analyzing log data

While loading a file, some statistics about it are collected. Those are shown
when running with:

```
go test -v -bench=BenchmarkEncoding/none -run=none  .
```
