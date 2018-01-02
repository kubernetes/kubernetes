# Benchmarking etcd v2.2.0

## Physical Machines

GCE n1-highcpu-2 machine type

- 1x dedicated local SSD mounted as etcd data directory
- 1x dedicated slow disk for the OS
- 1.8 GB memory
- 2x CPUs

## etcd Cluster

3 etcd 2.2.0 members, each runs on a single machine.

Detailed versions:

```
etcd Version: 2.2.0
Git SHA: e4561dd
Go Version: go1.5
Go OS/Arch: linux/amd64
```

## Testing

Bootstrap another machine, outside of the etcd cluster, and run the [`boom` HTTP benchmark tool][boom] with a connection reuse patch to send requests to each etcd cluster member. See the [benchmark instructions][hack] for the patch and the steps to reproduce our procedures.

The performance is calulated through results of 100 benchmark rounds.

## Performance

### Single Key Read Performance

| key size in bytes | number of clients | target etcd server | average read QPS | read QPS stddev | average 90th Percentile Latency (ms) | latency stddev |
|-------------------|-------------------|--------------------|------------------|-----------------|--------------------------------------|----------------|
| 64 | 1 | leader only | 2303 | 200 | 0.49 | 0.06 |
| 64 | 64 | leader only | 15048 | 685 | 7.60 | 0.46 |
| 64 | 256 | leader only | 14508 | 434 | 29.76 | 1.05 |
| 256 | 1 | leader only | 2162 | 214 | 0.52 | 0.06 |
| 256 | 64 | leader only | 14789 | 792 | 7.69| 0.48 |
| 256 | 256 | leader only | 14424 | 512 | 29.92 | 1.42 |
| 64 | 64 | all servers | 45752 | 2048 | 2.47 | 0.14 |
| 64 | 256 | all servers | 46592 | 1273 | 10.14 | 0.59 |
| 256 | 64 | all servers | 45332 | 1847 | 2.48| 0.12 |
| 256 | 256 | all servers | 46485 | 1340 | 10.18 | 0.74 |

### Single Key Write Performance

| key size in bytes | number of clients | target etcd server | average write QPS | write QPS stddev | average 90th Percentile Latency (ms) | latency stddev |
|-------------------|-------------------|--------------------|------------------|-----------------|--------------------------------------|----------------|
| 64 | 1 | leader only | 55 | 4 | 24.51 | 13.26 |
| 64 | 64 | leader only | 2139 | 125 | 35.23 | 3.40 |
| 64 | 256 | leader only | 4581 | 581 | 70.53 | 10.22 |
| 256 | 1 | leader only | 56 | 4 | 22.37| 4.33 |
| 256 | 64 | leader only | 2052 | 151 | 36.83 | 4.20 |
| 256 | 256 | leader only | 4442 | 560 | 71.59 | 10.03 |
| 64 | 64 | all servers | 1625 | 85 | 58.51 | 5.14 |
| 64 | 256 | all servers | 4461 | 298 | 89.47 | 36.48 |
| 256 | 64 | all servers | 1599 | 94 | 60.11| 6.43 |
| 256 | 256 | all servers | 4315 | 193 | 88.98 | 7.01 |

## Performance Changes

- Because etcd now records metrics for each API call, read QPS performance seems to see a minor decrease in most scenarios. This minimal performance impact was judged a reasonable investment for the breadth of monitoring and debugging information returned.

- Write QPS to cluster leaders seems to be increased by a small margin. This is because the main loop and entry apply loops were decoupled in the etcd raft logic, eliminating several blocks between them.

- Write QPS to all members seems to be increased by a significant margin, because followers now receive the latest commit index sooner, and commit proposals more quickly.

[boom]: https://github.com/rakyll/boom
[hack]: ../../../hack/benchmark/
