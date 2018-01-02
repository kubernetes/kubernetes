## Physical machines

GCE n1-highcpu-2 machine type

- 1x dedicated local SSD mounted under /var/lib/etcd
- 1x dedicated slow disk for the OS
- 1.8 GB memory
- 2x CPUs

## etcd Cluster

3 etcd 2.2.0-rc members, each runs on a single machine.

Detailed versions:

```
etcd Version: 2.2.0-alpha.1+git
Git SHA: 59a5a7e
Go Version: go1.4.2
Go OS/Arch: linux/amd64
```

Also, we use 3 etcd 2.1.0 alpha-stage members to form cluster to get base performance. etcd's commit head is at [c7146bd5][c7146bd5], which is the same as the one that we use in [etcd 2.1 benchmark][etcd-2.1-benchmark].

## Testing

Bootstrap another machine and use the [boom HTTP benchmark tool][boom] to send requests to each etcd member. Check the [benchmark hacking guide][hack-benchmark] for detailed instructions.

## Performance

### reading one single key

| key size in bytes | number of clients | target etcd server | read QPS | 90th Percentile Latency (ms) |
|-------------------|-------------------|--------------------|----------|---------------|
| 64                | 1                 | leader only        | 2804 (-5%) | 0.4 (+0%) |
| 64                | 64                | leader only        | 17816 (+0%) | 5.7 (-6%) |
| 64                | 256               | leader only        | 18667 (-6%) | 20.4 (+2%) |
| 256               | 1                 | leader only        | 2181 (-15%) | 0.5 (+25%) |
| 256               | 64                | leader only        | 17435 (-7%) | 6.0 (+9%) |
| 256               | 256               | leader only        | 18180 (-8%) | 21.3 (+3%) |
| 64                | 64                | all servers        | 46965 (-4%) | 2.1 (+0%) |
| 64                | 256               | all servers        | 55286 (-6%) | 7.4 (+6%) |
| 256               | 64                | all servers        | 46603 (-6%) | 2.1 (+5%) |
| 256               | 256               | all servers        | 55291 (-6%) | 7.3 (+4%) |

### writing one single key

| key size in bytes | number of clients | target etcd server | write QPS | 90th Percentile Latency (ms) |
|-------------------|-------------------|--------------------|-----------|---------------|
| 64                | 1                 | leader only        | 76 (+22%)  | 19.4 (-15%) |
| 64                | 64                | leader only        | 2461 (+45%) | 31.8 (-32%) |
| 64                | 256               | leader only        | 4275 (+1%) | 69.6 (-10%) |
| 256               | 1                 | leader only        | 64 (+20%)  | 16.7 (-30%) |
| 256               | 64                | leader only        | 2385 (+30%) | 31.5 (-19%) |
| 256               | 256               | leader only        | 4353 (-3%) | 74.0 (+9%) |
| 64                | 64                | all servers        | 2005 (+81%) | 49.8 (-55%) |
| 64                | 256               | all servers        | 4868 (+35%) | 81.5 (-40%) |
| 256               | 64                | all servers        | 1925 (+72%) | 47.7 (-59%) |
| 256               | 256               | all servers        | 4975 (+36%) | 70.3 (-36%) |

### performance changes explanation

- read QPS in most scenarios is decreased by 5~8%. The reason is that etcd records store metrics for each store operation. The metrics is important for monitoring and debugging, so this is acceptable.

- write QPS to leader is increased by 20~30%. This is because we decouple raft main loop and entry apply loop, which avoids them blocking each other.

- write QPS to all servers is increased by 30~80% because follower could receive latest commit index earlier and commit proposals faster.

[boom]: https://github.com/rakyll/boom
[c7146bd5]: https://github.com/coreos/etcd/commits/c7146bd5f2c73716091262edc638401bb8229144
[etcd-2.1-benchmark]: etcd-2-1-0-alpha-benchmarks.md
[hack-benchmark]: ../../../hack/benchmark/
