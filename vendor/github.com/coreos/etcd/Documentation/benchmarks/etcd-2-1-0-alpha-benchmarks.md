## Physical machines

GCE n1-highcpu-2 machine type

- 1x dedicated local SSD mounted under /var/lib/etcd
- 1x dedicated slow disk for the OS
- 1.8 GB memory
- 2x CPUs
- etcd version 2.1.0 alpha

## etcd Cluster

3 etcd members, each runs on a single machine

## Testing

Bootstrap another machine and use the [hey HTTP benchmark tool][hey] to send requests to each etcd member. Check the [benchmark hacking guide][hack-benchmark] for detailed instructions.

## Performance

### reading one single key

| key size in bytes | number of clients | target etcd server | read QPS | 90th Percentile Latency (ms) |
|-------------------|-------------------|--------------------|----------|---------------|
| 64                | 1                 | leader only        | 1534     | 0.7        |
| 64                | 64                | leader only        | 10125    | 9.1      |
| 64                | 256               | leader only        | 13892    | 27.1      |
| 256               | 1                 | leader only        | 1530     | 0.8       |
| 256               | 64                | leader only        | 10106    | 10.1      |
| 256               | 256               | leader only        | 14667    | 27.0      |
| 64                | 64                | all servers        | 24200    | 3.9      |
| 64                | 256               | all servers        | 33300    | 11.8      |
| 256               | 64                | all servers        | 24800    | 3.9      |
| 256               | 256               | all servers        | 33000    | 11.5      |

### writing one single key

| key size in bytes | number of clients | target etcd server | write QPS | 90th Percentile Latency (ms) |
|-------------------|-------------------|--------------------|-----------|---------------|
| 64                | 1                 | leader only        | 60        | 21.4 |
| 64                | 64                | leader only        | 1742      | 46.8 |
| 64                | 256               | leader only        | 3982      | 90.5 |
| 256               | 1                 | leader only        | 58        | 20.3 |
| 256               | 64                | leader only        | 1770      | 47.8 |
| 256               | 256               | leader only        | 4157      | 105.3 |
| 64                | 64                | all servers        | 1028      | 123.4 |
| 64                | 256               | all servers        | 3260      | 123.8 |
| 256               | 64                | all servers        | 1033      | 121.5 |
| 256               | 256               | all servers        | 3061      | 119.3 |

[hey]: https://github.com/rakyll/hey
[hack-benchmark]: https://github.com/coreos/etcd/tree/master/hack/benchmark
