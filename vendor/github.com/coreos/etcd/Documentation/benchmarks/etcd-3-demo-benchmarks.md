## Physical machines

GCE n1-highcpu-2 machine type

- 1x dedicated local SSD mounted under /var/lib/etcd
- 1x dedicated slow disk for the OS
- 1.8 GB memory
- 2x CPUs
- etcd version 2.2.0

## etcd Cluster

1 etcd member running in v3 demo mode

## Testing

Use [etcd v3 benchmark tool][etcd-v3-benchmark].

## Performance

### reading one single key

| key size in bytes | number of clients | read QPS | 90th Percentile Latency (ms) |
|-------------------|-------------------|----------|---------------|
| 256               | 1                 | 2716  | 0.4      |
| 256               | 64                | 16623 | 6.1      |
| 256               | 256               | 16622 | 21.7     |

The performance is nearly the same as the one with empty server handler.

### reading one single key after putting

| key size in bytes | number of clients | read QPS | 90th Percentile Latency (ms) |
|-------------------|-------------------|----------|---------------|
| 256               | 1                 | 2269  | 0.5      |
| 256               | 64                | 13582 | 8.6      |
| 256               | 256               | 13262 | 47.5     |

The performance with empty server handler is not affected by one put. So the
performance downgrade should be caused by storage package.

[etcd-v3-benchmark]: ../../tools/benchmark/
