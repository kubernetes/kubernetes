## Physical machine

GCE n1-standard-2 machine type

- 1x dedicated local SSD mounted under /var/lib/etcd
- 1x dedicated slow disk for the OS
- 7.5 GB memory
- 2x CPUs

## etcd

```
etcd Version: 2.2.0-rc.0+git
Git SHA: 103cb5c
Go Version: go1.5
Go OS/Arch: linux/amd64
```

## Testing

Start 3-member etcd cluster, each of which uses 2 cores.

The length of key name is always 64 bytes, which is a reasonable length of average key bytes.

## Memory Maximal Usage

- etcd may use maximal memory if one follower is dead and the leader keeps sending snapshots.
- `max RSS` is the maximal memory usage recorded in 3 runs.

| value bytes | key number  | data size(MB) | max RSS(MB) | max RSS/data rate on leader |
|-------------|-------------|---------------|-------------|-----------------------------|
| 128  | 50000  | 6 | 433 | 72x |
| 128  | 100000 | 12 | 659 | 54x |
| 128  | 200000 | 24 | 1466 | 61x |
| 1024 | 50000  | 48 | 1253 | 26x |
| 1024 | 100000 | 96 | 2344 | 24x |
| 1024 | 200000 | 192 | 4361 | 22x |

## Data Size Threshold

- When etcd reaches data size threshold, it may trigger leader election easily and drop part of proposals.
- At most cases, etcd cluster should work smoothly if it doesn't hit the threshold. If it doesn't work well due to insufficient resources, you need to decrease its data size.

| value bytes | key number limitation | suggested data size threshold(MB) | consumed RSS(MB) |
|-------------|-----------------------|-----------------------------------|------------------|
| 128 | 400K | 48 | 2400 |
| 1024 | 300K | 292 | 6500 |
