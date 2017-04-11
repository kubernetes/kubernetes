# Watch Memory Usage Benchmark

*NOTE*: The watch features are under active development, and their memory usage may change as that development progresses. We do not expect it to significantly increase beyond the figures stated below.

A primary goal of etcd is supporting a very large number of watchers doing a massively large amount of watching. etcd aims to support O(10k) clients, O(100K) watch streams (O(10) streams per client) and O(10M) total watchings (O(100) watching per stream). The memory consumed by each individual watching accounts for the largest portion of etcd's overall usage, and is therefore the focus of current and future optimizations.


Three related components of etcd watch consume physical memory: each `grpc.Conn`, each watch stream, and each instance of the watching activity. `grpc.Conn` maintains the actual TCP connection and other gRPC connection state. Each `grpc.Conn` consumes O(10kb) of memory, and might have multiple watch streams attached. 

Each watch stream is an independent HTTP2 connection which consumes another O(10kb) of memory. 
Multiple watchings might share one watch stream. 

Watching is the actual struct that tracks the changes on the key-value store. Each watching should only consume < O(1kb).

```
                                          +-------+
                                          | watch |
                              +---------> | foo   |
                              |           +-------+
                       +------+-----+
                       |   stream   |
      +--------------> |            |
      |                +------+-----+     +-------+
      |                       |           | watch |
      |                       +---------> | bar   |
+-----+------+                            +-------+
|            |         +------------+
|   conn     +-------> |   stream   |
|            |         |            |
+-----+------+         +------------+
      |
      |
      |
      |                +------------+
      +--------------> |   stream   |
                       |            |
                       +------------+
```

The theoretical memory consumption of watch can be approximated with the formula:
`memory = c1 * number_of_conn + c2 * avg_number_of_stream_per_conn + c3 * avg_number_of_watch_stream`

## Testing Environment

etcd version
- git head https://github.com/coreos/etcd/commit/185097ffaa627b909007e772c175e8fefac17af3

GCE n1-standard-2 machine type
- 7.5 GB memory
- 2x CPUs

## Overall memory usage

The overall memory usage captures how much [RSS][rss] etcd consumes with the client watchers. While the result may vary by as much as 10%, it is still meaningful, since the goal is to learn about the rough memory usage and the pattern of allocations.

With the benchmark result, we can calculate roughly that `c1 = 17kb`, `c2 = 18kb` and `c3 = 350bytes`. So each additional client connection consumes 17kb of memory and each additional stream consumes 18kb of memory, and each additional watching only cause 350bytes. A single etcd server can maintain millions of watchings with a few GB of memory in normal case.


| clients | streams per client | watchings per stream | total watching | memory usage |
|---------|---------|-----------|----------------|--------------|
| 1k |  1 |   1 |   1k |   50MB |
| 2k |  1 |   1 |   2k |   90MB |
| 5k |  1 |   1 |   5k |  200MB |
| 1k | 10 |   1 |  10k |  217MB |
| 2k | 10 |   1 |  20k |  417MB |
| 5k | 10 |   1 |  50k |  980MB |
| 1k | 50 |   1 |  50k | 1001MB |
| 2k | 50 |   1 | 100k | 1960MB |
| 5k | 50 |   1 | 250k | 4700MB |
| 1k | 50 |  10 | 500k | 1171MB |
| 2k | 50 |  10 |   1M | 2371MB |
| 5k | 50 |  10 | 2.5M | 5710MB |
| 1k | 50 | 100 |   5M | 2380MB |
| 2k | 50 | 100 |  10M | 4672MB |
| 5k | 50 | 100 |  25M |  *OOM* |

[rss]: https://en.wikipedia.org/wiki/Resident_set_size
