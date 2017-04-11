# Performance

## Understanding performance

etcd provides stable, sustained high performance. Two factors define performance: latency and throughput. Latency is the time taken to complete an operation. Throughput is the total operations completed within some time period. Usually average latency increases as the overall throughput increases when etcd accepts concurrent client requests. In common cloud environments, like a standard `n-4` on Google Compute Engine (GCE) or a comparable machine type on AWS, a three member etcd cluster finishes a request in less than one millisecond under light load, and can complete more than 30,000 requests per second under heavy load.

etcd uses the Raft consensus algorithm to replicate requests among members and reach agreement. Consensus performance, especially commit latency, is limited by two physical constraints: network IO latency and disk IO latency. The minimum time to finish an etcd request is the network Round Trip Time (RTT) between members, plus the time `fdatasync` requires to commit the data to permanant storage. The RTT within a datacenter may be as long as several hundred microseconds. A typical RTT within the United States is around 50ms, and can be as slow as 400ms between continents. The typical fdatasync latency for a spinning disk is about 10ms. For SSDs, the latency is often lower than 1ms. To increase throughput, etcd batches multiple requests together and submits them to Raft. This batching policy lets etcd attain high throughput despite heavy load.

There are other sub-systems which impact the overall performance of etcd. Each serialized etcd request must run through etcd’s boltdb-backed MVCC storage engine, which usually takes tens of microseconds to finish. Periodically etcd incrementally snapshots its recently applied requests, merging them back with the previous on-disk snapshot. This process may lead to a latency spike. Although this is usually not a problem on SSDs, it may double the observed latency on HDD. Likewise, inflight compactions can impact etcd’s performance. Fortunately, the impact is often insignificant since the compaction is staggered so it does not compete for resources with regular requests. The RPC system, gRPC, gives etcd a well-defined, extensible API, but it also introduces additional latency, especially for local reads.

## Benchmarks

Benchmarking etcd performance can be done with the [benchmark](https://github.com/coreos/etcd/tree/master/tools/benchmark) CLI tool included with etcd.

For some baseline performance numbers, we consider a three member etcd cluster with the following hardware configuration:

- Google Cloud Compute Engine
- 3 machines of 8 vCPUs + 16GB Memory + 50GB SSD
- 1 machine(client) of 16 vCPUs + 30GB Memory + 50GB SSD
- Ubuntu 15.10
- etcd v3 master branch (commit SHA d8f325d), Go 1.6.2

With this configuration, etcd can approximately write:

| Number of keys | Key size in bytes | Value size in bytes | Number of connections | Number of clients | Target etcd server | Average write QPS | Average latency per request | Memory |
|----------------|-------------------|---------------------|-----------------------|-------------------|--------------------|-------------------|-----------------------------|--------|
| 10,000 | 8 | 256 | 1 | 1 | leader only | 525 | 2ms | 35 MB |
| 100,000 | 8 | 256 | 100 | 1000 | leader only | 25,000 | 30ms | 35 MB |
| 100,000 | 8 | 256 | 100 | 1000 | all members | 33,000 | 25ms | 35 MB |

Sample commands are:

```
# assuming IP_1 is leader, write requests to the leader
benchmark --endpoints={IP_1} --conns=1 --clients=1 \
    put --key-size=8 --sequential-keys --total=10000 --val-size=256
benchmark --endpoints={IP_1} --conns=100 --clients=1000 \
    put --key-size=8 --sequential-keys --total=100000 --val-size=256

# write to all members
benchmark --endpoints={IP_1},{IP_2},{IP_3} --conns=100 --clients=1000 \
    put --key-size=8 --sequential-keys --total=100000 --val-size=256
```

Linearizable read requests go through a quorum of cluster members for consensus to fetch the most recent data. Serializable read requests are cheaper than linearizable reads since they are served by any single etcd member, instead of a quorum of members, in exchange for possibly serving stale data. etcd can read: 

| Number of requests | Key size in bytes | Value size in bytes | Number of connections | Number of clients | Consistency | Average latency per request | Average read QPS |
|--------------------|-------------------|---------------------|-----------------------|-------------------|-------------|-----------------------------|------------------|
| 10,000 | 8 | 256 | 1 | 1 | Linearizable | 2ms | 560 |
| 10,000 | 8 | 256 | 1 | 1 | Serializable | 0.4ms | 7,500 |
| 100,000 | 8 | 256 | 100 | 1000 | Linearizable | 15ms | 43,000 |
| 100,000 | 8 | 256 | 100 | 1000 | Serializable | 9ms | 93,000 |

Sample commands are:

```
# Linearizable read requests
benchmark --endpoints={IP_1},{IP_2},{IP_3} --conns=1 --clients=1 \
    range YOUR_KEY --consistency=l --total=10000
benchmark --endpoints={IP_1},{IP_2},{IP_3} --conns=100 --clients=1000 \
    range YOUR_KEY --consistency=l --total=100000

# Serializable read requests for each member and sum up the numbers
for endpoint in {IP_1} {IP_2} {IP_3}; do
    benchmark --endpoints=$endpoint --conns=1 --clients=1 \
        range YOUR_KEY --consistency=s --total=10000
done
for endpoint in {IP_1} {IP_2} {IP_3}; do
    benchmark --endpoints=$endpoint --conns=100 --clients=1000 \
        range YOUR_KEY --consistency=s --total=100000
done
```

We encourage running the benchmark test when setting up an etcd cluster for the first time in a new environment to ensure the cluster achieves adequate performance; cluster latency and throughput can be sensitive to minor environment differences.