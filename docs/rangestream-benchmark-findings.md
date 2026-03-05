# RangeStream Watch Cache Initialization: Benchmark Findings

## Overview

This document captures the results of benchmarking the RangeStream feature for Kubernetes watch cache initialization. RangeStream replaces the paginated unary `Range` RPC with a server-streaming `RangeStream` RPC for loading data from etcd during kube-apiserver startup.

**Test setup**: 500,000 pods (~1.7KB each, ~850MB total) in an external etcd (custom build 3.7.0-alpha.0 with RangeStream support).

## Baseline Results

5-run averages, no instrumentation overhead:

| Config | s/init (mean) | stdev |
|--------|--------------|-------|
| Paginated (RangeStream=off) | 5.535s | 0.083s |
| Stream (RangeStream=on) | 3.751s | 0.121s |
| **Delta** | **-32%** | |

For context, isolated etcd benchmarks show RangeStream at 93 MB/s vs paginated at 38 MB/s (2.4x faster). The Kubernetes integration only realizes 32% of this because the bottleneck is in the processing pipeline, not the network.

## Pipeline Architecture

Watch cache initialization flows through this pipeline:

```
sync() → queueEvent() → incomingEventChan(100) → processEvents(decode) → resultChan(100) → reflector → watchCache
```

- `sync()`: Fetches data from etcd (paginated Get or RangeStream)
- `queueEvent()`: Pushes raw KV events to `incomingEventChan` (buffer=100)
- `processEvents()`: Reads events, calls `transform()` (protobuf decode), sends to `resultChan` (buffer=100)
- Reflector: Reads decoded `watch.Event` objects from `resultChan`, adds to temporary store, eventually calls `watchCache.Replace()`

By default, decode is **serial** (single goroutine). `ConcurrentWatchObjectDecode` feature gate (Beta, default false) enables 10 parallel decode workers.

## Instrumented Breakdown

Adding per-phase timing to `sync()` and `serialProcessEvents()` reveals where time is spent:

| Metric | Paginated | Stream |
|--------|-----------|--------|
| **s/init** | **5.958s** | **3.990s** |
| Data fetch (etcdGet / streamRecv) | 2.250s | 0.039s |
| Queue backpressure (sync blocked) | 3.673s | 3.917s |
| Decode (transform) | 3.210s | 3.323s |
| Decode starvation (chanWait) | 2.395s | 0.253s |
| Send to resultChan | 0.290s | 0.351s |

### Key Observations

1. **Stream data arrives ~58x faster** (0.039s vs 2.250s). From the Kubernetes client's perspective, all 500k items are essentially instantly available — gRPC/OS buffers hold the data.

2. **Decode time is identical** (~3.3s both paths). The same `TransformFromStorage` + `decodeObj` (protobuf unmarshal) work happens regardless of transport.

3. **The 32% improvement comes from decode starvation**: With paginated, the decode goroutine starves between page fetches (chanWait=2.4s). With stream, decode is almost always fed (chanWait=0.25s).

4. **Both paths are bottlenecked by serial decode**: ~3.9s of queue backpressure means sync() is blocked half the time waiting for the 100-item channel to drain.

### etcd Server-Side Behavior

The etcd server's `RangeStream` implementation does NOT stream individual KVs. It performs **paginated reads internally** with adaptive batch sizing:

- Starts at `r.Limit = 10` items per response
- Doubles when response size < `MaxRequestBytes/2` (~750KB)
- At ~1.7KB per pod, caps around ~440 items per response

This means the stream produces hundreds of small-to-medium responses (10-440 items each), compared to paginated which uses a fixed 10,000-item page size.

## Optimization Experiments

### 1. ConcurrentWatchObjectDecode (existing feature gate)

Enables 10 parallel decode workers in `processEvents()`.

| Config | s/init | Change |
|--------|--------|--------|
| Paginated + Serial | 6.756s | baseline |
| Stream + Serial | 3.946s | baseline |
| Paginated + Concurrent | 5.343s | -21% |
| **Stream + Concurrent** | **3.932s** | **0%** |

**Finding**: ConcurrentDecode does not help the stream path. The decode workers run in parallel but an **order-preserving collector** (`collectEventProcessing`) re-serializes results back into submission order before sending to `resultChan`. This negates the parallelism — if worker 1 is slow, workers 2-10 block waiting even though they've already finished. Combined with the single-threaded reflector downstream on `resultChan(100)`, the bottleneck shifts from decode to the ordered collector + reflector consumer.

It helps paginated modestly (-21%) because the natural pauses between pages give the reflector more time to consume.

### 2. Larger Channel Buffers (100 → 10,000)

Increasing `incomingBufSize` and `outgoingBufSize` from 100 to 10,000.

| Config | s/init (buf=100) | s/init (buf=10k) | Change |
|--------|-----------------|------------------|--------|
| Paginated | 6.756s | 3.902s | -42% |
| Stream | 3.946s | 3.820s | -3% |

**Finding**: Huge win for paginated — eliminates ~5,000 buffer fill/drain cycles. Each 10k-item page fits entirely in the buffer, allowing decode to run continuously. Stream barely changes because it's already near the decode floor; data arrives instantly regardless of buffer size.

### 3. Batch Channel Sends (`chan []*event`)

Changed `incomingEventChan` from `chan *event` to `chan []*event`, sending entire page/response batches in a single channel operation.

| Config | s/init (per-item) | s/init (batch) | Change |
|--------|------------------|----------------|--------|
| Paginated | 5.958s | 2.521s | -58% |
| Stream | 3.990s | 3.273s | -18% |

**Finding**: Massive win for paginated — reduces 500,000 channel operations to 50 (one per 10k-item page). Paginated becomes **faster than stream** (2.5s vs 3.3s) because its natural 10k-item pages make more efficient batches than stream's small adaptive batches (10-440 items).

### 4. Bypass Event Pipeline (parallel decode in sync)

Replaced the entire channel pipeline for the stream path. Instead of going through `incomingEventChan` → `processEvents` → `resultChan`, decode directly in `sync()` with parallel worker goroutines. Unlike `ConcurrentWatchObjectDecode`, there is no order-preserving collector — workers send decoded events directly. This is safe because initial cache fill uses `watchCache.Replace()`, so ordering doesn't matter.

#### 4a. Collect-then-decode (sequential phases)

Receive all stream data, then parallel decode, then send to resultChan.

| Metric | Value |
|--------|-------|
| s/init | 2.938s |
| recv | 0.742s |
| decode (10 workers) | 1.794s |
| send | 0.402s |

#### 4b. Pipelined decode (overlapped recv + decode)

Feed decode workers as items arrive from the stream, overlapping network and CPU.

| Metric | Value |
|--------|-------|
| s/init | **2.597s** |
| recv+decode (overlapped) | 2.218s |
| send | 0.376s |

Worker count scaling:

| Workers | s/init |
|---------|--------|
| 5 | 2.788s |
| **10** | **2.597s** |
| 20 | 2.669s |
| 50 | 2.852s |

**Finding**: 10 workers is optimal. More workers increases contention (likely Go allocator and protobuf codec). Serial decode takes 3.3s; 10 parallel workers achieve ~2.2s (only ~1.5x speedup), suggesting significant shared-state contention in the decode path.

## Summary

| Configuration | s/init | vs Paginated baseline |
|---|---|---|
| Paginated baseline | 5.535s | — |
| Stream baseline | 3.751s | -32% |
| Stream + pipelined parallel decode | **2.597s** | **-53%** |

## Remaining Bottlenecks

1. **Send phase (0.4s)**: Even after parallel decode, 500k decoded events must be sent through `resultChan` one-at-a-time to the reflector. Bulk insertion into the watch cache would eliminate this.

2. **Decode contention (~2.2s with 10 workers vs 3.3s serial)**: Only ~1.5x speedup from 10 workers suggests contention in `decodeObj` (protobuf codec, Go allocator, or `runtime.Object` interface dispatch).

3. **etcd server adaptive batching**: Starts at `r.Limit=10` and doubles slowly, producing many small gRPC responses. A higher initial limit (e.g., 10,000) would reach peak throughput faster and enable more efficient batch processing on the client side.

4. **GetStream client buffer**: The etcd client's `GetStream` uses `make(chan MaybeRangeStreamResponse, 1)`. While not a bottleneck today (data arrives in 0.039s), a larger buffer could help under higher latency conditions.

## Files

- `staging/src/k8s.io/apiserver/pkg/storage/etcd3/watcher.go` — sync(), processEvents, queueEvent
- `staging/src/k8s.io/apiserver/pkg/storage/etcd3/metrics/metrics.go` — WatchCacheInitialization metric
- `test/integration/apiserver/rangestream_load_benchmark_test.go` — Large-scale benchmark
- `test/integration/apiserver/rangestream_benchmark_test.go` — Small-scale benchmark, metric helpers
- `vendor/go.etcd.io/etcd/client/v3/kv.go` — GetStream client implementation
- `vendor/go.etcd.io/etcd/server/v3/etcdserver/v3_server.go` — Server-side RangeStream (adaptive batching)
