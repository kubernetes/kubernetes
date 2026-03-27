# Watch Stability and Latency Troubleshooting Guide

This guide describes how to use metrics to troubleshoot slow watch events in Kubernetes.

## Watch Event Pipeline

An event flows from etcd to the client through several stages:

1.  **etcd3 Layer (`pkg/storage/etcd3/watcher.go`)**
    *   `incomingEventChan`: Queue for events received from etcd.
    *   `transform`: Decodes and decrypts events.
    *   `processingQueue`: Concurrent processing queue.
    *   `resultChan`: Sends events to the cacher or client.

2.  **Cacher Layer (`pkg/storage/cacher/cacher.go`)**
    *   `incoming`: Queue for events dispatched to watchers.
    *   `cacheWatcher.input`: Queue for a specific watcher.
    *   `cacheWatcher.result`: Final output to client.

## Single-Threaded vs Concurrent Processing

Event processing inside `etcd3` watcher can run in two modes, controlled by the `ConcurrentWatchObjectDecode` feature gate:

1.  **Concurrent Mode**: Multiple workers decode events in parallel.
    *   **Note**: This mode is disabled by default in v1.31+ and can be enabled via the `ConcurrentWatchObjectDecode` feature gate.
    *   **Useful Metrics**:
        *   `etcd_watcher_concurrent_processing_block_duration_seconds`: Measures time spent waiting for a free worker. If high, check `etcd_watcher_transform_duration_seconds` (CPU bound) or if downstream is slow.
        *   `etcd_watcher_transform_duration_seconds`: Measures CPU cost of decoding.
2.  **Single-Threaded Mode (Default)**: Events are decoded sequentially in a single loop (`serialProcessEvents`).
    *   **Useful Metrics**:
        *   `etcd_watcher_transform_duration_seconds`: Measures CPU cost of decoding.
        *   `etcd_watcher_queue_event_block_duration_seconds`: If this is high in single-threaded mode, it can be caused by a slow `transform` (blocking the read loop).
    *   **Note**: `etcd_watcher_concurrent_processing_block_duration_seconds` is NOT used in this mode.


## Troubleshooting Slow Watches

If you observe `etcd_watcher_queue_event_block_duration_seconds` high, it means we are slow to process events *after* receiving them from etcd. Here is how to isolate the bottleneck:

### Step 1: Check `etcd_watcher_send_event_block_duration_seconds`

*   **Metric**: `etcd_watcher_send_event_block_duration_seconds`
*   **Meaning**: Time spent trying to send the event to the next layer (Cacher or Client).
*   **High Value**: This means the downstream is slow. If using Cacher, it means `watchCache` is not reading fast enough. If bypassing Cacher, it means the client connection is slow to read.
*   **Action**: Investigate Cacher or Client throughput.

### Step 2: Check `etcd_watcher_transform_duration_seconds`

*   **Metric**: `etcd_watcher_transform_duration_seconds`
*   **Meaning**: Time spent decoding/decrypting objects.
*   **High Value**: Objects are large or decryption is slow. CPU might be a bottleneck.
*   **Action**: Profile CPU for decoding/decryption cost. Check object sizes.

### Step 3: Check `etcd_watcher_concurrent_processing_block_duration_seconds`

*   **Metric**: `etcd_watcher_concurrent_processing_block_duration_seconds`
*   **Meaning**: Time spent waiting to schedule concurrent decoding when `p.processingQueue` is full.
*   **High Value**: We reached the concurrency limit (default 10). Downstream processing is slow or decoding is too slow for the event rate.
*   **Action**: Check if `transform` is slow or if downstream is slow.

### Step 4: Check Cacher Layer

*   **Metric**: `apiserver_watch_cache_incoming_queue_block_duration_seconds`
*   **Meaning**: Time spent waiting to push to `c.incoming` in Cacher (dispatching to watchers).
*   **High Value**: Cacher dispatching is slow. Too many cache watchers are slow or disconnected.
*   **Action**: Investigate individual watcher queues.

### Step 5: Check Cache Watcher Layer

*   **Metric**: `apiserver_watch_cache_cache_watcher_input_queue_block_duration_seconds`
*   **Meaning**: Time spent waiting to write to the input channel of a cache watcher.
*   **High Value**: The watcher's final output to the client is blocked (slow client connection or client processing).
*   **Action**: Check client network latency or client throughput.
