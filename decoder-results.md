# Decoder Test Results

## Summary

| Decoder | Serializer | Run | Pod replicas | CPU per replica | Workers | Envelope buffer | Output buffer | Processing-results buffer | p50 latency | p90 latency | p99 latency | Max latency | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ConcurrentDecoder | protobuf | 1 | 10 | 4 | 10 | 100 | — | 9 | 6786.3 ms | 7757.3 ms | 7975.7 ms | — | 433 server-observed WATCHLIST requests; SlowCount 0 |
| ConcurrentDecoderWithOutputQueue | protobuf | 1 | 10 | 4 | 10 | 100 | 100 | 9 | 6776.5 ms | 7755.3 ms | 7975.5 ms | — | 434 server-observed WATCHLIST requests; SlowCount 0 |
| ConcurrentDecoderWithFixedWorkersAndOutputQueue | protobuf | 1 | 10 | 4 | 10 | 100 | 100 | 9 | 5508.0 ms | 5914.4 ms | 6725.7 ms | — | 454 server-observed WATCHLIST requests; SlowCount 0 |
| DefaultDecoder | protobuf | 1 | 10 | 4 | — | — | — | — | 7008.6 ms | 7815.5 ms | 7997.0 ms | — | Baseline; read-ahead disabled; 358 server-observed WATCHLIST requests; SlowCount 0 |
| RingDecoder | protobuf | 1 | 10 | 4 | 10 | 100 | — | — | 5506.7 ms | 5912.1 ms | 6490.0 ms | — | Ring read-ahead decoder; 461 server-observed WATCHLIST requests; SlowCount 0 |

## Configuration details

Use one subsection per decoder when the summary table does not provide enough space for implementation-specific settings.

### ConcurrentDecoder

| Setting | Value |
|---|---|
| Serializer | protobuf |
| Pod replicas | 10 |
| CPU per replica | 4 |
| Workers | 10 |
| Envelope buffer | 100 |
| Output buffer | — |
| Processing-results buffer | 9 |

### DefaultDecoder

| Setting | Value |
|---|---|
| Serializer | protobuf |
| Pod replicas | 10 |
| CPU per replica | 4 |
| Workers | — |
| Envelope buffer | — |
| Output buffer | — |
| Processing-results buffer | — |

### RingDecoder

| Setting | Value |
|---|---|
| Serializer | protobuf |
| Pod replicas | 10 |
| CPU per replica | 4 |
| Workers | 10 |
| Envelope buffer | 100 |
| Output buffer | — |
| Processing-results buffer | — |

### ConcurrentDecoderWithFixedWorkersAndOutputQueue

| Setting | Value |
|---|---|
| Serializer | protobuf |
| Pod replicas | 10 |
| CPU per replica | 4 |
| Workers | 10 |
| Envelope buffer | 100 |
| Output buffer | 100 |
| Processing-results buffer | 9 |

### ConcurrentDecoderWithOutputQueue

| Setting | Value |
|---|---|
| Serializer | protobuf |
| Pod replicas | 10 |
| CPU per replica | 4 |
| Workers | 10 |
| Envelope buffer | 100 |
| Output buffer | 100 |
| Processing-results buffer | 9 |

## Run notes

| Decoder | Serializer | Run | Source files | Observations |
|---|---|---:|---|---|
| ConcurrentDecoder | protobuf | 1 | `watchlist-logs-029d8b512-concurrent-cpu4/client-0.log`; `cl2-artifacts/APIResponsivenessPrometheus_list-benchmark_2026-10-14T11:32:04Z.json`; `cl2-artifacts/generatedConfig_list-benchmark.yaml` | Server WATCHLIST latency: p50 6786.3 ms, p90 7757.3 ms, p99 7975.7 ms; 433 requests |
| ConcurrentDecoderWithOutputQueue | protobuf | 1 | `watchlist-logs-029d8b512-concurrent-output-cpu4/client-0.log`; `cl2-artifacts/APIResponsivenessPrometheus_list-benchmark_2026-10-14T11:42:37Z.json`; `cl2-artifacts/generatedConfig_list-benchmark.yaml` | Server WATCHLIST latency: p50 6776.5 ms, p90 7755.3 ms, p99 7975.5 ms; 434 requests |
| ConcurrentDecoderWithFixedWorkersAndOutputQueue | protobuf | 1 | `watchlist-logs-029d8b512-fixed-workers-output-cpu4/client-0.log`; `cl2-artifacts/APIResponsivenessPrometheus_list-benchmark_2026-10-14T11:50:11Z.json`; `cl2-artifacts/generatedConfig_list-benchmark.yaml` | Server WATCHLIST latency: p50 5508.0 ms, p90 5914.4 ms, p99 6725.7 ms; 454 requests |
| DefaultDecoder | protobuf | 1 | `watchlist-logs-029d8b512-baseline-cpu4/client-0.log`; `cl2-artifacts/APIResponsivenessPrometheus_list-benchmark_2026-10-14T12:05:18Z.json`; `cl2-artifacts/generatedConfig_list-benchmark.yaml` | Baseline with read-ahead disabled. Server WATCHLIST latency: p50 7008.6 ms, p90 7815.5 ms, p99 7997.0 ms; 358 requests |
| RingDecoder | protobuf | 1 | `watchlist-logs-029d8b512-ring-cpu4/client-0.log`; `cl2-artifacts/APIResponsivenessPrometheus_list-benchmark_2026-10-14T12:15:15Z.json`; `cl2-artifacts/generatedConfig_list-benchmark.yaml` | Ring read-ahead decoder. Server WATCHLIST latency: p50 5506.7 ms, p90 5912.1 ms, p99 6490.0 ms; 461 requests |

## Field mapping

| Log/configuration field | Reported as |
|---|---|
| `envelopeBufferSize` | Envelope buffer |
| `outputQueueSize` | Output buffer |
| `processingResultsBufferSize` | Processing-results buffer |
| `workerCount` | Workers |
| `mediaType` / `serializer` | Serializer (short name, for example `protobuf`) |

