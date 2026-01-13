# Package: metrics

## Purpose
Provides Prometheus metrics for monitoring external JWT signer operations, including key fetches and token signing requests.

## Key Metrics
- `apiserver_externaljwt_fetch_keys_success_timestamp` - Timestamp of last successful key fetch
- `apiserver_externaljwt_fetch_keys_data_timestamp` - Data timestamp from external signer
- `apiserver_externaljwt_fetch_keys_request_total` - Total key fetch attempts by status code
- `apiserver_externaljwt_sign_request_total` - Total token signing attempts by status code
- `apiserver_externaljwt_request_duration_seconds` - Request latency histogram

## Key Functions
- `RegisterMetrics()` - Registers all external JWT metrics with Prometheus
- `RecordFetchKeysAttempt()` - Records a key fetch attempt with result
- `RecordTokenGenAttempt()` - Records a token generation attempt with result
- `RecordKeyDataTimeStamp()` - Records the data timestamp from external signer
- `OuboundRequestMetricsInterceptor()` - gRPC interceptor for recording request metrics

## Design Patterns
- Uses sync.Once for safe metric registration
- gRPC error code extraction for metric labels
- Interceptor pattern for automatic metric collection on gRPC calls
