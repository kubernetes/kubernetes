# ENGINEERING_LOG

## 2026-08-01 | kubernetes/kubernetes | Fix High Cardinality on apiserver_request_terminations_total

### Metadata
- Date: 2026-08-01
- Repository: kubernetes/kubernetes
- Issue: 141007
- PR: N/A
- Category: bug fix

### Problem
The `apiserver_request_terminations_total` metric (alpha stability level) has unbounded cardinality due to labels built from unvalidated request paths.

### Root Cause
The `RequestInfo` used to populate the `group`, `version`, `resource`, and `subresource` labels in the Prometheus metric is parsed directly from raw request path segments. Since the path is unvalidated, a maliciously fuzzed request or just high volume of bad requests can create an unbounded number of Prometheus time series. This can lead to a memory leak and OOM in the API Server or the monitoring infrastructure.

### Investigation
Traced the `requestTerminationsTotal` metric in `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go`. Found that `group`, `version`, `resource`, and `subresource` labels were defined in the metric instantiation and explicitly set in `RecordRequestTermination`. For non-resource requests, the raw `requestInfo.Path` was being supplied into the `subresource` label bucket, demonstrating clear unbound cardinality.

### Architecture Learned
The API Server metric emission layer for early self-defense terminations cannot trust standard parsing, as it might happen before route resolution or with completely invalid paths. Metric labels in this layer must be highly bounded and vetted.

### Files Inspected
- `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go`

### Solution
Removed the unvalidated labels (`group`, `version`, `resource`, `subresource`) from the `requestTerminationsTotal` metric initialization. Updated the `RecordRequestTermination` func to pass only `verb`, `scope`, `component`, and `code`. 

### Alternatives Considered
- `Validate labels against OpenAPI schema`: Adds significant computational overhead to a self-defense termination path which must be fast.

### Why This Approach
This is the simplest fix, adds no performance overhead, completely bounds the metric cardinality, and perfectly aligns with the direct request made by the API Machinery maintainer on the GitHub issue.

### Tests
- `make test WHAT=./staging/src/k8s.io/apiserver/pkg/endpoints/metrics`: Passed

### Edge Cases
- `Valid requests hitting termination limit`: Loss of granularity on exact resources hitting self-defense. However, overall metric trends for component/verb remain.

### Files Modified
- `staging/src/k8s.io/apiserver/pkg/endpoints/metrics/metrics.go`

### Review Feedback
None yet

### Lessons Learned
- Always consider the cardinality of Prometheus metrics on edge/ingress components that accept untrusted input.
- Metrics emitted in the "unauthenticated/early rejection" phase shouldn't use request-derived labels directly.

### Architecture Knowledge
- Kubernetes API Server uses `staging/src/k8s.io/apiserver` as the core foundation for apiservers. `metrics.go` centrally manages all Prometheus metric definitions.

### Skills Practiced
- architecture tracing
- security/hardening
- bug fixing

### Resume Bullet
- Fixed high-cardinality unbounded memory leak vulnerability in Kubernetes API Server Prometheus metrics by pruning unvalidated request labels.

### STAR Interview Story
Situation: The Kubernetes API Server had an unbounded cardinality issue on a self-defense metric.
Task: Prevent memory leaks and OOM crashes from malicious request path fuzzing.
Action: Located the unvalidated label emission in the metrics registry and simplified the labels to a strict, bounded set requested by SIG API Machinery.
Result: Mitigated a major DoS vector on the API Server and Prometheus scraping infrastructure with zero overhead added to the request path.

### Personal Notes
- Always check the cardinality of `compbasemetrics.NewCounterVec` keys. Anything mapping from a raw URL path is dangerous.
