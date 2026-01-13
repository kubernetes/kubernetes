# Metrics Eligibility Report - PR #1: component-base Metrics

This report verifies the Alpha → Beta graduation requirements for the 4 metrics in PR #1 (component-base).

## Alpha → Beta Graduation Requirements Checklist

For each metric, we verify:
1. **Help Text**: Exists and is accurate
2. **Registration/Emission**: Where the metric is registered and emitted
3. **Labels**: What labels the metric has
4. **Tests**: Whether tests exist that validate registration/emission + labels/values

---

## 1. `kubernetes_build_info`

**File Path**: `staging/src/k8s.io/component-base/metrics/version.go`

### (1) Help Text
✅ **EXISTS AND ACCURATE**

**Help Text**: `"A metric with a constant '1' value labeled by major, minor, git version, git commit, git tree state, build date, Go version, and compiler from which Kubernetes was built, and platform on which it is running."`

**Analysis**: The help text clearly describes:
- The metric value (constant '1')
- All labels (major, minor, git_version, git_commit, git_tree_state, build_date, go_version, compiler, platform)
- The purpose (build and version info)

### (2) Registration/Emission
✅ **REGISTERED AND EMITTED**

**Registration Location**: 
- `RegisterBuildInfo(r KubeRegistry)` function in `staging/src/k8s.io/component-base/metrics/version.go:33`
- Also auto-registered via `init()` in `staging/src/k8s.io/component-base/metrics/prometheus/version/metrics.go:37`

**Emission Location**: 
- Line 36 in `staging/src/k8s.io/component-base/metrics/version.go`:
  ```go
  buildInfo.WithLabelValues(info.Major, info.Minor, info.GitVersion, info.GitCommit, info.GitTreeState, info.BuildDate, info.GoVersion, info.Compiler, info.Platform).Set(1)
  ```

**Analysis**: The metric is registered via `r.MustRegister(buildInfo)` and then immediately emitted with label values from `version.Get()`.

### (3) Labels
✅ **VERIFIED**

**Labels**: `["major", "minor", "git_version", "git_commit", "git_tree_state", "build_date", "go_version", "compiler", "platform"]`

**Source**: Line 28 in `staging/src/k8s.io/component-base/metrics/version.go`

**Label Values Source**: From `version.Get()` struct fields: `info.Major`, `info.Minor`, `info.GitVersion`, `info.GitCommit`, `info.GitTreeState`, `info.BuildDate`, `info.GoVersion`, `info.Compiler`, `info.Platform`

### (4) Tests
❌ **NEEDS_CONFIRMATION**

**Status**: No test file found specifically for `kubernetes_build_info` metric.

**Search Performed**:
- Searched for test files in `staging/src/k8s.io/component-base/metrics/` directory
- Found `version_parser_test.go` but it only tests version parsing logic, not the metric itself
- Searched codebase-wide for tests referencing `kubernetes_build_info`, `RegisterBuildInfo`, or `buildInfo` metric
- No test found that validates:
  - Metric registration
  - Metric emission
  - Label values
  - Expected metric values

**Missing Evidence**: 
- No test that calls `RegisterBuildInfo()` and verifies the metric is registered
- No test that validates the metric is emitted with correct label values
- No test that validates the metric value is set to 1
- No test that validates all 9 labels are present and have expected values

---

## 2. `rest_client_request_duration_seconds`

**File Path**: `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go`

### (1) Help Text
✅ **EXISTS AND ACCURATE**

**Help Text**: `"Request latency in seconds. Broken down by verb, and host."`

**Analysis**: The help text clearly describes:
- What the metric measures (request latency in seconds)
- How it's partitioned (by verb and host)

### (2) Registration/Emission
✅ **REGISTERED AND EMITTED**

**Registration Location**: 
- `init()` function in `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go:200`
- Registered via: `legacyregistry.MustRegister(requestLatency)`

**Emission Location**: 
- Via `latencyAdapter.Observe()` method (lines 228-234):
  ```go
  func (l *latencyAdapter) Observe(ctx context.Context, verb string, u url.URL, latency time.Duration) {
      l.m.WithContext(ctx).WithLabelValues(verb, u.Host).Observe(latency.Seconds())
  }
  ```
- The adapter is registered with the metrics system at line 214: `RequestLatency: &latencyAdapter{m: requestLatency}`

**Analysis**: The metric is registered in `init()` and emitted through the adapter pattern when HTTP requests are made via the rest client.

### (3) Labels
✅ **VERIFIED**

**Labels**: `["verb", "host"]`

**Source**: Line 41 in `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go`

**Label Values Source**: 
- `verb`: From the `verb` parameter in `latencyAdapter.Observe()`
- `host`: From `u.Host` (URL.Host field) in `latencyAdapter.Observe()`

### (4) Tests
❌ **NEEDS_CONFIRMATION**

**Status**: No test found specifically for `rest_client_request_duration_seconds` metric.

**Search Performed**:
- Found test file: `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics_test.go`
- The test file `TestClientGOMetrics()` only tests:
  - `rest_client_requests_total` (lines 37-49)
  - `rest_client_request_retries_total` (lines 50-62)
- No test case for `rest_client_request_duration_seconds` (requestLatency)

**Missing Evidence**:
- No test that validates `requestLatency` metric registration
- No test that validates metric emission via `latencyAdapter.Observe()`
- No test that validates the "verb" and "host" labels
- No test that validates latency values are correctly recorded
- No test that validates histogram buckets are working correctly

---

## 3. `rest_client_requests_total`

**File Path**: `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go`

### (1) Help Text
✅ **EXISTS AND ACCURATE**

**Help Text**: `"Number of HTTP requests, partitioned by status code, method, and host."`

**Analysis**: The help text clearly describes:
- What the metric measures (number of HTTP requests)
- How it's partitioned (by status code, method, and host)

### (2) Registration/Emission
✅ **REGISTERED AND EMITTED**

**Registration Location**: 
- `init()` function in `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go:204`
- Registered via: `legacyregistry.MustRegister(requestResult)`

**Emission Location**: 
- Via `resultAdapter.Increment()` method (lines 252-258):
  ```go
  func (r *resultAdapter) Increment(ctx context.Context, code, method, host string) {
      r.m.WithContext(ctx).WithLabelValues(code, method, host).Inc()
  }
  ```
- The adapter is registered with the metrics system at line 219: `RequestResult: &resultAdapter{requestResult}`

**Analysis**: The metric is registered in `init()` and emitted through the adapter pattern when HTTP requests complete.

### (3) Labels
✅ **VERIFIED**

**Labels**: `["code", "method", "host"]`

**Source**: Line 94 in `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go`

**Label Values Source**: 
- `code`: HTTP status code (string) from `resultAdapter.Increment()` parameter
- `method`: HTTP method (string) from `resultAdapter.Increment()` parameter  
- `host`: Host name (string) from `resultAdapter.Increment()` parameter

**Note**: The help text says "status code, method, and host" which accurately describes the labels.

### (4) Tests
✅ **EXISTS**

**Test File**: `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics_test.go`

**Test Function**: `TestClientGOMetrics()` (lines 29-87)

**Test Case**: Lines 37-49 test `rest_client_requests_total`

**What the Test Validates**:
- ✅ **Registration**: Test uses the metric after it's registered in `init()` (line 67 comment: "no need to register the metrics here, since the init function of the package registers all the client-go metrics")
- ✅ **Emission**: Calls `metrics.RequestResult.Increment(context.TODO(), "200", "POST", "www.foo.com")` (line 42)
- ✅ **Labels**: Validates expected output with labels: `code="200"`, `method="POST"`, `host="www.foo.com"` (line 47)
- ✅ **Values**: Validates counter value is 1 after increment (line 47)
- ✅ **Help Text**: Validates help text in output matches expected format (line 45)

**Test Output Validation**: Uses `testutil.GatherAndCompare()` to validate the metric output matches expected Prometheus format including help text, type, labels, and value.

**Analysis**: The test fully satisfies the Alpha → Beta graduation requirement for testing.

---

## 4. `running_managed_controllers`

**File Path**: `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go`

### (1) Help Text
✅ **EXISTS AND ACCURATE**

**Help Text**: `"Indicates where instances of a controller are currently running"`

**Analysis**: The help text clearly describes:
- What the metric indicates (where controller instances are running)
- The metric is documented in auto-generated documentation at `test/instrumentation/documentation/documentation.md:3332`

**Note**: The help text could be more descriptive (e.g., mentioning the labels), but it accurately describes the metric's purpose.

### (2) Registration/Emission
✅ **REGISTERED AND EMITTED**

**Registration Location**: 
- `Register()` function in `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go:52`
- Registered via: `legacyregistry.MustRegister(controllerInstanceCount)` (line 54)
- Uses `sync.Once` to ensure registration happens only once

**Emission Locations**: 
- `ControllerStarted()` method (lines 61-63):
  ```go
  func (a *ControllerManagerMetrics) ControllerStarted(name string) {
      controllerInstanceCount.With(k8smetrics.Labels{"name": name, "manager": a.manager}).Set(float64(1))
  }
  ```
- `ControllerStopped()` method (lines 66-68):
  ```go
  func (a *ControllerManagerMetrics) ControllerStopped(name string) {
      controllerInstanceCount.With(k8smetrics.Labels{"name": name, "manager": a.manager}).Set(float64(0))
  }
  ```

**Usage**: The metric is used by controller managers (e.g., kube-controller-manager, cloud-provider controllers) to track which controllers are running.

**Analysis**: The metric is registered via `Register()` and emitted when controllers start (Set to 1) or stop (Set to 0).

### (3) Labels
✅ **VERIFIED**

**Labels**: `["name", "manager"]`

**Source**: Line 34 in `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go`

**Label Values Source**: 
- `name`: Controller name (string) from `ControllerStarted()`/`ControllerStopped()` parameter
- `manager`: Manager name (string) from `ControllerManagerMetrics.manager` field (set when creating `ControllerManagerMetrics` via `NewControllerManagerMetrics()`)

**Documentation Confirmation**: The auto-generated documentation at `test/instrumentation/documentation/documentation.md:3337` confirms labels are `["manager", "name"]` (order may vary, but both labels are present).

### (4) Tests
❌ **NEEDS_CONFIRMATION**

**Status**: No test file found specifically for `running_managed_controllers` metric.

**Search Performed**:
- Searched for test files in `staging/src/k8s.io/component-base/metrics/prometheus/controllers/` directory
- No test files found in that directory
- Searched codebase-wide for tests referencing `running_managed_controllers`, `ControllerStarted`, `ControllerStopped`, or `controllerInstanceCount`
- Found usage in integration tests and controller code, but no unit tests that validate the metric itself

**Missing Evidence**:
- No test that calls `Register()` and verifies the metric is registered
- No test that validates `ControllerStarted()` emits the metric with correct labels and value (1)
- No test that validates `ControllerStopped()` emits the metric with correct labels and value (0)
- No test that validates the "name" and "manager" labels are present and correct
- No test that validates the metric value changes correctly when controllers start/stop

**Note**: While the metric is used in integration tests (e.g., controller manager tests), there are no unit tests that specifically validate the metric registration, emission, labels, and values in isolation.

---

## Summary

| Metric | Help Text | Registration/Emission | Labels | Tests | Status |
|--------|-----------|----------------------|--------|-------|--------|
| `kubernetes_build_info` | ✅ | ✅ | ✅ | ❌ | **NEEDS_CONFIRMATION** - Missing tests |
| `rest_client_request_duration_seconds` | ✅ | ✅ | ✅ | ❌ | **NEEDS_CONFIRMATION** - Missing tests |
| `rest_client_requests_total` | ✅ | ✅ | ✅ | ✅ | **READY** - All requirements met |
| `running_managed_controllers` | ✅ | ✅ | ✅ | ❌ | **NEEDS_CONFIRMATION** - Missing tests |

### Overall PR Status

**3 out of 4 metrics need tests** before they can graduate to Beta.

### Required Actions

1. **`kubernetes_build_info`**: Add test that validates:
   - Metric registration via `RegisterBuildInfo()`
   - Metric emission with all 9 label values
   - Metric value is set to 1
   - Label values match expected values from `version.Get()`

2. **`rest_client_request_duration_seconds`**: Add test case to `TestClientGOMetrics()` that validates:
   - Metric registration (already tested indirectly)
   - Metric emission via `latencyAdapter.Observe()`
   - "verb" and "host" labels are present and correct
   - Latency values are correctly recorded in histogram buckets

3. **`running_managed_controllers`**: Add test file (e.g., `metrics_test.go`) that validates:
   - Metric registration via `Register()`
   - `ControllerStarted()` emits metric with value 1 and correct labels
   - `ControllerStopped()` emits metric with value 0 and correct labels
   - "name" and "manager" labels are present and correct

4. **`rest_client_requests_total`**: ✅ **No action needed** - Test already exists and validates all requirements.
