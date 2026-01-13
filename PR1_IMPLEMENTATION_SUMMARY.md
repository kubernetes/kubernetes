# PR #1 Implementation Summary: Promote component-base Metrics from ALPHA to BETA

This document summarizes all changes made to promote the 4 component-base metrics from ALPHA to BETA stability level.

## Modified Files List

1. `staging/src/k8s.io/component-base/metrics/version.go` - Changed `kubernetes_build_info` stability from ALPHA to BETA
2. `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go` - Changed `rest_client_request_duration_seconds` and `rest_client_requests_total` stability from ALPHA to BETA
3. `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go` - Changed `running_managed_controllers` stability from ALPHA to BETA
4. `staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics_test.go` - Updated existing test and added new test for histogram metric
5. `staging/src/k8s.io/component-base/metrics/version_test.go` - **NEW FILE** - Added test for `kubernetes_build_info`
6. `staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics_test.go` - **NEW FILE** - Added test for `running_managed_controllers`

## Complete Diffs

### 1. staging/src/k8s.io/component-base/metrics/version.go

```diff
--- a/staging/src/k8s.io/component-base/metrics/version.go
+++ b/staging/src/k8s.io/component-base/metrics/version.go
@@ -23,7 +23,7 @@ var (
 	buildInfo = NewGaugeVec(
 		&GaugeOpts{
 			Name:           "kubernetes_build_info",
 			Help:           "A metric with a constant '1' value labeled by major, minor, git version, git commit, git tree state, build date, Go version, and compiler from which Kubernetes was built, and platform on which it is running.",
-			StabilityLevel: ALPHA,
+			StabilityLevel: BETA,
 		},
 		[]string{"major", "minor", "git_version", "git_commit", "git_tree_state", "build_date", "go_version", "compiler", "platform"},
 	)
```

### 2. staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go

```diff
--- a/staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go
+++ b/staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics.go
@@ -34,7 +34,7 @@ var (
 	requestLatency = k8smetrics.NewHistogramVec(
 		&k8smetrics.HistogramOpts{
 			Name:           "rest_client_request_duration_seconds",
 			Help:           "Request latency in seconds. Broken down by verb, and host.",
-			StabilityLevel: k8smetrics.ALPHA,
+			StabilityLevel: k8smetrics.BETA,
 			Buckets:        []float64{0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0},
 		},
 		[]string{"verb", "host"},
@@ -88,7 +88,7 @@ var (
 	requestResult = k8smetrics.NewCounterVec(
 		&k8smetrics.CounterOpts{
 			Name:           "rest_client_requests_total",
-			StabilityLevel: k8smetrics.ALPHA,
+			StabilityLevel: k8smetrics.BETA,
 			Help:           "Number of HTTP requests, partitioned by status code, method, and host.",
 		},
 		[]string{"code", "method", "host"},
```

### 3. staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go

```diff
--- a/staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go
+++ b/staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics.go
@@ -28,7 +28,7 @@ var (
 	controllerInstanceCount = k8smetrics.NewGaugeVec(
 		&k8smetrics.GaugeOpts{
 			Name:           "running_managed_controllers",
 			Help:           "Indicates where instances of a controller are currently running",
-			StabilityLevel: k8smetrics.ALPHA,
+			StabilityLevel: k8smetrics.BETA,
 		},
 		[]string{"name", "manager"},
 	)
```

### 4. staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics_test.go

```diff
--- a/staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics_test.go
+++ b/staging/src/k8s.io/component-base/metrics/prometheus/restclient/metrics_test.go
@@ -17,6 +17,15 @@ package restclient
 import (
 	"context"
+	"net/url"
 	"strings"
 	"testing"
+	"time"
 
 	"k8s.io/client-go/tools/metrics"
 	"k8s.io/component-base/metrics/legacyregistry"
 	"k8s.io/component-base/metrics/testutil"
 )
+
+func mustParseURL(rawURL string) url.URL {
+	u, err := url.Parse(rawURL)
+	if err != nil {
+		panic(err)
+	}
+	return *u
+}
 
 func TestClientGOMetrics(t *testing.T) {
@@ -44,7 +53,7 @@ func TestClientGOMetrics(t *testing.T) {
 			},
 			want: `
-			            # HELP rest_client_requests_total [ALPHA] Number of HTTP requests, partitioned by status code, method, and host.
+			            # HELP rest_client_requests_total [BETA] Number of HTTP requests, partitioned by status code, method, and host.
 			            # TYPE rest_client_requests_total counter
 			            rest_client_requests_total{code="200",host="www.foo.com",method="POST"} 1
 				`,
@@ -50,6 +59,35 @@ func TestClientGOMetrics(t *testing.T) {
 		},
 		{
+			description: "Request latency in seconds. Broken down by verb, and host.",
+			name:        "rest_client_request_duration_seconds",
+			metric:      requestLatency,
+			update: func() {
+				metrics.RequestLatency.Observe(context.TODO(), "GET", mustParseURL("https://www.example.com"), 100*time.Millisecond)
+			},
+			want: `
+			            # HELP rest_client_request_duration_seconds [BETA] Request latency in seconds. Broken down by verb, and host.
+			            # TYPE rest_client_request_duration_seconds histogram
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="0.005"} 0
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="0.025"} 0
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="0.1"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="0.25"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="0.5"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="1.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="2.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="4.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="8.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="15.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="30.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="60.0"} 1
+			            rest_client_request_duration_seconds_bucket{host="www.example.com",verb="GET",le="+Inf"} 1
+			            rest_client_request_duration_seconds_sum{host="www.example.com",verb="GET"} 0.1
+			            rest_client_request_duration_seconds_count{host="www.example.com",verb="GET"} 1
+				`,
+		},
+		{
 			description: "Number of request retries, partitioned by status code, verb, and host.",
```

### 5. staging/src/k8s.io/component-base/metrics/version_test.go (NEW FILE)

```go
/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegisterBuildInfo(t *testing.T) {
	registry := NewKubeRegistry()

	RegisterBuildInfo(registry)

	mfs, err := registry.Gather()
	require.NoError(t, err, "Gather failed")

	// Find the kubernetes_build_info metric
	var found bool
	for _, mf := range mfs {
		if *mf.Name == "kubernetes_build_info" {
			found = true

			// Verify help text includes [BETA]
			assert.Contains(t, mf.GetHelp(), "[BETA]", "Help text should contain [BETA]")
			assert.Contains(t, mf.GetHelp(), "A metric with a constant '1' value labeled by major, minor, git version", "Help text should describe the metric")

			// Verify metric type is gauge
			assert.Equal(t, "gauge", mf.GetType().String(), "Metric type should be gauge")

			// Verify we have exactly one metric
			metrics := mf.GetMetric()
			require.Len(t, metrics, 1, "Should have exactly one metric")

			metric := metrics[0]

			// Verify the value is 1
			assert.Equal(t, 1.0, metric.GetGauge().GetValue(), "Metric value should be 1")

			// Verify all expected labels are present
			expectedLabels := map[string]bool{
				"major":          false,
				"minor":          false,
				"git_version":    false,
				"git_commit":     false,
				"git_tree_state": false,
				"build_date":     false,
				"go_version":     false,
				"compiler":       false,
				"platform":       false,
			}

			labelMap := make(map[string]string)
			for _, label := range metric.GetLabel() {
				labelMap[*label.Name] = *label.Value
				if _, ok := expectedLabels[*label.Name]; ok {
					expectedLabels[*label.Name] = true
				}
			}

			// Verify all expected labels are present
			for labelName, found := range expectedLabels {
				assert.True(t, found, "Label %s should be present", labelName)
			}

			// Verify label values are non-empty (they come from version.Get() which is dynamic)
			for labelName, labelValue := range labelMap {
				if _, ok := expectedLabels[labelName]; ok {
					assert.NotEmpty(t, labelValue, "Label %s should have a non-empty value", labelName)
				}
			}
		}
	}

	assert.True(t, found, "kubernetes_build_info metric should be found")
}
```

### 6. staging/src/k8s.io/component-base/metrics/prometheus/controllers/metrics_test.go (NEW FILE)

```go
/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRunningManagedControllers(t *testing.T) {
	// Reset the metric to ensure clean state
	controllerInstanceCount.Reset()

	// Register the metric
	Register()

	// Create controller manager metrics instance
	managerName := "test-manager"
	metrics := NewControllerManagerMetrics(managerName)

	// Test ControllerStarted
	controllerName := "test-controller"
	metrics.ControllerStarted(controllerName)

	wantStarted := `
		# HELP running_managed_controllers [BETA] Indicates where instances of a controller are currently running
		# TYPE running_managed_controllers gauge
		running_managed_controllers{manager="test-manager",name="test-controller"} 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantStarted), "running_managed_controllers"); err != nil {
		t.Fatal(err)
	}

	// Test ControllerStopped
	metrics.ControllerStopped(controllerName)

	wantStopped := `
		# HELP running_managed_controllers [BETA] Indicates where instances of a controller are currently running
		# TYPE running_managed_controllers gauge
		running_managed_controllers{manager="test-manager",name="test-controller"} 0
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantStopped), "running_managed_controllers"); err != nil {
		t.Fatal(err)
	}
}
```

## Summary of Changes

### Stability Level Changes
- Changed 4 metrics from `ALPHA` to `BETA` stability level:
  1. `kubernetes_build_info`
  2. `rest_client_request_duration_seconds`
  3. `rest_client_requests_total`
  4. `running_managed_controllers`

### Test Additions/Updates
- **Updated** `restclient/metrics_test.go`:
  - Changed expected stability level from `[ALPHA]` to `[BETA]` for `rest_client_requests_total` test
  - Added new test case for `rest_client_request_duration_seconds` metric
  
- **Created** `version_test.go`:
  - Added comprehensive test for `kubernetes_build_info` metric
  - Validates registration, emission, labels, and values
  
- **Created** `controllers/metrics_test.go`:
  - Added test for `running_managed_controllers` metric
  - Validates `ControllerStarted()` and `ControllerStopped()` methods
  - Validates label values and metric values (1 for started, 0 for stopped)

### Test Coverage
All 4 metrics now have tests that validate:
- ✅ Metric registration
- ✅ Metric emission
- ✅ Expected labels
- ✅ Expected label values
- ✅ Expected metric values
- ✅ Help text includes `[BETA]` stability annotation

## Notes

- All changes follow existing test patterns in the codebase
- Tests use `testutil.GatherAndCompare` for validation, consistent with existing tests
- No changes were made to non-component-base metrics
- All linter errors have been resolved
