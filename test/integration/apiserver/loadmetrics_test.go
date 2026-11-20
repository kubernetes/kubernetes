/*
Copyright The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/server/loadmetrics"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestEndpointLoadMetrics(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skipf("endpoint load metrics CPU measurement is not supported on %s", runtime.GOOS)
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RequestCostAggregation, true)

	server := kubeapiservertesting.StartTestServerOrDie(
		t,
		nil,
		[]string{"--enable-endpoint-load-metrics=true"},
		framework.SharedEtcd(),
	)
	defer server.TearDownFn()

	httpClient, err := restclient.HTTPClientFor(server.ClientConfig)
	if err != nil {
		t.Fatalf("failed to create HTTP client: %v", err)
	}

	var latestHeader string
	var parsedMetrics map[string]float64
	err = wait.PollUntilContextTimeout(context.Background(), 200*time.Millisecond, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		header, status, reqErr := sendLoadMetricsRequest(ctx, httpClient, server.ClientConfig.Host+"/livez", loadmetrics.FormatText)
		if reqErr != nil {
			return false, reqErr
		}
		if status != http.StatusOK {
			return false, fmt.Errorf("unexpected status code %d", status)
		}
		if header == "" {
			return false, nil
		}
		metrics, parseErr := parseORCATextReport(header)
		if parseErr != nil {
			return false, parseErr
		}
		if metrics["rps_fractional"] <= 0 {
			return false, nil
		}
		latestHeader = header
		parsedMetrics = metrics
		return true, nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for endpoint-load-metrics header with positive RPS: %v", err)
	}
	t.Logf("Received ORCA load report header: %q (parsed: %+v)", latestHeader, parsedMetrics)

	headerWithoutOptIn, status, err := sendLoadMetricsRequest(context.Background(), httpClient, server.ClientConfig.Host+"/livez", "")
	if err != nil || status != http.StatusOK {
		t.Fatalf("request without opt-in failed: status=%d, err=%v", status, err)
	}
	if headerWithoutOptIn != "" {
		t.Errorf("expected no %s header when request header is omitted, got %q", loadmetrics.ResponseHeader, headerWithoutOptIn)
	}

	headerWithUnsupportedFormat, status, err := sendLoadMetricsRequest(context.Background(), httpClient, server.ClientConfig.Host+"/livez", "JSON")
	if err != nil || status != http.StatusOK {
		t.Fatalf("request with unsupported format failed: status=%d, err=%v", status, err)
	}
	if headerWithUnsupportedFormat != "" {
		t.Errorf("expected no %s header for unsupported format, got %q", loadmetrics.ResponseHeader, headerWithUnsupportedFormat)
	}
}

func TestEndpointLoadMetrics_Disabled(t *testing.T) {
	testCases := []struct {
		name               string
		featureGateEnabled bool
		customFlags        []string
	}{
		{
			name:               "feature gate enabled but flag disabled by default",
			featureGateEnabled: true,
			customFlags:        nil,
		},
		{
			name:               "flag enabled but feature gate disabled",
			featureGateEnabled: false,
			customFlags:        []string{"--enable-endpoint-load-metrics=true"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RequestCostAggregation, tc.featureGateEnabled)

			server := kubeapiservertesting.StartTestServerOrDie(
				t,
				nil,
				tc.customFlags,
				framework.SharedEtcd(),
			)
			defer server.TearDownFn()

			httpClient, err := restclient.HTTPClientFor(server.ClientConfig)
			if err != nil {
				t.Fatalf("failed to create HTTP client: %v", err)
			}

			header, status, err := sendLoadMetricsRequest(context.Background(), httpClient, server.ClientConfig.Host+"/livez", loadmetrics.FormatText)
			if err != nil || status != http.StatusOK {
				t.Fatalf("request failed: status=%d, err=%v", status, err)
			}
			if header != "" {
				t.Errorf("expected no %s header when disabled, got %q", loadmetrics.ResponseHeader, header)
			}
		})
	}
}

func sendLoadMetricsRequest(ctx context.Context, client *http.Client, url string, requestedFormat string) (string, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", 0, err
	}
	if requestedFormat != "" {
		req.Header.Set(loadmetrics.RequestHeader, requestedFormat)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	return resp.Header.Get(loadmetrics.ResponseHeader), resp.StatusCode, nil
}

func parseORCATextReport(report string) (map[string]float64, error) {
	if !strings.HasPrefix(report, "TEXT ") {
		return nil, fmt.Errorf("expected ORCA report to start with %q, got %q", "TEXT ", report)
	}
	metricsPayload := strings.TrimPrefix(report, "TEXT ")
	parsed := make(map[string]float64)
	for entry := range strings.SplitSeq(metricsPayload, ",") {
		parts := strings.SplitN(strings.TrimSpace(entry), "=", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("malformed key=value pair %q in report %q", entry, report)
		}
		val, err := strconv.ParseFloat(parts[1], 64)
		if err != nil || math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			return nil, fmt.Errorf("invalid metric value for %q: %q (err=%w)", parts[0], parts[1], err)
		}
		parsed[parts[0]] = val
	}
	for _, requiredKey := range []string{"cpu_utilization", "rps_fractional"} {
		if _, ok := parsed[requiredKey]; !ok {
			return nil, fmt.Errorf("missing required ORCA key %q in report %q", requiredKey, report)
		}
	}
	return parsed, nil
}
