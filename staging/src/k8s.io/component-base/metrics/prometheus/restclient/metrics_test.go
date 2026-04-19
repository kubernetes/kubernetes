/*
Copyright 2022 The Kubernetes Authors.

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

package restclient

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/metrics"
	cbmetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
)

func TestClientGOMetrics(t *testing.T) {
	tests := []struct {
		description string
		name        string
		metric      interface{}
		update      func()
		want        string
	}{
		{
			description: "Number of HTTP requests, partitioned by status code, verb, and host.",
			name:        "rest_client_requests_total",
			metric:      requestResult,
			update: func() {
				metrics.RequestResult.Increment(context.TODO(), "200", "POST", "www.foo.com")
			},
			want: `
			            # HELP rest_client_requests_total [BETA] Number of HTTP requests, partitioned by status code, method, and host.
			            # TYPE rest_client_requests_total counter
			            rest_client_requests_total{code="200",host="www.foo.com",method="POST"} 1
				`,
		},
		{
			description: "Number of request retries, partitioned by status code, verb, and host.",
			name:        "rest_client_request_retries_total",
			metric:      requestRetry,
			update: func() {
				metrics.RequestRetry.IncrementRetry(context.TODO(), "500", "GET", "www.bar.com")
			},
			want: `
			            # HELP rest_client_request_retries_total [ALPHA] Number of request retries, partitioned by status code, verb, and host.
			            # TYPE rest_client_request_retries_total counter
			            rest_client_request_retries_total{code="500",host="www.bar.com",verb="GET"} 1
				`,
		},
		{
			description: "Number of calls to an exec plugin",
			name:        "rest_client_exec_plugin_call_total",
			metric:      execPluginCalls,
			update: func() {
				metrics.ExecPluginCalls.Increment(0, "no_error")
			},
			want: `
						# HELP rest_client_exec_plugin_call_total [ALPHA] Number of calls to an exec plugin, partitioned by the type of event encountered (no_error, plugin_execution_error, plugin_not_found_error, client_internal_error) and an optional exit code. The exit code will be set to 0 if and only if the plugin call was successful.
        				# TYPE rest_client_exec_plugin_call_total counter
        				rest_client_exec_plugin_call_total{call_status="no_error",code="0"} 1
				`,
		},
		{
			description: "Number of calls to get a new transport",
			name:        "rest_client_transport_create_calls_total",
			metric:      transportCacheCalls,
			update: func() {
				metrics.TransportCreateCalls.Increment("hit")
			},
			want: `
			            # HELP rest_client_transport_create_calls_total [ALPHA] Number of calls to get a new transport, partitioned by the result of the operation hit: obtained from the cache, miss: created and added to the cache, miss-gc: recreated and added back to the cache after being garbage collected, uncacheable: created and not cached
			            # TYPE rest_client_transport_create_calls_total counter
			            rest_client_transport_create_calls_total{result="hit"} 1
				`,
		},
	}

	// no need to register the metrics here, since the init function of
	// the package registers all the client-go metrics.
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			resetter, resettable := test.metric.(interface {
				Reset()
			})
			if !resettable {
				t.Fatalf("the metric must be resettaable: %s", test.name)
			}

			// Since prometheus' gatherer is global, other tests may have updated
			// metrics already, so we need to reset them prior to running this test.
			// This also implies that we can't run this test in parallel with other tests.
			resetter.Reset()
			test.update()

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), test.name); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRequestLatencyMetric(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	registry := cbmetrics.NewKubeRegistry()
	defer registry.Reset()
	registry.MustRegister(requestLatency)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg := &rest.Config{
		Host: srv.URL,
		ContentConfig: rest.ContentConfig{
			NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
			GroupVersion:         &schema.GroupVersion{Version: "v1"},
		},
	}

	client, err := rest.RESTClientFor(cfg)
	if err != nil {
		t.Fatalf("RESTClientFor: %v", err)
	}

	if _, err := client.Get().AbsPath("/api").DoRaw(ctx); err != nil {
		t.Fatalf("request: %v", err)
	}

	srvURL, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse srv.URL: %v", err)
	}

	// The real request above drives metrics.RequestLatency.Observe via the
	// registered latencyAdapter. Strip timing-dependent histogram fields so the
	// comparison only verifies _count (bucket/sum depend on real wall-clock latency).
	want := fmt.Sprintf(`# HELP rest_client_request_duration_seconds [BETA] Request latency in seconds. Broken down by verb, and host.
# TYPE rest_client_request_duration_seconds histogram
rest_client_request_duration_seconds_count{host=%q,verb="GET"} 1
`, srvURL.Host)

	if err := testutil.GatherAndCompare(gatherWithoutDurations(registry), strings.NewReader(want), "rest_client_request_duration_seconds"); err != nil {
		t.Fatal(err)
	}
}

// gatherWithoutDurations wraps a gatherer and strips timing-dependent fields
// (SampleSum and Bucket) from histograms so tests only verify _count.
func gatherWithoutDurations(registry cbmetrics.KubeRegistry) testutil.GathererFunc {
	return func() ([]*testutil.MetricFamily, error) {
		got, err := registry.Gather()
		for _, mf := range got {
			for _, m := range mf.Metric {
				if m.Histogram == nil {
					continue
				}
				m.Histogram.SampleSum = nil
				m.Histogram.Bucket = nil
			}
		}
		return got, err
	}
}

func TestTransportCAReloadsMetric(t *testing.T) {
	tests := []struct {
		description string
		name        string
		metric      interface{}
		update      func()
		want        string
	}{
		{
			description: "Reload success, reason: unchanged",
			name:        "rest_client_transport_ca_reload_total",
			metric:      transportCAReloads,
			update: func() {
				metrics.TransportCAReloads.Increment("success", "unchanged")
			},
			want: `
			            # HELP rest_client_transport_ca_reload_total [ALPHA] Number of times a CA reload is attempted, partitioned by the result and reason for the reload attempt
			            # TYPE rest_client_transport_ca_reload_total counter
			            rest_client_transport_ca_reload_total{reason="unchanged", result="success"} 1
				`,
		},
		{
			description: "Reload success, reason: updated",
			name:        "rest_client_transport_ca_reload_total",
			metric:      transportCAReloads,
			update: func() {
				metrics.TransportCAReloads.Increment("success", "updated")
			},
			want: `
			            # HELP rest_client_transport_ca_reload_total [ALPHA] Number of times a CA reload is attempted, partitioned by the result and reason for the reload attempt
			            # TYPE rest_client_transport_ca_reload_total counter
			            rest_client_transport_ca_reload_total{reason="updated", result="success"} 1
				`,
		},
		{
			description: "Reload failure, reason: empty",
			name:        "rest_client_transport_ca_reload_total",
			metric:      transportCAReloads,
			update: func() {
				metrics.TransportCAReloads.Increment("failure", "empty")
			},
			want: `
			            # HELP rest_client_transport_ca_reload_total [ALPHA] Number of times a CA reload is attempted, partitioned by the result and reason for the reload attempt
			            # TYPE rest_client_transport_ca_reload_total counter
			            rest_client_transport_ca_reload_total{reason="empty", result="failure"} 1
				`,
		},
		{
			description: "Reload failure, reason: read_error",
			name:        "rest_client_transport_ca_reload_total",
			metric:      transportCAReloads,
			update: func() {
				metrics.TransportCAReloads.Increment("failure", "read_error")
			},
			want: `
			            # HELP rest_client_transport_ca_reload_total [ALPHA] Number of times a CA reload is attempted, partitioned by the result and reason for the reload attempt
			            # TYPE rest_client_transport_ca_reload_total counter
			            rest_client_transport_ca_reload_total{reason="read_error", result="failure"} 1
				`,
		},
		{
			description: "Reload failure, reason: ca_parse_error",
			name:        "rest_client_transport_ca_reload_total",
			metric:      transportCAReloads,
			update: func() {
				metrics.TransportCAReloads.Increment("failure", "ca_parse_error")
			},
			want: `
			            # HELP rest_client_transport_ca_reload_total [ALPHA] Number of times a CA reload is attempted, partitioned by the result and reason for the reload attempt
			            # TYPE rest_client_transport_ca_reload_total counter
			            rest_client_transport_ca_reload_total{reason="ca_parse_error", result="failure"} 1
				`,
		},
	}
	// no need to register the metrics here, since the init function of
	// the package registers all the client-go metrics.
	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			resetter, resettable := test.metric.(interface {
				Reset()
			})
			if !resettable {
				t.Fatalf("the metric must be resettaable: %s", test.name)
			}

			// Since prometheus' gatherer is global, other tests may have updated
			// metrics already, so we need to reset them prior to running this test.
			// This also implies that we can't run this test in parallel with other tests.
			resetter.Reset()
			test.update()

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), test.name); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTransportCleanupMetrics(t *testing.T) {
	t.Run("cleanup cancel calls", func(t *testing.T) {
		transportCertRotationGCCalls.Reset()
		metrics.TransportCertRotationGCCalls.Increment()
		metrics.TransportCertRotationGCCalls.Increment()

		want := `
			# HELP rest_client_transport_cert_rotation_gc_calls_total [ALPHA] Number of times a cert rotation goroutine cancel func is called via GC cleanup of the associated transport
			# TYPE rest_client_transport_cert_rotation_gc_calls_total counter
			rest_client_transport_cert_rotation_gc_calls_total 2
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "rest_client_transport_cert_rotation_gc_calls_total"); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("cleanup delete calls: deleted", func(t *testing.T) {
		transportCacheGCCalls.Reset()
		metrics.TransportCacheGCCalls.Increment("deleted")

		want := `
			# HELP rest_client_transport_cache_gc_calls_total [ALPHA] Number of times a GC cleanup attempts to delete a transport cache entry, partitioned by the result: deleted, skipped
			# TYPE rest_client_transport_cache_gc_calls_total counter
			rest_client_transport_cache_gc_calls_total{result="deleted"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "rest_client_transport_cache_gc_calls_total"); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("cleanup delete calls: skipped", func(t *testing.T) {
		transportCacheGCCalls.Reset()
		metrics.TransportCacheGCCalls.Increment("skipped")

		want := `
			# HELP rest_client_transport_cache_gc_calls_total [ALPHA] Number of times a GC cleanup attempts to delete a transport cache entry, partitioned by the result: deleted, skipped
			# TYPE rest_client_transport_cache_gc_calls_total counter
			rest_client_transport_cache_gc_calls_total{result="skipped"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "rest_client_transport_cache_gc_calls_total"); err != nil {
			t.Fatal(err)
		}
	})
}
