/*
Copyright 2024 The Kubernetes Authors.

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

package reflector

import (
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"strings"
	"testing"
)

func TestReflectorMetrics(t *testing.T) {
	tests := []struct {
		description string
		name        string
		metric      interface{}
		update      func(metricsProvider cache.MetricsProvider)
		want        string
	}{
		{
			description: "Total number of API lists done by the reflectors.",
			name:        "reflector_lists_total",
			metric:      listsTotal,
			update: func(metricsProvider cache.MetricsProvider) {
				metricsProvider.NewListsMetric("test_reflector").Inc()
			},
			want: `
			            # HELP reflector_lists_total [ALPHA] Total number of API lists done by the reflectors
			            # TYPE reflector_lists_total counter
			            reflector_lists_total{name="test_reflector"} 1
				`,
		},
		{
			description: "How long an API list takes to return and decode for the reflectors.",
			name:        "reflector_list_duration_seconds",
			metric:      listsDuration,
			update: func(metricsProvider cache.MetricsProvider) {
				metricsProvider.NewListDurationMetric("test_reflector").Observe(10)
			},
			want: `
			            # HELP reflector_list_duration_seconds [ALPHA] How long an API list takes to return and decode for the reflectors
			            # TYPE reflector_list_duration_seconds histogram
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.005"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.01"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.025"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.05"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.1"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.25"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="0.5"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="1"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="2.5"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="5"} 0
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="10"} 1
			            reflector_list_duration_seconds_bucket{name="test_reflector",le="+Inf"} 1
			            reflector_list_duration_seconds_sum{name="test_reflector"} 10
			            reflector_list_duration_seconds_count{name="test_reflector"} 1
				`,
		},
		{
			description: "Last resource version seen for the reflectors.",
			name:        "reflector_last_resource_version",
			metric:      lastResourceVersion,
			update: func(metricsProvider cache.MetricsProvider) {
				metricsProvider.NewLastResourceVersionMetric("test_reflector").Set(1234)
			},
			want: `
			            # HELP reflector_last_resource_version [ALPHA] Last resource version seen for the reflectors
			            # TYPE reflector_last_resource_version gauge
			            reflector_last_resource_version{name="test_reflector"} 1234
				`,
		},
	}

	metricsProvider := reflectorMetricsProvider{}

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
			test.update(metricsProvider)

			// ensure metric exposed
			exposed := false
			metrics, _ := legacyregistry.DefaultGatherer.Gather()
			for _, mf := range metrics {
				if mf.GetName() == test.name {
					exposed = true
					break
				}
			}
			if !exposed {
				t.Fatalf("the metric must be exposed: %s", test.name)
			}

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), test.name); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestReflectorMetricsDeletion(t *testing.T) {
	tests := []struct {
		description string
		name        string
		metric      interface{}
		update      func(metricsProvider cache.MetricsProvider)
		delete      func(metricsProvider cache.MetricsProvider)
	}{
		{
			description: "Total number of API lists done by the reflectors.",
			name:        "reflector_lists_total",
			metric:      listsTotal,
			update: func(metricsProvider cache.MetricsProvider) {
				metricsProvider.NewListsMetric("test_reflector").Inc()
			},
			delete: func(metricsProvider cache.MetricsProvider) {
				metricsProvider.DeleteListsMetric("test_reflector")
			},
		},
	}

	metricsProvider := reflectorMetricsProvider{}

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
			test.update(metricsProvider)

			// ensure metric exposed
			exposed := false
			metrics, _ := legacyregistry.DefaultGatherer.Gather()
			for _, mf := range metrics {
				if mf.GetName() == test.name {
					exposed = true
					break
				}
			}
			if !exposed {
				t.Fatalf("the metric must be exposed: %s", test.name)
			}

			test.delete(metricsProvider)
			// ensure metric is deleted
			deleted := true
			metrics, _ = legacyregistry.DefaultGatherer.Gather()
			for _, mf := range metrics {
				if mf.GetName() == test.name {
					deleted = false
					break
				}
			}
			if !deleted {
				t.Fatalf("the metric must be deleted: %s", test.name)
			}

			test.update(metricsProvider)
			exposed = false
			metrics, _ = legacyregistry.DefaultGatherer.Gather()
			for _, mf := range metrics {
				if mf.GetName() == test.name {
					exposed = true
					break
				}
			}
			if !exposed {
				t.Fatalf("the metric must be exposed: %s", test.name)
			}
		})
	}
}
