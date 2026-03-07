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
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/component-base/metrics/legacyregistry"
	metricstestutil "k8s.io/component-base/metrics/testutil"
)

func setupMetrics(t *testing.T) {
	t.Helper()
	legacyregistry.Reset()
	Register()
	legacyregistry.Reset()
	t.Cleanup(legacyregistry.Reset)
}

func TestRecordListCacheMetrics(t *testing.T) {
	setupMetrics(t)

	groupResource := schema.GroupResource{Group: "apps", Resource: "deployments"}
	RecordListCacheMetrics(groupResource, "byNamespace", 10, 5)

	expected := `
	# HELP apiserver_cache_list_total [ALPHA] Number of LIST requests served from watch cache
	# TYPE apiserver_cache_list_total counter
	apiserver_cache_list_total{group="apps",index="byNamespace",resource="deployments"} 1
	# HELP apiserver_cache_list_fetched_objects_total [ALPHA] Number of objects read from watch cache in the course of serving a LIST request
	# TYPE apiserver_cache_list_fetched_objects_total counter
	apiserver_cache_list_fetched_objects_total{group="apps",index="byNamespace",resource="deployments"} 10
	# HELP apiserver_cache_list_returned_objects_total [ALPHA] Number of objects returned for a LIST request from watch cache
	# TYPE apiserver_cache_list_returned_objects_total counter
	apiserver_cache_list_returned_objects_total{group="apps",resource="deployments"} 5
	`

	if err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer,
		strings.NewReader(expected),
		"apiserver_cache_list_total",
		"apiserver_cache_list_fetched_objects_total",
		"apiserver_cache_list_returned_objects_total",
	); err != nil {
		t.Fatalf("unexpected metrics output: %v", err)
	}
}

func TestRecordResourceVersion(t *testing.T) {
	setupMetrics(t)

	groupResource := schema.GroupResource{Group: "apps", Resource: "deployments"}
	RecordResourceVersion(groupResource, 12345)

	expected := `
	# HELP apiserver_watch_cache_resource_version [BETA] Current resource version of watch cache broken by resource type.
	# TYPE apiserver_watch_cache_resource_version gauge
	apiserver_watch_cache_resource_version{group="apps",resource="deployments"} 12345
	`

	if err := metricstestutil.GatherAndCompare(
		legacyregistry.DefaultGatherer,
		strings.NewReader(expected),
		"apiserver_watch_cache_resource_version",
	); err != nil {
		t.Fatalf("unexpected metrics output: %v", err)
	}
}

func TestRecordsWatchCacheCapacityChange(t *testing.T) {
	tests := []struct {
		name     string
		old      int
		new      int
		counter  string
		other    string
		expected string
	}{
		{
			name:    "increase",
			old:     10,
			new:     20,
			counter: "watch_cache_capacity_increase_total",
			other:   "watch_cache_capacity_decrease_total",
			expected: `
				# HELP watch_cache_capacity [BETA] Total capacity of watch cache broken by resource type.
				# TYPE watch_cache_capacity gauge
				watch_cache_capacity{group="apps",resource="deployments"} 20
				# HELP watch_cache_capacity_increase_total [BETA] Total number of watch cache capacity increase events broken by resource type.
				# TYPE watch_cache_capacity_increase_total counter
				watch_cache_capacity_increase_total{group="apps",resource="deployments"} 1
			`,
		},
		{
			name:    "decrease",
			old:     25,
			new:     15,
			counter: "watch_cache_capacity_decrease_total",
			other:   "watch_cache_capacity_increase_total",
			expected: `
				# HELP watch_cache_capacity [BETA] Total capacity of watch cache broken by resource type.
				# TYPE watch_cache_capacity gauge
				watch_cache_capacity{group="apps",resource="deployments"} 15
				# HELP watch_cache_capacity_decrease_total [BETA] Total number of watch cache capacity decrease events broken by resource type.
				# TYPE watch_cache_capacity_decrease_total counter
				watch_cache_capacity_decrease_total{group="apps",resource="deployments"} 1
			`,
		},
	}

	groupResource := schema.GroupResource{Group: "apps", Resource: "deployments"}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			setupMetrics(t)

			RecordsWatchCacheCapacityChange(groupResource, rt.old, rt.new)

			if err := metricstestutil.GatherAndCompare(
				legacyregistry.DefaultGatherer,
				strings.NewReader(rt.expected),
				"watch_cache_capacity",
				rt.counter,
			); err != nil {
				t.Fatalf("unexpected watch cache capacity metric: %v", err)
			}

			labels := map[string]string{"group": "apps", "resource": "deployments"}
			metricstestutil.AssertVectorCount(t, rt.counter, labels, 1)
			metricstestutil.AssertVectorCount(t, rt.other, labels, 0)
		})
	}
}

func TestListCacheCountMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	listCacheCount.WithLabelValues(gr.Group, gr.Resource, "byNamespace").Inc()
	metricstestutil.AssertVectorCount(t, "apiserver_cache_list_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
		"index":    "byNamespace",
	}, 1)
}

func TestListCacheNumFetchedMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	listCacheNumFetched.WithLabelValues(gr.Group, gr.Resource, "byNamespace").Add(3)
	metricstestutil.AssertVectorCount(t, "apiserver_cache_list_fetched_objects_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
		"index":    "byNamespace",
	}, 3)
}

func TestListCacheNumReturnedMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	listCacheNumReturned.WithLabelValues(gr.Group, gr.Resource).Add(4)
	metricstestutil.AssertVectorCount(t, "apiserver_cache_list_returned_objects_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 4)
}

func TestInitCounterMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	InitCounter.WithLabelValues(gr.Group, gr.Resource).Inc()
	metricstestutil.AssertVectorCount(t, "apiserver_init_events_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 1)
}

func TestEventsReceivedCounterMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	EventsReceivedCounter.WithLabelValues(gr.Group, gr.Resource).Add(2)
	metricstestutil.AssertVectorCount(t, "apiserver_watch_cache_events_received_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 2)
}

func TestEventsCounterMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	EventsCounter.WithLabelValues(gr.Group, gr.Resource).Add(5)
	metricstestutil.AssertVectorCount(t, "apiserver_watch_cache_events_dispatched_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 5)
}

func TestTerminatedWatchersCounterMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	TerminatedWatchersCounter.WithLabelValues(gr.Group, gr.Resource).Inc()
	metricstestutil.AssertVectorCount(t, "apiserver_terminated_watchers_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 1)
}

func TestWatchCacheResourceVersionMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	watchCacheResourceVersion.WithLabelValues(gr.Group, gr.Resource).Set(99)
	metricstestutil.AssertGaugeValue(t, "apiserver_watch_cache_resource_version", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 99)
}

func TestWatchCacheCapacityIncreaseMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	watchCacheCapacityIncreaseTotal.WithLabelValues(gr.Group, gr.Resource).Inc()
	metricstestutil.AssertVectorCount(t, "watch_cache_capacity_increase_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 1)
}

func TestWatchCacheCapacityDecreaseMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	watchCacheCapacityDecreaseTotal.WithLabelValues(gr.Group, gr.Resource).Inc()
	metricstestutil.AssertVectorCount(t, "watch_cache_capacity_decrease_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 1)
}

func TestWatchCacheCapacityGaugeMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	WatchCacheCapacity.WithLabelValues(gr.Group, gr.Resource).Set(50)
	metricstestutil.AssertGaugeValue(t, "watch_cache_capacity", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 50)
}

func TestWatchCacheInitializationsMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	WatchCacheInitializations.WithLabelValues(gr.Group, gr.Resource).Inc()
	metricstestutil.AssertVectorCount(t, "apiserver_watch_cache_initializations_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 1)
}

func TestWatchCacheReadWaitMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	WatchCacheReadWait.WithLabelValues(gr.Group, gr.Resource).Observe(0.2)
	metricstestutil.AssertHistogramTotalCount(t, "apiserver_watch_cache_read_wait_seconds", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
	}, 1)
}

func TestConsistentReadTotalMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	ConsistentReadTotal.WithLabelValues(gr.Group, gr.Resource, "true", "false").Add(7)
	metricstestutil.AssertVectorCount(t, "apiserver_watch_cache_consistent_read_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
		"success":  "true",
		"fallback": "false",
	}, 7)
}

func TestStorageConsistencyCheckTotalMetric(t *testing.T) {
	setupMetrics(t)
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	StorageConsistencyCheckTotal.WithLabelValues(gr.Group, gr.Resource, "success").Inc()
	metricstestutil.AssertVectorCount(t, "apiserver_storage_consistency_checks_total", map[string]string{
		"group":    gr.Group,
		"resource": gr.Resource,
		"status":   "success",
	}, 1)
}
