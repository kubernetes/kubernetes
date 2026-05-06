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

func TestRecordListCacheMetrics(t *testing.T) {
	metricstestutil.SetupMetrics(t, Register)

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
	metricstestutil.SetupMetrics(t, Register)

	groupResource := schema.GroupResource{Group: "apps", Resource: "deployments"}
	RecordResourceVersion(groupResource, 12345)

	expected := `
	# HELP apiserver_watch_cache_resource_version [BETA] Current resource version of watch cache broken by resource type. This is truncated to the 15 least significant digits.
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
			metricstestutil.SetupMetrics(t, Register)

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
