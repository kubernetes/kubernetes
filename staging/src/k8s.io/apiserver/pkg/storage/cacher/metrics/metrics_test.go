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

package metrics

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestWatchCacheGaugesDeletedWithLastWatchCache(t *testing.T) {
	registry := compbasemetrics.NewKubeRegistry()
	registry.MustRegister(watchCacheResourceVersion, WatchCacheCapacity)
	widgets := schema.GroupResource{Group: "example.com", Resource: "widgets"}
	pods := schema.GroupResource{Resource: "pods"}

	// A custom resource with two versions has two watch caches.
	RecordWatchCacheStart(widgets)
	RecordWatchCacheStart(widgets)
	RecordWatchCacheStart(pods)
	RecordResourceVersion(widgets, 10)
	RecordsWatchCacheCapacityChange(widgets, 0, 100)
	RecordResourceVersion(pods, 20)

	bothGauges := `
# HELP apiserver_watch_cache_resource_version [ALPHA] Current resource version of watch cache broken by resource type. This is truncated to the 15 least significant digits.
# TYPE apiserver_watch_cache_resource_version gauge
apiserver_watch_cache_resource_version{group="",resource="pods"} 20
apiserver_watch_cache_resource_version{group="example.com",resource="widgets"} 10
# HELP watch_cache_capacity [ALPHA] Total capacity of watch cache broken by resource type.
# TYPE watch_cache_capacity gauge
watch_cache_capacity{group="example.com",resource="widgets"} 100
`
	podsGauges := `
# HELP apiserver_watch_cache_resource_version [ALPHA] Current resource version of watch cache broken by resource type. This is truncated to the 15 least significant digits.
# TYPE apiserver_watch_cache_resource_version gauge
apiserver_watch_cache_resource_version{group="",resource="pods"} 20
`
	metricNames := []string{"apiserver_watch_cache_resource_version", "watch_cache_capacity"}

	RecordWatchCacheStop(widgets)
	if err := testutil.GatherAndCompare(registry, strings.NewReader(bothGauges), metricNames...); err != nil {
		t.Errorf("gauges after stopping one of two watch caches: %v", err)
	}

	RecordWatchCacheStop(widgets)
	if err := testutil.GatherAndCompare(registry, strings.NewReader(podsGauges), metricNames...); err != nil {
		t.Errorf("gauges after stopping the last watch cache: %v", err)
	}
	if _, ok := runningWatchCaches[widgets]; ok {
		t.Errorf("resource type is still counted after its last watch cache stopped")
	}

	// The gauges of a recreated resource type are reported again.
	RecordWatchCacheStart(widgets)
	RecordResourceVersion(widgets, 10)
	RecordsWatchCacheCapacityChange(widgets, 0, 100)
	if err := testutil.GatherAndCompare(registry, strings.NewReader(bothGauges), metricNames...); err != nil {
		t.Errorf("gauges after restarting a watch cache: %v", err)
	}
}
