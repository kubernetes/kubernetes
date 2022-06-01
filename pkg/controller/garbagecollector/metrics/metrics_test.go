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

package metrics

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"strings"
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

func newFakeMetrics(collectionTime time.Time, config map[string]time.Duration) QueueMetrics {
	fakeClock := testingclock.NewFakeClock(collectionTime)
	result := queueMetrics{
		clock:           fakeClock,
		startRetryTimes: map[interface{}]time.Time{},
	}

	for itemName, duration := range config {
		item := testItem{name: itemName}

		fakeClock.SetTime(collectionTime.Add(-duration))
		result.AddRateLimited(item)
	}
	fakeClock.SetTime(collectionTime)

	return &result
}

func TestGCMetricsCollector(t *testing.T) {
	collectionTime := time.Now()

	attemptToDeleteMetrics := newFakeMetrics(collectionTime, map[string]time.Duration{
		"a": time.Millisecond,
		"b": 100 * time.Millisecond,
		"c": 2 * time.Second,
		"d": 2 * time.Second,
		"e": 7 * time.Second,
		"f": 3 * time.Hour,
	})
	attemptToOrphanMetrics := newFakeMetrics(collectionTime, map[string]time.Duration{
		"u": 100 * time.Second,
		"v": time.Hour,
	})

	expectedMetrics := `
		# HELP garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds How long in seconds an item has been retrying in attempt to delete workqueue.
        # TYPE garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds histogram
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="0.001"} 1
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="0.01"} 1
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="0.1"} 2
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="1"} 2
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="10"} 5
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="100"} 5
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="1000"} 5
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="10000"} 5
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="100000"} 6
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="1e+06"} 6
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_bucket{le="+Inf"} 6
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_sum 10811.101
        garbagecollector_controller_attempt_to_delete_queue_retry_since_seconds_count 6
        # HELP garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds How long in seconds an item has been retrying in attempt to orphan workqueue.
        # TYPE garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds histogram
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="0.001"} 0
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="0.01"} 0
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="0.1"} 0
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="1"} 0
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="10"} 0
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="100"} 1
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="1000"} 1
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="10000"} 2
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="100000"} 2
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="1e+06"} 2
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_bucket{le="+Inf"} 2
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_sum 3700
        garbagecollector_controller_attempt_to_orphan_queue_retry_since_seconds_count 2
	`

	collector := newGCMetricsCollector(attemptToDeleteMetrics, attemptToOrphanMetrics)
	registry := metrics.NewKubeRegistry()
	registry.CustomMustRegister(collector)
	err := testutil.GatherAndCompare(registry, strings.NewReader(expectedMetrics))
	if err != nil {
		t.Fatal(err)
	}
}
