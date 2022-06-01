/*
Copyright 2021 The Kubernetes Authors.

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
	"sync"

	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const GarbageCollectorControllerSubsystem = "garbagecollector_controller"

var (
	GarbageCollectorResourcesSyncError = metrics.NewCounter(
		&metrics.CounterOpts{
			Subsystem:      GarbageCollectorControllerSubsystem,
			Name:           "resources_sync_error_total",
			Help:           "Number of garbage collector resources sync errors",
			StabilityLevel: metrics.ALPHA,
		})

	deleteQueueRetrySinceSecondsOpts = prometheus.HistogramOpts{
		Subsystem: GarbageCollectorControllerSubsystem,
		Name:      "attempt_to_delete_queue_retry_since_seconds",
		Help:      "How long in seconds an item has been retrying in attempt to delete workqueue.",
		Buckets:   metrics.ExponentialBuckets(0.001, 10, 10),
	}
	orphanQueueRetrySinceSecondsOpts = prometheus.HistogramOpts{
		Subsystem: GarbageCollectorControllerSubsystem,
		Name:      "attempt_to_orphan_queue_retry_since_seconds",
		Help:      "How long in seconds an item has been retrying in attempt to orphan workqueue.",
		Buckets:   metrics.ExponentialBuckets(0.001, 10, 10),
	}

	deleteQueueRetrySinceSecondsDesc = metrics.NewDesc(metrics.BuildFQName(deleteQueueRetrySinceSecondsOpts.Namespace, deleteQueueRetrySinceSecondsOpts.Subsystem, deleteQueueRetrySinceSecondsOpts.Name),
		deleteQueueRetrySinceSecondsOpts.Help,
		nil,
		nil,
		metrics.ALPHA,
		"")
	orphanQueueRetrySinceSecondsDesc = metrics.NewDesc(metrics.BuildFQName(orphanQueueRetrySinceSecondsOpts.Namespace, orphanQueueRetrySinceSecondsOpts.Subsystem, orphanQueueRetrySinceSecondsOpts.Name),
		orphanQueueRetrySinceSecondsOpts.Help,
		nil,
		nil,
		metrics.ALPHA,
		"")
)

var registerMetrics sync.Once

// Register registers GarbageCollectorController metrics.
func Register(attemptToDeleteMetrics QueueMetrics, attemptToOrphanMetrics QueueMetrics) {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(GarbageCollectorResourcesSyncError)
		legacyregistry.CustomMustRegister(newGCMetricsCollector(attemptToDeleteMetrics, attemptToOrphanMetrics))
	})
}

func newGCMetricsCollector(attemptToDeleteMetrics, attemptToOrphanMetrics QueueMetrics) metrics.StableCollector {
	return &gcMetricsCollector{
		attemptToDeleteMetrics: attemptToDeleteMetrics,
		attemptToOrphanMetrics: attemptToOrphanMetrics,
	}
}

type gcMetricsCollector struct {
	metrics.BaseStableCollector
	attemptToDeleteMetrics QueueMetrics
	attemptToOrphanMetrics QueueMetrics
}

func (g *gcMetricsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- deleteQueueRetrySinceSecondsDesc
	ch <- orphanQueueRetrySinceSecondsDesc
}
func (g *gcMetricsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	g.collectDeleteQueue(ch)
	g.collectOrphanQueue(ch)
}

func (g *gcMetricsCollector) collectDeleteQueue(ch chan<- metrics.Metric) {
	if deleteQueueRetrySinceSecondsDesc.IsHidden() {
		return
	}

	histogram := prometheus.NewHistogram(deleteQueueRetrySinceSecondsOpts)
	for _, startRetryTime := range g.attemptToDeleteMetrics.GetRetrySinceDurations() {
		histogram.Observe(startRetryTime.Seconds())
	}

	ch <- histogram
}

func (g *gcMetricsCollector) collectOrphanQueue(ch chan<- metrics.Metric) {
	if orphanQueueRetrySinceSecondsDesc.IsHidden() {
		return
	}

	histogram := prometheus.NewHistogram(orphanQueueRetrySinceSecondsOpts)
	for _, startRetryTime := range g.attemptToOrphanMetrics.GetRetrySinceDurations() {
		histogram.Observe(startRetryTime.Seconds())
	}

	ch <- histogram
}
