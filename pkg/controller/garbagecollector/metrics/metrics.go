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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
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

	deleteQueueRetrySinceSecondsOpts = &metrics.HistogramOpts{
		Subsystem:      GarbageCollectorControllerSubsystem,
		Name:           "attempt_to_delete_queue_retry_since_seconds",
		Help:           "How long in seconds an item has been retrying in attempt to delete workqueue.",
		Buckets:        metrics.ExponentialBuckets(0.001, 10, 10),
		StabilityLevel: metrics.ALPHA,
	}
	orphanQueueRetrySinceSecondsOpts = &metrics.HistogramOpts{
		Subsystem:      GarbageCollectorControllerSubsystem,
		Name:           "attempt_to_orphan_queue_retry_since_seconds",
		Help:           "How long in seconds an item has been retrying in attempt to orphan workqueue.",
		Buckets:        metrics.ExponentialBuckets(0.001, 10, 10),
		StabilityLevel: metrics.ALPHA,
	}

	deleteQueueRetrySinceSecondsDesc = metrics.NewDesc(metrics.BuildFQName(deleteQueueRetrySinceSecondsOpts.Namespace, deleteQueueRetrySinceSecondsOpts.Subsystem, deleteQueueRetrySinceSecondsOpts.Name),
		deleteQueueRetrySinceSecondsOpts.Help,
		nil,
		deleteQueueRetrySinceSecondsOpts.ConstLabels,
		deleteQueueRetrySinceSecondsOpts.StabilityLevel,
		deleteQueueRetrySinceSecondsOpts.DeprecatedVersion)
	orphanQueueRetrySinceSecondsDesc = metrics.NewDesc(metrics.BuildFQName(orphanQueueRetrySinceSecondsOpts.Namespace, orphanQueueRetrySinceSecondsOpts.Subsystem, orphanQueueRetrySinceSecondsOpts.Name),
		orphanQueueRetrySinceSecondsOpts.Help,
		nil,
		orphanQueueRetrySinceSecondsOpts.ConstLabels,
		orphanQueueRetrySinceSecondsOpts.StabilityLevel,
		orphanQueueRetrySinceSecondsOpts.DeprecatedVersion)
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
	g.collect(ch, deleteQueueRetrySinceSecondsDesc, deleteQueueRetrySinceSecondsOpts, g.attemptToDeleteMetrics)
	g.collect(ch, orphanQueueRetrySinceSecondsDesc, orphanQueueRetrySinceSecondsOpts, g.attemptToOrphanMetrics)
}

func (g *gcMetricsCollector) collect(ch chan<- metrics.Metric, desc *metrics.Desc, histogramOpts *metrics.HistogramOpts, qMetrics QueueMetrics) {
	if !desc.IsCreated() || desc.IsHidden() {
		return
	}
	// use this just as a wrapper of prometheus.Histogram
	histogram := metrics.NewHistogram(histogramOpts)
	// no need to pass version or check result as this is handled by desc
	histogram.Create(nil)

	for _, startRetryTime := range qMetrics.GetRetrySinceDurations() {
		histogram.Observe(startRetryTime.Seconds())
	}

	if h, ok := histogram.ObserverMetric.(metrics.Metric); ok {
		ch <- h
	} else {
		klog.Warningf("%v metric is not collecting properly\n", histogramOpts.Name)
	}
}
