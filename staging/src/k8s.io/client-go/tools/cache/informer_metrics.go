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

// Package cache is a client-side caching mechanism. It is useful for
// reducing the number of server calls you'd otherwise need to make.
// Reflector watches a server and updates a Store. Two stores are provided;
// one that simply caches objects (for example, to allow a scheduler to
// list currently available nodes), and one that additionally acts as
// a FIFO queue (for example, to allow a scheduler to process incoming
// pods).
package cache

import "sync"

var (
	globalInformerMetricsProvider  InformerMetricsProvider = noopInformerMetricsProvider{}
	setInformerMetricsProviderOnce sync.Once
)

type noopInformerMetricsProvider struct{}

// BatchTerminationReason describes why RealFIFO.PopBatch stopped accumulating
// deltas into the current batch.
//
// A batch size alone does not say why batches are smaller than the configured
// limit. The reason is only known at the moment the batch is cut, so it is
// recorded there.
type BatchTerminationReason int

const (
	// BatchFull means the batch reached the configured batch size.
	BatchFull BatchTerminationReason = iota
	// BatchQueueDrained means the queue ran out of items before the batch
	// filled, i.e. the consumer is keeping up with the producer.
	BatchQueueDrained
	// BatchUnbatchableDelta means a Bookmark, ReplacedAll or SyncAll delta was
	// encountered. These are always processed alone.
	BatchUnbatchableDelta
	// BatchInitialPopulationBoundary means the batch was cut at the end of the
	// initial list, so that HasSynced does not flip mid-batch.
	BatchInitialPopulationBoundary

	// numBatchTerminationReasons must remain last.
	numBatchTerminationReasons
)

var batchTerminationReasonNames = [numBatchTerminationReasons]string{
	BatchFull:                      "batch_full",
	BatchQueueDrained:              "queue_drained",
	BatchUnbatchableDelta:          "unbatchable_delta",
	BatchInitialPopulationBoundary: "initial_population_boundary",
}

// String returns the metric label value for the reason.
func (r BatchTerminationReason) String() string {
	if r < 0 || r >= numBatchTerminationReasons {
		return "unknown"
	}
	return batchTerminationReasonNames[r]
}

// AllBatchTerminationReasons returns every BatchTerminationReason.
func AllBatchTerminationReasons() []BatchTerminationReason {
	reasons := make([]BatchTerminationReason, 0, numBatchTerminationReasons)
	for r := BatchTerminationReason(0); r < numBatchTerminationReasons; r++ {
		reasons = append(reasons, r)
	}
	return reasons
}

// InformerMetricsProvider defines an interface for creating metrics that track informer operations.
type InformerMetricsProvider interface {
	// NewQueuedItemMetric returns a gauge metric for tracking the total number of items
	// currently queued and waiting to be processed.
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	//
	// For DeltaFIFO: Represents len(f.items) - the number of unique keys with pending deltas
	// For RealFIFO: Represents len(f.items) - the total number of individual delta events queued
	NewQueuedItemMetric(id InformerNameAndResource) GaugeMetric

	// NewProcessingLatencyMetric returns a histogram metric for tracking the time taken
	// to process events (execute handlers) after they are popped from the queue.
	// The latency is measured in seconds.
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	NewProcessingLatencyMetric(id InformerNameAndResource) HistogramMetric

	// NewStoreResourceVersionMetric returns a gauge metric for tracking the resource version of the store.
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	NewStoreResourceVersionMetric(id InformerNameAndResource) GaugeMetric

	// NewBatchSizeMetric returns a histogram metric for tracking the number of
	// deltas popped together by RealFIFO.PopBatch.
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	NewBatchSizeMetric(id InformerNameAndResource) HistogramMetric

	// NewBatchTerminationMetric returns a counter metric for tracking how many
	// batches RealFIFO.PopBatch closed for the given reason.
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	NewBatchTerminationMetric(id InformerNameAndResource, reason BatchTerminationReason) CounterMetric

	// NewEventsEnqueuedMetric returns a counter metric for tracking the number of
	// deltas added to a RealFIFO. Paired with the batch metrics, it separates
	// "few events are arriving" from "events arrive but batches stay small".
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	NewEventsEnqueuedMetric(id InformerNameAndResource) CounterMetric
}

// fifoMetrics holds all metrics for a FIFO.
type fifoMetrics struct {
	numberOfQueuedItem GaugeMetric
	processingLatency  HistogramMetric
	batchSize          HistogramMetric
	batchTerminations  [numBatchTerminationReasons]CounterMetric
	eventsEnqueued     CounterMetric
}

// storeMetrics holds all metrics for a store.
type storeMetrics struct {
	storeResourceVersion GaugeMetric
}

// SetInformerMetricsProvider sets the metrics provider for all subsequently created
// informers. Only the first call has an effect.
func SetInformerMetricsProvider(metricsProvider InformerMetricsProvider) {
	setInformerMetricsProviderOnce.Do(func() {
		globalInformerMetricsProvider = metricsProvider
	})
}

func newFIFOMetrics(id InformerNameAndResource, metricsProvider InformerMetricsProvider) *fifoMetrics {
	if metricsProvider == nil {
		metricsProvider = globalInformerMetricsProvider
	}
	metrics := &fifoMetrics{
		numberOfQueuedItem: noopMetric{},
		processingLatency:  noopMetric{},
		batchSize:          noopMetric{},
		eventsEnqueued:     noopMetric{},
	}
	for _, reason := range AllBatchTerminationReasons() {
		metrics.batchTerminations[reason] = noopMetric{}
	}

	if id.Reserved() {
		metrics.numberOfQueuedItem = metricsProvider.NewQueuedItemMetric(id)
		metrics.processingLatency = metricsProvider.NewProcessingLatencyMetric(id)
		metrics.batchSize = metricsProvider.NewBatchSizeMetric(id)
		metrics.eventsEnqueued = metricsProvider.NewEventsEnqueuedMetric(id)
		for _, reason := range AllBatchTerminationReasons() {
			metrics.batchTerminations[reason] = metricsProvider.NewBatchTerminationMetric(id, reason)
		}
	}

	return metrics
}

func newStoreMetrics(id InformerNameAndResource, metricsProvider InformerMetricsProvider) *storeMetrics {
	if metricsProvider == nil {
		metricsProvider = globalInformerMetricsProvider
	}
	metrics := &storeMetrics{
		storeResourceVersion: noopMetric{},
	}

	if id.Reserved() {
		metrics.storeResourceVersion = &lazyStoreResourceVersionMetric{
			id:              id,
			metricsProvider: metricsProvider,
		}
	}

	return metrics
}

// lazyStoreResourceVersionMetric avoids publishing a 0 series before initial sync.
type lazyStoreResourceVersionMetric struct {
	id              InformerNameAndResource
	metricsProvider InformerMetricsProvider

	once   sync.Once
	metric GaugeMetric
}

func (m *lazyStoreResourceVersionMetric) Set(value float64) {
	m.once.Do(func() {
		m.metric = m.metricsProvider.NewStoreResourceVersionMetric(m.id)
	})
	m.metric.Set(value)
}

func (noopInformerMetricsProvider) NewQueuedItemMetric(InformerNameAndResource) GaugeMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewProcessingLatencyMetric(InformerNameAndResource) HistogramMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewStoreResourceVersionMetric(InformerNameAndResource) GaugeMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewBatchSizeMetric(InformerNameAndResource) HistogramMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewBatchTerminationMetric(InformerNameAndResource, BatchTerminationReason) CounterMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewEventsEnqueuedMetric(InformerNameAndResource) CounterMetric {
	return noopMetric{}
}
