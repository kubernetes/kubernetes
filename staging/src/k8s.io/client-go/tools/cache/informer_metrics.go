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

	// NewListDurationMetric returns a histogram metric for tracking the time taken by
	// the reflector to fetch the initial state (via LIST or WatchList) and replace
	// the contents of its store. Only successful attempts are observed.
	// The duration is measured in seconds.
	// verb is ListVerbList or ListVerbWatch. WatchList uses the watch API verb.
	// When WatchList falls back to LIST, the observation uses ListVerbList and
	// includes the failed WatchList attempt.
	// The returned metric should check id.Reserved() before updating to support
	// dynamic informers that may shut down while the process is still running.
	NewListDurationMetric(id InformerNameAndResource, verb string) HistogramMetric
}

// Values for the verb passed to InformerMetricsProvider.NewListDurationMetric.
const (
	ListVerbList  = "list"
	ListVerbWatch = "watch"
)

// fifoMetrics holds all metrics for a FIFO.
type fifoMetrics struct {
	numberOfQueuedItem GaugeMetric
	processingLatency  HistogramMetric
}

// storeMetrics holds all metrics for a store.
type storeMetrics struct {
	storeResourceVersion GaugeMetric
}

// reflectorMetrics holds all metrics for a reflector.
type reflectorMetrics struct {
	listDuration      HistogramMetric
	watchListDuration HistogramMetric
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
	}

	if id.Reserved() {
		metrics.numberOfQueuedItem = metricsProvider.NewQueuedItemMetric(id)
		metrics.processingLatency = metricsProvider.NewProcessingLatencyMetric(id)
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

func newReflectorMetrics(id InformerNameAndResource, metricsProvider InformerMetricsProvider) *reflectorMetrics {
	if metricsProvider == nil {
		metricsProvider = globalInformerMetricsProvider
	}
	metrics := &reflectorMetrics{
		listDuration:      noopMetric{},
		watchListDuration: noopMetric{},
	}

	if id.Reserved() {
		metrics.listDuration = metricsProvider.NewListDurationMetric(id, ListVerbList)
		metrics.watchListDuration = metricsProvider.NewListDurationMetric(id, ListVerbWatch)
	}

	return metrics
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

func (noopInformerMetricsProvider) NewListDurationMetric(InformerNameAndResource, string) HistogramMetric {
	return noopMetric{}
}
