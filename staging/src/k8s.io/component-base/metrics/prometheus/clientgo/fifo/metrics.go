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

package fifo

import (
	"sync"

	"k8s.io/client-go/tools/cache"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	subsystem = "informer"

	fifoQueuedItems = k8smetrics.NewGaugeVec(
		&k8smetrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "queued_items",
			Help:           "Number of items currently queued in the FIFO.",
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"name", "group", "version", "resource"},
	)
	fifoProcessingLatency = k8smetrics.NewHistogramVec(
		&k8smetrics.HistogramOpts{
			Subsystem:      subsystem,
			Name:           "processing_latency_seconds",
			Help:           "Time taken to process events after popping from the queue.",
			Buckets:        []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
			StabilityLevel: k8smetrics.ALPHA,
		},
		[]string{"name", "group", "version", "resource"},
	)
	registerOnce sync.Once
)

func init() {
	Register()
}

// Register registers FIFO metrics and sets the metrics provider.
func Register() {
	registerOnce.Do(func() {
		legacyregistry.MustRegister(fifoQueuedItems)
		legacyregistry.MustRegister(fifoProcessingLatency)
	})
	cache.SetFIFOMetricsProvider(fifoMetricsProvider{})
}

type fifoMetricsProvider struct{}

func (fifoMetricsProvider) NewQueuedItemMetric(id cache.InformerNameAndResource) cache.GaugeMetric {
	return &reservedGaugeMetric{
		id: id,
		gauge: fifoQueuedItems.WithLabelValues(
			id.Name(),
			id.GroupVersionResource().Group,
			id.GroupVersionResource().Version,
			id.GroupVersionResource().Resource,
		),
	}
}

func (fifoMetricsProvider) NewProcessingLatencyMetric(id cache.InformerNameAndResource) cache.HistogramMetric {
	return &reservedHistogramMetric{
		id: id,
		histogram: fifoProcessingLatency.WithLabelValues(
			id.Name(),
			id.GroupVersionResource().Group,
			id.GroupVersionResource().Version,
			id.GroupVersionResource().Resource,
		),
	}
}

// reservedGaugeMetric wraps a gauge and only updates it if the identifier
// is still reserved. This supports dynamic informers (e.g., GC, ResourceQuota)
// that may shut down while the process is still running.
type reservedGaugeMetric struct {
	id    cache.InformerNameAndResource
	gauge cache.GaugeMetric
}

func (r *reservedGaugeMetric) Set(value float64) {
	if r.id.Reserved() {
		r.gauge.Set(value)
	}
}

// reservedHistogramMetric wraps a histogram and only updates it if the identifier
// is still reserved. This supports dynamic informers (e.g., GC, ResourceQuota)
// that may shut down while the process is still running.
type reservedHistogramMetric struct {
	id        cache.InformerNameAndResource
	histogram cache.HistogramMetric
}

func (r *reservedHistogramMetric) Observe(value float64) {
	if r.id.Reserved() {
		r.histogram.Observe(value)
	}
}
