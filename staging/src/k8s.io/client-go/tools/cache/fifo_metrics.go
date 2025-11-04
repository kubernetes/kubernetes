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

package cache

import (
	"sync"
)

// FIFOMetricsProvider defines an interface for creating metrics that track FIFO queue operations.
type FIFOMetricsProvider interface {
	// NewQueuedItemMetric returns a gauge metric for tracking the total number of items
	// currently queued and waiting to be processed.
	NewQueuedItemMetric(*Identifier) GaugeMetric
}

// SetFIFOMetricsProvider sets the global metrics provider for FIFO queues.
func SetFIFOMetricsProvider(metricsProvider FIFOMetricsProvider) {
	setFIFOMetricsOnce.Do(func() {
		fifoMetricsProvider = metricsProvider
	})
}

var (
	fifoMetricsProvider FIFOMetricsProvider = noopFIFOMetricsProvider{}
	setFIFOMetricsOnce  sync.Once
)

// newFIFOMetrics creates a new fifoMetrics instance for tracking metrics related to FIFO queues.
func newFIFOMetrics(id *Identifier, metricsProvider FIFOMetricsProvider) *fifoMetrics {
	if metricsProvider == nil {
		metricsProvider = fifoMetricsProvider
	}
	metrics := &fifoMetrics{
		numberOfQueuedItem: noopMetric{},
	}

	if id.IsUnique() {
		metrics.numberOfQueuedItem = metricsProvider.NewQueuedItemMetric(id)
	}
	return metrics
}

// fifoMetrics tracks metrics for a FIFO queue, including the number of stored and queued items.
type fifoMetrics struct {
	numberOfQueuedItem GaugeMetric
}

type noopFIFOMetricsProvider struct{}

func (noopFIFOMetricsProvider) NewQueuedItemMetric(*Identifier) GaugeMetric {
	return noopMetric{}
}
