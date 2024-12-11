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

	"k8s.io/utils/clock"
)

// fifoMetrics tracks metrics for a FIFO queue, including the number of stored and queued items.
type fifoMetrics struct {
	clock clock.Clock

	numberOfStoredItem GaugeMetric
	numberOfQueuedItem GaugeMetric
}

// FIFOMetricsProvider defines an interface for creating metrics that track FIFO queue operations.
type FIFOMetricsProvider interface {
	// NewStoredItemMetric returns a gauge metric for tracking the total number of items
	// currently stored in the FIFO's internal storage.
	//
	// For DeltaFIFO: Represents len(f.items) - the number of unique keys with pending deltas
	// For RealFIFO: Would represent len(f.items) - the total number of individual deltas
	//
	// Parameters:
	//   - name: Identifier for the queue (e.g., controller name, queue name)
	//   - itemType: Type of objects being queued (e.g., "pods", "services")
	NewStoredItemMetric(name string, itemType string) GaugeMetric

	// NewQueuedItemMetric returns a gauge metric for tracking the total number of items
	// currently queued and waiting to be processed.
	//
	// For DeltaFIFO: Represents len(f.queue) - the number of keys in processing order
	// For RealFIFO: Would represent len(f.items) - same as stored items due to strict ordering
	//
	// Parameters:
	//   - name: Identifier for the queue (e.g., controller name, queue name)
	//   - itemType: Type of objects being queued (e.g., "pods", "services")
	NewQueuedItemMetric(name string, itemType string) GaugeMetric
}

type noopFIFOMetricsProvider struct{}

func (noopFIFOMetricsProvider) NewStoredItemMetric(name string, itemType string) GaugeMetric {
	return noopMetric{}
}

func (noopFIFOMetricsProvider) NewQueuedItemMetric(name string, itemType string) GaugeMetric {
	return noopMetric{}
}

var globalFIFOMetricsProvider FIFOMetricsProvider = noopFIFOMetricsProvider{}
var setGlobalFIFOMetricsProvider sync.Once

// newFIFOMetrics creates a new fifoMetrics instance for tracking metrics related to FIFO queues.
func newFIFOMetrics(queueName string, itemType string, metricsProvider FIFOMetricsProvider) *fifoMetrics {
	var ret *fifoMetrics
	if queueName == "" || itemType == "" {
		return ret
	}

	if metricsProvider == nil {
		metricsProvider = globalFIFOMetricsProvider
	}

	return &fifoMetrics{
		clock:              &clock.RealClock{},
		numberOfStoredItem: metricsProvider.NewStoredItemMetric(queueName, itemType),
		numberOfQueuedItem: metricsProvider.NewQueuedItemMetric(queueName, itemType),
	}
}

// SetFIFOMetricsProvider sets the global metrics provider for FIFO queues.
func SetFIFOMetricsProvider(metricsProvider FIFOMetricsProvider) {
	setGlobalFIFOMetricsProvider.Do(func() {
		globalFIFOMetricsProvider = metricsProvider
	})
}
