/*
Copyright 2024 The Kubernetes Authors.

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

// fifoMetrics tracks metrics for a FIFO queue.
type fifoMetrics struct {
	clock clock.Clock

	// numberOfStoredItem represents the total number of items in store.
	numberOfStoredItem GaugeMetric

	// numberOfQueuedItem represents the total number of items in queue.
	numberOfQueuedItem GaugeMetric
}

// FIFOMetricsProvider generates various metrics used by the FIFO queue.
type FIFOMetricsProvider interface {
	// NewStoredItemMetric returns metric for total number of item in store.
	NewStoredItemMetric(name string, itemType string) GaugeMetric
	// NewQueuedItemMetric returns metric for total number of items in queue.
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

// newFIFOMetrics creates a new fifoMetrics instance for tracking metrics related to the FIFO queue.
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

// SetFIFOMetricsProvider sets the metrics provider for the FIFO queue.
func SetFIFOMetricsProvider(metricsProvider FIFOMetricsProvider) {
	setGlobalFIFOMetricsProvider.Do(func() {
		globalFIFOMetricsProvider = metricsProvider
	})
}
