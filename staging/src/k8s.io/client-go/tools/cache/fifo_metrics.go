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

// Package cache is a client-side caching mechanism. It is useful for
// reducing the number of server calls you'd otherwise need to make.
// Reflector watches a server and updates a Store. Two stores are provided;
// one that simply caches objects (for example, to allow a scheduler to
// list currently available nodes), and one that additionally acts as
// a FIFO queue (for example, to allow a scheduler to process incoming
// pods).
package cache

import (
	"k8s.io/klog/v2"
)

// FIFOMetricsProvider defines an interface for creating metrics that track FIFO queue operations.
type FIFOMetricsProvider interface {
	// NewQueuedItemMetric returns a gauge metric for tracking the total number of items
	// currently queued and waiting to be processed.
	//
	// For DeltaFIFO: Represents len(f.items) - the number of unique keys with pending deltas
	// For RealFIFO: Represents len(f.items) - the total number of individual delta events queued
	NewQueuedItemMetric(*Identifier) GaugeMetric
}

// fifoMetrics holds all metrics for a FIFO.
type fifoMetrics struct {
	numberOfQueuedItem GaugeMetric
}

type noopFIFOMetricsProvider struct{}

func newFIFOMetrics(id *Identifier, metricsProvider FIFOMetricsProvider) *fifoMetrics {
	if metricsProvider == nil {
		metricsProvider = noopFIFOMetricsProvider{}
	}
	metrics := &fifoMetrics{
		numberOfQueuedItem: noopMetric{},
	}

	if !id.IsUnique() {
		klog.ErrorS(nil, "FIFO metrics not published: empty name or duplicate identifier", "name", id.Name(), "itemType", id.ItemType())
		return metrics
	}

	metrics.numberOfQueuedItem = metricsProvider.NewQueuedItemMetric(id)
	return metrics
}

func (noopFIFOMetricsProvider) NewQueuedItemMetric(*Identifier) GaugeMetric {
	return noopMetric{}
}
