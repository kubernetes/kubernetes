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

// informerMetrics tracks metrics for a shared informer, including metrics for each event handler.
type informerMetrics struct {
	clock clock.Clock

	metricsProvider InformerMetricsProvider
}

// eventHandlerMetrics tracks the metrics for each event handler.
type eventHandlerMetrics struct {
	// numberOfPendingNotifications tracks the count of notifications that are currently queued
	// in the ring buffer and waiting to be processed by this event handler.
	numberOfPendingNotifications GaugeMetric

	// sizeOfRingGrowing represents the current capacity of the ring buffer used to store pending notifications
	// for this event handler. The ring buffer grows dynamically as needed to accommodate more notifications.
	sizeOfRingGrowing GaugeMetric

	// processDuration tracks the time taken to process each event by this event handler.
	// This metric helps identify performance bottlenecks and monitor event processing latency.
	processDuration HistogramMetric
}

// InformerMetricsProvider defines the interface for generating metrics in shared informers.
type InformerMetricsProvider interface {
	// NewPendingNotificationsMetric returns a metric for the number of pending notifications for an event handler.
	NewPendingNotificationsMetric(informerName string, resourceType string, eventHandlerName string) GaugeMetric
	// NewRingGrowingMetric returns a metric for the size of the growing ring buffer for an event handler.
	NewRingGrowingMetric(informerName string, resourceType string, eventHandlerName string) GaugeMetric
	// NewProcessDurationMetric returns a metric for the duration of event processing for an event handler.
	NewProcessDurationMetric(informerName string, resourceType string, eventHandlerName string) HistogramMetric
}

type noopInformerMetricsProvider struct{}

func (noopInformerMetricsProvider) NewPendingNotificationsMetric(informerName string, resourceType string, eventHandlerName string) GaugeMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewRingGrowingMetric(informerName string, resourceType string, eventHandlerName string) GaugeMetric {
	return noopMetric{}
}

func (noopInformerMetricsProvider) NewProcessDurationMetric(informerName string, resourceType string, eventHandlerName string) HistogramMetric {
	return noopMetric{}
}

var globalInformerMetricsProvider InformerMetricsProvider = noopInformerMetricsProvider{}
var setGlobalInformerMetricsProvider sync.Once

// newInformerMetrics creates a new informerMetrics instance.
func newInformerMetrics(name string, resourceType string, metricsProvider InformerMetricsProvider) *informerMetrics {
	var ret *informerMetrics
	if name == "" || resourceType == "" {
		return ret
	}

	if metricsProvider == nil {
		metricsProvider = globalInformerMetricsProvider
	}

	return &informerMetrics{
		clock:           &clock.RealClock{},
		metricsProvider: metricsProvider,
	}
}

// SetInformerMetricsProvider sets the metrics provider for the informer.
func SetInformerMetricsProvider(metricsProvider InformerMetricsProvider) {
	setGlobalInformerMetricsProvider.Do(func() {
		globalInformerMetricsProvider = metricsProvider
	})
}

// NewEventHandlerMetrics creates a new eventHandlerMetrics for the given event handler name
func (m *informerMetrics) NewEventHandlerMetrics(informerName string, resourceType string, handlerName string) *eventHandlerMetrics {
	if m == nil {
		return nil
	}

	metrics := &eventHandlerMetrics{
		numberOfPendingNotifications: m.metricsProvider.NewPendingNotificationsMetric(informerName, resourceType, handlerName),
		sizeOfRingGrowing:            m.metricsProvider.NewRingGrowingMetric(informerName, resourceType, handlerName),
		processDuration:              m.metricsProvider.NewProcessDurationMetric(informerName, resourceType, handlerName),
	}

	return metrics
}
