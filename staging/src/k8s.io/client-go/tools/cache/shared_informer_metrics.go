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
// FIFO metrics have been decoupled from informer metrics and moved to a separate file (fifo_metrics.go).
type informerMetrics struct {
	clock clock.Clock

	// metricsProvider is the metrics provider for the informer.
	metricsProvider InformerMetricsProvider

	// mu protects concurrent access to eventHandlerMetrics
	mu sync.Mutex

	// eventHandlerMetrics tracks the metrics for each event handler.
	eventHandlerMetrics map[string]eventHandlerMetrics
}

// eventHandlerMetrics tracks the metrics for each event handler.
type eventHandlerMetrics struct {
	// numberOfPendingNotifications represents the total number of notifications pending
	// to be delivered to this event handler.
	numberOfPendingNotifications GaugeMetric

	// sizeOfRingGrowing represents the size of the ring buffer that grows when the
	// event handler cannot keep up with the rate of incoming events.
	sizeOfRingGrowing GaugeMetric

	// processDuration represents the duration of processing events for this event handler.
	// It measures how long it takes for the event handler to process each event.
	processDuration HistogramMetric
}

// InformerMetricsProvider generates various metrics used by the shared informer.
type InformerMetricsProvider interface {
	// NewPendingNotificationsMetric returns metric for pending notifications.
	NewPendingNotificationsMetric(informerName string, resourceType string, eventHandlerName string) GaugeMetric
	// NewRingGrowingMetric returns metric for the size of the growing ring buffer.
	NewRingGrowingMetric(informerName string, resourceType string, eventHandlerName string) GaugeMetric
	// NewProcessDurationMetric returns metric for event processing duration.
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
		clock:               &clock.RealClock{},
		metricsProvider:     metricsProvider,
		eventHandlerMetrics: make(map[string]eventHandlerMetrics),
	}
}

// SetInformerMetricsProvider sets the metrics provider for the informer.
func SetInformerMetricsProvider(metricsProvider InformerMetricsProvider) {
	setGlobalInformerMetricsProvider.Do(func() {
		globalInformerMetricsProvider = metricsProvider
	})
}

// addEventHandlerMetrics adds metrics for a new event handler to the map
func (m *informerMetrics) addEventHandlerMetrics(handlerName string, metrics eventHandlerMetrics) {
	if m == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.eventHandlerMetrics[handlerName] = metrics
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

	m.addEventHandlerMetrics(handlerName, *metrics)

	return metrics
}
