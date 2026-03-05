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

package cache

import (
	"sync"
)

// ProcessorListenerMetricsProvider creates metrics for processorListener.
type ProcessorListenerMetricsProvider interface {
	// NewPendingNotificationsMetric returns a gauge metric for tracking the number of
	// pending notifications buffered in a processorListener's ring buffer.
	// The handlerName parameter distinguishes multiple handlers registered
	// on the same informer (identified by id).
	NewPendingNotificationsMetric(id InformerNameAndResource, handlerName string) GaugeMetric
}

type noopProcessorListenerMetricsProvider struct{}

func (noopProcessorListenerMetricsProvider) NewPendingNotificationsMetric(InformerNameAndResource, string) GaugeMetric {
	return noopMetric{}
}

var (
	globalProcessorListenerMetricsProvider  ProcessorListenerMetricsProvider = noopProcessorListenerMetricsProvider{}
	setProcessorListenerMetricsProviderOnce sync.Once
)

// processorListenerMetrics holds all metrics for a processorListener.
type processorListenerMetrics struct {
	pendingNotifications GaugeMetric
}

// SetProcessorListenerMetricsProvider sets the metrics provider for all subsequently created
// processorListeners. Only the first call has an effect.
func SetProcessorListenerMetricsProvider(provider ProcessorListenerMetricsProvider) {
	setProcessorListenerMetricsProviderOnce.Do(func() {
		globalProcessorListenerMetricsProvider = provider
	})
}

func newProcessorListenerMetrics(id InformerNameAndResource, handlerName string) *processorListenerMetrics {
	metrics := &processorListenerMetrics{
		pendingNotifications: noopMetric{},
	}

	if id.Reserved() {
		metrics.pendingNotifications = globalProcessorListenerMetricsProvider.NewPendingNotificationsMetric(id, handlerName)
	}

	return metrics
}
