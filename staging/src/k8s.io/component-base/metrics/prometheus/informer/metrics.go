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

package informer

import (
	"k8s.io/client-go/tools/cache"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	Subsystem               = "informer"
	StoredItemsKey          = "stored_items"
	QueuedItemsKey          = "queued_items"
	PendingNotificationsKey = "pending_notifications"
	RingGrowingCapacityKey  = "ring_growing_capacity"
	EventProcessDurationKey = "event_process_duration_seconds"
)

var (
	storedItems = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      Subsystem,
		Name:           StoredItemsKey,
		Help:           "Total number of items stored in the informer",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name"})
	queuedItems = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      Subsystem,
		Name:           QueuedItemsKey,
		Help:           "Total number of items queued in the informer",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name"})
	pendingNotifications = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      Subsystem,
		Name:           PendingNotificationsKey,
		Help:           "Total number of pending notifications in the informer",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name"})
	ringGrowingCapacity = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      Subsystem,
		Name:           RingGrowingCapacityKey,
		Help:           "Current capacity of the ring buffer in the informer",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name"})
	eventProcessDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem:      Subsystem,
		Name:           EventProcessDurationKey,
		Help:           "Duration of event processing in the informer",
		StabilityLevel: k8smetrics.ALPHA,
		Buckets:        []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
	}, []string{"name"})

	metrics = []k8smetrics.Registerable{
		storedItems, queuedItems, pendingNotifications, ringGrowingCapacity, eventProcessDuration,
	}
)

type informerMetricsProvider struct{}

// Register registers informer metrics.
func Register() {
	for _, m := range metrics {
		legacyregistry.MustRegister(m)
	}
	cache.SetInformerMetricsProvider(informerMetricsProvider{})
}

func (informerMetricsProvider) NewStoredItemMetric(name string) cache.GaugeMetric {
	return storedItems.WithLabelValues(name)
}

func (informerMetricsProvider) NewQueuedItemMetric(name string) cache.GaugeMetric {
	return queuedItems.WithLabelValues(name)
}

func (informerMetricsProvider) NewPendingNotificationsMetric(name string) cache.GaugeMetric {
	return pendingNotifications.WithLabelValues(name)
}

func (informerMetricsProvider) NewRingGrowingMetric(name string) cache.GaugeMetric {
	return ringGrowingCapacity.WithLabelValues(name)
}

func (informerMetricsProvider) NewPrcoessDurationMetric(name string) cache.HistogramMetric {
	return eventProcessDuration.WithLabelValues(name)
}
