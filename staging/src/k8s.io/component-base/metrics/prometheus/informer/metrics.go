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

package informer

import (
	"k8s.io/client-go/tools/cache"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	Subsystem               = "informer"
	PendingNotificationsKey = "pending_notifications"
	RingGrowingCapacityKey  = "ring_growing_capacity"
	EventProcessDurationKey = "event_process_duration_seconds"
)

var (
	pendingNotifications = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem: Subsystem,
		Name:      PendingNotificationsKey,
		Help: "Total number of pending notifications in the ring buffer for each event handler in the informer.\n" +
			"Labels:\n" +
			"* informer_name: Name of the informer instance (e.g., pod_informer, deployment_informer).\n" +
			"* resource_type: Type of Kubernetes resource being watched (e.g., Pod, Deployment).\n" +
			"* event_handler_name: Identifier of the event handler. If not explicitly specified by the user, a default name is generated in the format: <informer>_<resource>_<handler_type>_<hash>, where the hash ensures uniqueness.",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"informer_name", "resource_type", "event_handler_name"})
	ringGrowingCapacity = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem: Subsystem,
		Name:      RingGrowingCapacityKey,
		Help: "Current capacity of the ring buffer for each event handler in the informer.\n" +
			"Labels:\n" +
			"* informer_name: Name of the informer instance (e.g., pod_informer, deployment_informer).\n" +
			"* resource_type: Type of Kubernetes resource being watched (e.g., Pod, Deployment).\n" +
			"* event_handler_name: Identifier of the event handler. If not explicitly specified by the user, a default name is generated in the format: <informer>_<resource>_<handler_type>_<hash>, where the hash ensures uniqueness.",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"informer_name", "resource_type", "event_handler_name"})
	eventProcessDuration = k8smetrics.NewHistogramVec(&k8smetrics.HistogramOpts{
		Subsystem: Subsystem,
		Name:      EventProcessDurationKey,
		Help: "Duration of event processing for each event handler in the informer.\n" +
			"Labels:\n" +
			"* informer_name: Name of the informer instance (e.g., pod_informer, deployment_informer).\n" +
			"* resource_type: Type of Kubernetes resource being watched (e.g., Pod, Deployment).\n" +
			"* event_handler_name: Identifier of the event handler. If not explicitly specified by the user, a default name is generated in the format: <informer>_<resource>_<handler_type>_<hash>, where the hash ensures uniqueness.",
		StabilityLevel: k8smetrics.ALPHA,
		Buckets:        []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
	}, []string{"informer_name", "resource_type", "event_handler_name"})

	metrics = []k8smetrics.Registerable{pendingNotifications, ringGrowingCapacity, eventProcessDuration}
)

type informerMetricsProvider struct{}

// Register registers informer metrics.
func Register() {
	for _, m := range metrics {
		legacyregistry.MustRegister(m)
	}
	cache.SetInformerMetricsProvider(informerMetricsProvider{})
}

func (informerMetricsProvider) NewPendingNotificationsMetric(informerName string, resourceType string, eventHandlerName string) cache.GaugeMetric {
	return pendingNotifications.WithLabelValues(informerName, resourceType, eventHandlerName)
}

func (informerMetricsProvider) NewRingGrowingMetric(informerName string, resourceType string, eventHandlerName string) cache.GaugeMetric {
	return ringGrowingCapacity.WithLabelValues(informerName, resourceType, eventHandlerName)
}

func (informerMetricsProvider) NewProcessDurationMetric(informerName string, resourceType string, eventHandlerName string) cache.HistogramMetric {
	return eventProcessDuration.WithLabelValues(informerName, resourceType, eventHandlerName)
}
