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

package fifo

import (
	"k8s.io/client-go/tools/cache"
	k8smetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	Subsystem      = "fifo"
	StoredItemsKey = "stored_items"
	QueuedItemsKey = "queued_items"
)

var (
	storedItems = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      Subsystem,
		Name:           StoredItemsKey,
		Help:           "Total number of items stored in the FIFO",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "item_type"})

	queuedItems = k8smetrics.NewGaugeVec(&k8smetrics.GaugeOpts{
		Subsystem:      Subsystem,
		Name:           QueuedItemsKey,
		Help:           "Total number of items queued in the FIFO",
		StabilityLevel: k8smetrics.ALPHA,
	}, []string{"name", "item_type"})

	metrics = []k8smetrics.Registerable{storedItems, queuedItems}
)

type fifoMetricsProvider struct{}

// Register registers FIFO metrics.
func Register() {
	for _, m := range metrics {
		legacyregistry.MustRegister(m)
	}
	cache.SetFIFOMetricsProvider(fifoMetricsProvider{})
}

func (fifoMetricsProvider) NewStoredItemMetric(name string, itemType string) cache.GaugeMetric {
	return storedItems.WithLabelValues(name, itemType)
}

func (fifoMetricsProvider) NewQueuedItemMetric(name string, itemType string) cache.GaugeMetric {
	return queuedItems.WithLabelValues(name, itemType)
}
