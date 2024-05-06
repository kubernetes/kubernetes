/*
Copyright 2023 The Kubernetes Authors.

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

package cloud

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	metricsSubsystem = "node_controller"
)

var (
	removeCloudProviderTaintDelay = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      metricsSubsystem,
			Name:           "cloud_provider_taint_removal_delay_seconds",
			Help:           "Number of seconds after node creation when NodeController removed the cloud-provider taint of a single node.",
			Buckets:        metrics.ExponentialBuckets(1, 4, 6), // 1s -> ~17m
			StabilityLevel: metrics.ALPHA,
		},
	)
	initialNodeSyncDelay = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      metricsSubsystem,
			Name:           "initial_node_sync_delay_seconds",
			Help:           "Number of seconds after node creation when NodeController finished the initial synchronization of a single node.",
			Buckets:        metrics.ExponentialBuckets(1, 4, 6), // 1s -> ~17m
			StabilityLevel: metrics.ALPHA,
		},
	)
)

var metricRegistration sync.Once

// registerMetrics registers the metrics that are to be monitored.
func registerMetrics() {
	metricRegistration.Do(func() {
		legacyregistry.MustRegister(removeCloudProviderTaintDelay)
		legacyregistry.MustRegister(initialNodeSyncDelay)
	})
}
