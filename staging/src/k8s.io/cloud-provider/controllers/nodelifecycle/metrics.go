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

package nodelifecycle

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	metricsSubsystem = "cloud_node_lifecycle_controller"
)

var (
	// monitorNodesDuration is a histogram that measures the duration in seconds of a single MonitorNodes loop execution.
	monitorNodesDuration = metrics.NewHistogram(
		&metrics.HistogramOpts{
			Subsystem:      metricsSubsystem,
			Name:           "monitor_nodes_duration_seconds",
			Help:           "A metric measuring the duration in seconds of a single MonitorNodes loop execution.",
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
			StabilityLevel: metrics.ALPHA,
		},
	)

	// cloudProviderCalls is a counter vector that tracks the total number of cloud provider API calls.
	// It is labeled by the operation (e.g., "instance_exists", "instance_shutdown", "instance_metadata") and the result ("success", "error", "instance_not_found", "not_implemented", "canceled").
	cloudProviderCalls = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      metricsSubsystem,
			Name:           "cloud_provider_calls_total",
			Help:           "The total number of cloud provider API calls, labeled by operation and result.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation", "result"},
	)
)

var metricRegistration sync.Once

// registerMetrics registers the metrics that are to be monitored.
func registerMetrics() {
	metricRegistration.Do(func() {
		legacyregistry.MustRegister(monitorNodesDuration)
		legacyregistry.MustRegister(cloudProviderCalls)
	})
}
