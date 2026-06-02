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

	// nodesProcessed is a counter vector that tracks the total number of nodes processed during the sync loop.
	// It is labeled by the execution path: "cache" for fast local check, or "cloud" for slow cloud provider API check.
	nodesProcessed = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      metricsSubsystem,
			Name:           "nodes_processed_total",
			Help:           "The total number of nodes processed during MonitorNodes loop execution, labeled by execution path.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"path"},
	)
)

var metricRegistration sync.Once

// registerMetrics registers the metrics that are to be monitored.
func registerMetrics() {
	metricRegistration.Do(func() {
		legacyregistry.MustRegister(monitorNodesDuration)
		legacyregistry.MustRegister(nodesProcessed)
	})
}
