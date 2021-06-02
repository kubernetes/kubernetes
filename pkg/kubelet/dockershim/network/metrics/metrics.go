// +build !dockerless

/*
Copyright 2017 The Kubernetes Authors.

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

package metrics

import (
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	// NetworkPluginOperationsKey is the key for operation count metrics.
	NetworkPluginOperationsKey = "network_plugin_operations_total"
	// NetworkPluginOperationsLatencyKey is the key for the operation latency metrics.
	NetworkPluginOperationsLatencyKey = "network_plugin_operations_duration_seconds"
	// NetworkPluginOperationsErrorsKey is the key for the operations error metrics.
	NetworkPluginOperationsErrorsKey = "network_plugin_operations_errors_total"

	// Keep the "kubelet" subsystem for backward compatibility.
	kubeletSubsystem = "kubelet"
)

var (
	// NetworkPluginOperationsLatency collects operation latency numbers by operation
	// type.
	NetworkPluginOperationsLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      kubeletSubsystem,
			Name:           NetworkPluginOperationsLatencyKey,
			Help:           "Latency in seconds of network plugin operations. Broken down by operation type.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)

	// NetworkPluginOperations collects operation counts by operation type.
	NetworkPluginOperations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeletSubsystem,
			Name:           NetworkPluginOperationsKey,
			Help:           "Cumulative number of network plugin operations by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)

	// NetworkPluginOperationsErrors collects operation errors by operation type.
	NetworkPluginOperationsErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeletSubsystem,
			Name:           NetworkPluginOperationsErrorsKey,
			Help:           "Cumulative number of network plugin operation errors by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(NetworkPluginOperationsLatency)
		legacyregistry.MustRegister(NetworkPluginOperations)
		legacyregistry.MustRegister(NetworkPluginOperationsErrors)
	})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
