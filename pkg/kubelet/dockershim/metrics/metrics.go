// +build !dockerless

/*
Copyright 2015 The Kubernetes Authors.

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
	// DockerOperationsKey is the key for docker operation metrics.
	DockerOperationsKey = "docker_operations_total"
	// DockerOperationsLatencyKey is the key for the operation latency metrics.
	DockerOperationsLatencyKey = "docker_operations_duration_seconds"
	// DockerOperationsErrorsKey is the key for the operation error metrics.
	DockerOperationsErrorsKey = "docker_operations_errors_total"
	// DockerOperationsTimeoutKey is the key for the operation timeout metrics.
	DockerOperationsTimeoutKey = "docker_operations_timeout_total"

	// Keep the "kubelet" subsystem for backward compatibility.
	kubeletSubsystem = "kubelet"
)

var (
	// DockerOperationsLatency collects operation latency numbers by operation
	// type.
	DockerOperationsLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      kubeletSubsystem,
			Name:           DockerOperationsLatencyKey,
			Help:           "Latency in seconds of Docker operations. Broken down by operation type.",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
	// DockerOperations collects operation counts by operation type.
	DockerOperations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeletSubsystem,
			Name:           DockerOperationsKey,
			Help:           "Cumulative number of Docker operations by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
	// DockerOperationsErrors collects operation errors by operation
	// type.
	DockerOperationsErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeletSubsystem,
			Name:           DockerOperationsErrorsKey,
			Help:           "Cumulative number of Docker operation errors by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
	// DockerOperationsTimeout collects operation timeouts by operation type.
	DockerOperationsTimeout = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      kubeletSubsystem,
			Name:           DockerOperationsTimeoutKey,
			Help:           "Cumulative number of Docker operation timeout by operation type.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation_type"},
	)
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(DockerOperationsLatency)
		legacyregistry.MustRegister(DockerOperations)
		legacyregistry.MustRegister(DockerOperationsErrors)
		legacyregistry.MustRegister(DockerOperationsTimeout)
	})
}

// SinceInSeconds gets the time since the specified start in seconds.
func SinceInSeconds(start time.Time) float64 {
	return time.Since(start).Seconds()
}
