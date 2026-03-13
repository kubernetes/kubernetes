/*
Copyright 2021 The Kubernetes Authors.

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

package ipallocator

import (
	"sync"
	"time"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "kube_apiserver"
	subsystem = "clusterip_allocator"
)

var (
	// clusterIPAllocated indicates the amount of cluster IP allocated by Service CIDR.
	clusterIPAllocated = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocated_ips",
			Help:           "Gauge measuring the number of allocated IPs for Services",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"cidr"},
	)
	// clusterIPAvailable indicates the amount of cluster IP available by Service CIDR.
	clusterIPAvailable = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "available_ips",
			Help:           "Gauge measuring the number of available IPs for Services",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"cidr"},
	)
	// clusterIPAllocation counts the total number of ClusterIP allocation and allocation mode: static or dynamic.
	clusterIPAllocations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocation_total",
			Help:           "Number of Cluster IPs allocations",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"cidr", "scope"},
	)
	// clusterIPAllocationErrors counts the number of error trying to allocate a ClusterIP and allocation mode: static or dynamic.
	clusterIPAllocationErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocation_errors_total",
			Help:           "Number of errors trying to allocate Cluster IPs",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"cidr", "scope"},
	)
	clusterIPAllocationLatency = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocation_duration_seconds",
			Help:           "Duration in seconds to allocate a Cluster IP by ServiceCIDR",
			Buckets:        metrics.DefBuckets,
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"cidr"},
	)
)

var registerMetricsOnce sync.Once

func registerMetrics() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(clusterIPAllocated)
		legacyregistry.MustRegister(clusterIPAvailable)
		legacyregistry.MustRegister(clusterIPAllocations)
		legacyregistry.MustRegister(clusterIPAllocationErrors)
		legacyregistry.MustRegister(clusterIPAllocationLatency)
	})
}

// metricsRecorderInterface is the interface to record metrics.
type metricsRecorderInterface interface {
	setAllocated(cidr string, allocated int)
	setAvailable(cidr string, available int)
	setLatency(cidr string, latency time.Duration)
	incrementAllocations(cidr, scope string)
	incrementAllocationErrors(cidr, scope string)
}

// metricsRecorder implements metricsRecorderInterface.
type metricsRecorder struct{}

func (m *metricsRecorder) setAllocated(cidr string, allocated int) {
	clusterIPAllocated.WithLabelValues(cidr).Set(float64(allocated))
}

func (m *metricsRecorder) setAvailable(cidr string, available int) {
	clusterIPAvailable.WithLabelValues(cidr).Set(float64(available))
}

func (m *metricsRecorder) setLatency(cidr string, latency time.Duration) {
	clusterIPAllocationLatency.WithLabelValues(cidr).Observe(latency.Seconds())
}

func (m *metricsRecorder) incrementAllocations(cidr, scope string) {
	clusterIPAllocations.WithLabelValues(cidr, scope).Inc()
}

func (m *metricsRecorder) incrementAllocationErrors(cidr, scope string) {
	clusterIPAllocationErrors.WithLabelValues(cidr, scope).Inc()
}

// emptyMetricsRecorder is a null object implements metricsRecorderInterface.
type emptyMetricsRecorder struct{}

func (*emptyMetricsRecorder) setAllocated(cidr string, allocated int)       {}
func (*emptyMetricsRecorder) setAvailable(cidr string, available int)       {}
func (*emptyMetricsRecorder) setLatency(cidr string, latency time.Duration) {}
func (*emptyMetricsRecorder) incrementAllocations(cidr, scope string)       {}
func (*emptyMetricsRecorder) incrementAllocationErrors(cidr, scope string)  {}
