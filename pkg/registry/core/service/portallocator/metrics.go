/*
Copyright 2022 The Kubernetes Authors.

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

package portallocator

import (
	"sync"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

const (
	namespace = "kube_apiserver"
	subsystem = "nodeport_allocator"
)

var (
	// nodePortAllocated indicates the amount of ports allocated by NodePort Service.
	nodePortAllocated = metrics.NewGauge(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocated_ports",
			Help:           "Gauge measuring the number of allocated NodePorts for Services",
			StabilityLevel: metrics.ALPHA,
		},
	)
	// nodePortAvailable indicates the amount of ports available by NodePort Service.
	nodePortAvailable = metrics.NewGauge(
		&metrics.GaugeOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "available_ports",
			Help:           "Gauge measuring the number of available NodePorts for Services",
			StabilityLevel: metrics.ALPHA,
		},
	)
	// nodePortAllocation counts the total number of ports allocation and allocation mode: static or dynamic.
	nodePortAllocations = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocation_total",
			Help:           "Number of NodePort allocations",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"scope"},
	)
	// nodePortAllocationErrors counts the number of error trying to allocate a nodePort and allocation mode: static or dynamic.
	nodePortAllocationErrors = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "allocation_errors_total",
			Help:           "Number of errors trying to allocate NodePort",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"scope"},
	)
)

var registerMetricsOnce sync.Once

func registerMetrics() {
	registerMetricsOnce.Do(func() {
		legacyregistry.MustRegister(nodePortAllocated)
		legacyregistry.MustRegister(nodePortAvailable)
		legacyregistry.MustRegister(nodePortAllocations)
		legacyregistry.MustRegister(nodePortAllocationErrors)
	})
}

// metricsRecorderInterface is the interface to record metrics.
type metricsRecorderInterface interface {
	setAllocated(allocated int)
	setAvailable(available int)
	incrementAllocations(scope string)
	incrementAllocationErrors(scope string)
}

// metricsRecorder implements metricsRecorderInterface.
type metricsRecorder struct{}

func (m *metricsRecorder) setAllocated(allocated int) {
	nodePortAllocated.Set(float64(allocated))
}

func (m *metricsRecorder) setAvailable(available int) {
	nodePortAvailable.Set(float64(available))
}

func (m *metricsRecorder) incrementAllocations(scope string) {
	nodePortAllocations.WithLabelValues(scope).Inc()
}

func (m *metricsRecorder) incrementAllocationErrors(scope string) {
	nodePortAllocationErrors.WithLabelValues(scope).Inc()
}

// emptyMetricsRecorder is a null object implements metricsRecorderInterface.
type emptyMetricsRecorder struct{}

func (*emptyMetricsRecorder) setAllocated(allocated int)             {}
func (*emptyMetricsRecorder) setAvailable(available int)             {}
func (*emptyMetricsRecorder) incrementAllocations(scope string)      {}
func (*emptyMetricsRecorder) incrementAllocationErrors(scope string) {}
