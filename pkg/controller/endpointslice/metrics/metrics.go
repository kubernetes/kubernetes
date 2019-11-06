/*
Copyright 2019 The Kubernetes Authors.

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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// EndpointSliceSubsystem - subsystem name used for Endpoint Slices.
const EndpointSliceSubsystem = "endpoint_slice_controller"

var (
	// EndpointsAddedPerSync tracks the number of endpoints added on each
	// Service sync.
	EndpointsAddedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceSubsystem,
			Name:           "endpoints_added_per_sync",
			Help:           "Number of endpoints added on each Service sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)
	// EndpointsRemovedPerSync tracks the number of endpoints removed on each
	// Service sync.
	EndpointsRemovedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceSubsystem,
			Name:           "endpoints_removed_per_sync",
			Help:           "Number of endpoints removed on each Service sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)
	// EndpointsDesired tracks the total number of desired endpoints.
	EndpointsDesired = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EndpointSliceSubsystem,
			Name:           "endpoints_desired",
			Help:           "Number of endpoints desired",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
	// NumEndpointSlices tracks the number of EndpointSlices in a cluster.
	NumEndpointSlices = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EndpointSliceSubsystem,
			Name:           "num_endpoint_slices",
			Help:           "Number of EndpointSlices",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
	// DesiredEndpointSlices tracks the number of EndpointSlices that would
	// exist with perfect endpoint allocation.
	DesiredEndpointSlices = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EndpointSliceSubsystem,
			Name:           "desired_endpoint_slices",
			Help:           "Number of EndpointSlices that would exist with perfect endpoint allocation",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)

	// EndpointSliceChanges tracks the number of changes to Endpoint Slices.
	EndpointSliceChanges = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      EndpointSliceSubsystem,
			Name:           "changes",
			Help:           "Number of EndpointSlice changes",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation"},
	)
)

var registerMetrics sync.Once

// RegisterMetrics registers EndpointSlice metrics.
func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(EndpointsAddedPerSync)
		legacyregistry.MustRegister(EndpointsRemovedPerSync)
		legacyregistry.MustRegister(EndpointsDesired)
		legacyregistry.MustRegister(NumEndpointSlices)
		legacyregistry.MustRegister(DesiredEndpointSlices)
		legacyregistry.MustRegister(EndpointSliceChanges)
	})
}
