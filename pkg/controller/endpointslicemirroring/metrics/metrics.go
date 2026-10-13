/*
Copyright 2020 The Kubernetes Authors.

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

// EndpointSliceMirroringSubsystem is the name of the subsystem used for
// EndpointSliceMirroring controller.
const EndpointSliceMirroringSubsystem = "endpoint_slice_mirroring_controller"

var (
	// EndpointsAddedPerSync tracks the number of endpoints added on each
	// Endpoints sync.
	EndpointsAddedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "endpoints_added_per_sync",
			Help:           "Number of endpoints added on each Endpoints sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)
	// EndpointsUpdatedPerSync tracks the number of endpoints updated on each
	// Endpoints sync.
	EndpointsUpdatedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "endpoints_updated_per_sync",
			Help:           "Number of endpoints updated on each Endpoints sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)
	// EndpointsRemovedPerSync tracks the number of endpoints removed on each
	// Endpoints sync.
	EndpointsRemovedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "endpoints_removed_per_sync",
			Help:           "Number of endpoints removed on each Endpoints sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)
	// AddressesSkippedPerSync tracks the number of addresses skipped on each
	// Endpoints sync due to being invalid or exceeding MaxEndpointsPerSubset.
	AddressesSkippedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "addresses_skipped_per_sync",
			Help:           "Number of addresses skipped on each Endpoints sync due to being invalid or exceeding MaxEndpointsPerSubset",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)
	// EndpointsSyncDuration tracks how long syncEndpoints() takes in a number
	// of Seconds.
	EndpointsSyncDuration = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "endpoints_sync_duration",
			Help:           "Duration of syncEndpoints() in seconds",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(0.001, 2, 15),
		},
		[]string{},
	)
	// EndpointsDesired tracks the total number of desired endpoints.
	EndpointsDesired = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "endpoints_desired",
			Help:           "Number of endpoints desired",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
	// NumEndpointSlices tracks the number of EndpointSlices in a cluster.
	NumEndpointSlices = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
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
			Subsystem:      EndpointSliceMirroringSubsystem,
			Name:           "desired_endpoint_slices",
			Help:           "Number of EndpointSlices that would exist with perfect endpoint allocation",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)
	// EndpointSliceChanges tracks the number of changes to Endpoint Slices.
	EndpointSliceChanges = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      EndpointSliceMirroringSubsystem,
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
		legacyregistry.MustRegister(EndpointsUpdatedPerSync)
		legacyregistry.MustRegister(EndpointsRemovedPerSync)
		legacyregistry.MustRegister(AddressesSkippedPerSync)
		legacyregistry.MustRegister(EndpointsSyncDuration)
		legacyregistry.MustRegister(EndpointsDesired)
		legacyregistry.MustRegister(NumEndpointSlices)
		legacyregistry.MustRegister(DesiredEndpointSlices)
		legacyregistry.MustRegister(EndpointSliceChanges)
	})
}
