/*
Copyright 2024 The Kubernetes Authors.

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

var endpointSliceMetricsMap sync.Map

// EndpointSliceMetrics holds the metrics instruments for a particular subsystem.
type EndpointSliceMetrics struct {
	EndpointsAddedPerSync              *metrics.HistogramVec
	EndpointsRemovedPerSync            *metrics.HistogramVec
	EndpointsDesired                   *metrics.GaugeVec
	NumEndpointSlices                  *metrics.GaugeVec
	DesiredEndpointSlices              *metrics.GaugeVec
	EndpointSliceChanges               *metrics.CounterVec
	EndpointSlicesChangedPerSync       *metrics.HistogramVec
	EndpointSliceSyncs                 *metrics.CounterVec
	ServicesCountByTrafficDistribution *metrics.GaugeVec
}

// NewEndpointSliceMetrics instantiate the metrics and registers them to the legacyregistry.
// If already registered, the existing instance for the subsystem will be returned.
func NewEndpointSliceMetrics(subsystem string) *EndpointSliceMetrics {
	esm := &EndpointSliceMetrics{}

	esm.EndpointsAddedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      subsystem,
			Name:           "endpoints_added_per_sync",
			Help:           "Number of endpoints added on each Service sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)

	esm.EndpointsRemovedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      subsystem,
			Name:           "endpoints_removed_per_sync",
			Help:           "Number of endpoints removed on each Service sync",
			StabilityLevel: metrics.ALPHA,
			Buckets:        metrics.ExponentialBuckets(2, 2, 15),
		},
		[]string{},
	)

	esm.EndpointsDesired = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "endpoints_desired",
			Help:           "Number of endpoints desired",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)

	esm.NumEndpointSlices = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "num_endpoint_slices",
			Help:           "Number of EndpointSlices",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)

	esm.DesiredEndpointSlices = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "desired_endpoint_slices",
			Help:           "Number of EndpointSlices that would exist with perfect endpoint allocation",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{},
	)

	esm.EndpointSliceChanges = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "changes",
			Help:           "Number of EndpointSlice changes",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"operation"},
	)

	esm.EndpointSlicesChangedPerSync = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem: subsystem,
			Name:      "endpointslices_changed_per_sync",
			Help:      "Number of EndpointSlices changed on each Service sync",
		},
		[]string{
			"topology",             // either "Auto" or "Disabled"
			"traffic_distribution", // "PreferClose" or <empty>
		},
	)

	esm.EndpointSliceSyncs = metrics.NewCounterVec(
		&metrics.CounterOpts{
			Subsystem:      subsystem,
			Name:           "syncs",
			Help:           "Number of EndpointSlice syncs",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result"}, // either "success", "stale", or "error"
	)

	esm.ServicesCountByTrafficDistribution = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      subsystem,
			Name:           "services_count_by_traffic_distribution",
			Help:           "Number of Services using some specific trafficDistribution",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"traffic_distribution"}, // One of ["PreferClose", "ImplementationSpecific"]
	)

	esmStored, loaded := endpointSliceMetricsMap.LoadOrStore(subsystem, esm)
	if loaded {
		return esmStored.(*EndpointSliceMetrics)
	}

	esm.registerMetrics()

	return esm
}

// Reset resets all metrics.
func (esm *EndpointSliceMetrics) Reset() {
	esm.EndpointsAddedPerSync.Reset()
	esm.EndpointsRemovedPerSync.Reset()
	esm.EndpointsDesired.Reset()
	esm.NumEndpointSlices.Reset()
	esm.DesiredEndpointSlices.Reset()
	esm.EndpointSliceChanges.Reset()
	esm.EndpointSlicesChangedPerSync.Reset()
	esm.EndpointSliceSyncs.Reset()
	esm.ServicesCountByTrafficDistribution.Reset()
}

// registerMetrics registers EndpointSlice metrics.
func (esm *EndpointSliceMetrics) registerMetrics() {
	legacyregistry.MustRegister(esm.EndpointsAddedPerSync)
	legacyregistry.MustRegister(esm.EndpointsRemovedPerSync)
	legacyregistry.MustRegister(esm.EndpointsDesired)
	legacyregistry.MustRegister(esm.NumEndpointSlices)
	legacyregistry.MustRegister(esm.DesiredEndpointSlices)
	legacyregistry.MustRegister(esm.EndpointSliceChanges)
	legacyregistry.MustRegister(esm.EndpointSlicesChangedPerSync)
	legacyregistry.MustRegister(esm.EndpointSliceSyncs)
	legacyregistry.MustRegister(esm.ServicesCountByTrafficDistribution)
}
