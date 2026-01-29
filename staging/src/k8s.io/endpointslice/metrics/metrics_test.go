/*
Copyright 2025 The Kubernetes Authors.

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
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestEndpointSliceMetrics(t *testing.T) {
	// Reset metrics to ensure clean state
	EndpointsAddedPerSync.Reset()
	EndpointsRemovedPerSync.Reset()
	EndpointsDesired.Reset()
	NumEndpointSlices.Reset()
	DesiredEndpointSlices.Reset()
	ServicesCountByTrafficDistribution.Reset()

	// Register metrics
	RegisterMetrics()

	// Test EndpointsAddedPerSync
	EndpointsAddedPerSync.WithLabelValues().Observe(5.0)
	wantAdded := `
		# HELP endpoint_slice_controller_endpoints_added_per_sync [BETA] Number of endpoints added on each Service sync
		# TYPE endpoint_slice_controller_endpoints_added_per_sync histogram
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="2"} 0
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="4"} 0
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="8"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="16"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="32"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="64"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="128"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="256"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="512"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="1024"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="2048"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="4096"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="8192"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="16384"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="32768"} 1
		endpoint_slice_controller_endpoints_added_per_sync_bucket{le="+Inf"} 1
		endpoint_slice_controller_endpoints_added_per_sync_sum 5
		endpoint_slice_controller_endpoints_added_per_sync_count 1
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantAdded), "endpoint_slice_controller_endpoints_added_per_sync"); err != nil {
		t.Fatal(err)
	}

	// Test EndpointsRemovedPerSync
	EndpointsRemovedPerSync.WithLabelValues().Observe(3.0)
	wantRemoved := `
		# HELP endpoint_slice_controller_endpoints_removed_per_sync [BETA] Number of endpoints removed on each Service sync
		# TYPE endpoint_slice_controller_endpoints_removed_per_sync histogram
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="2"} 0
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="4"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="8"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="16"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="32"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="64"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="128"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="256"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="512"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="1024"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="2048"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="4096"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="8192"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="16384"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="32768"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_bucket{le="+Inf"} 1
		endpoint_slice_controller_endpoints_removed_per_sync_sum 3
		endpoint_slice_controller_endpoints_removed_per_sync_count 1
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantRemoved), "endpoint_slice_controller_endpoints_removed_per_sync"); err != nil {
		t.Fatal(err)
	}

	// Test EndpointsDesired
	EndpointsDesired.WithLabelValues().Set(10.0)
	wantDesired := `
		# HELP endpoint_slice_controller_endpoints_desired [BETA] Number of endpoints desired
		# TYPE endpoint_slice_controller_endpoints_desired gauge
		endpoint_slice_controller_endpoints_desired 10
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantDesired), "endpoint_slice_controller_endpoints_desired"); err != nil {
		t.Fatal(err)
	}

	// Test NumEndpointSlices
	NumEndpointSlices.WithLabelValues().Set(5.0)
	wantNumSlices := `
		# HELP endpoint_slice_controller_num_endpoint_slices [BETA] Number of EndpointSlices
		# TYPE endpoint_slice_controller_num_endpoint_slices gauge
		endpoint_slice_controller_num_endpoint_slices 5
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantNumSlices), "endpoint_slice_controller_num_endpoint_slices"); err != nil {
		t.Fatal(err)
	}

	// Test DesiredEndpointSlices
	DesiredEndpointSlices.WithLabelValues().Set(3.0)
	wantDesiredSlices := `
		# HELP endpoint_slice_controller_desired_endpoint_slices [BETA] Number of EndpointSlices that would exist with perfect endpoint allocation
		# TYPE endpoint_slice_controller_desired_endpoint_slices gauge
		endpoint_slice_controller_desired_endpoint_slices 3
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantDesiredSlices), "endpoint_slice_controller_desired_endpoint_slices"); err != nil {
		t.Fatal(err)
	}

	// Test ServicesCountByTrafficDistribution
	ServicesCountByTrafficDistribution.WithLabelValues("PreferClose").Set(2.0)
	wantServicesCount := `
		# HELP endpoint_slice_controller_services_count_by_traffic_distribution [BETA] Number of Services using some specific trafficDistribution
		# TYPE endpoint_slice_controller_services_count_by_traffic_distribution gauge
		endpoint_slice_controller_services_count_by_traffic_distribution{traffic_distribution="PreferClose"} 2
	`
	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantServicesCount), "endpoint_slice_controller_services_count_by_traffic_distribution"); err != nil {
		t.Fatal(err)
	}
}
