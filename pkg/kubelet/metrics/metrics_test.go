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
	"os"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/testutil"
)

const imagePullDurationKey = "kubelet_" + ImagePullDurationKey

func TestImagePullDurationMetric(t *testing.T) {
	t.Run("register image pull duration", func(t *testing.T) {
		Register()
		defer clearMetrics()

		// Pairs of image size in bytes and pull duration in seconds
		dataPoints := [][]float64{
			// 0 byets, 0 seconds
			{0, 0},
			// 5MB, 10 seconds
			{5 * 1024 * 1024, 10},
			// 15MB, 20 seconds
			{15 * 1024 * 1024, 20},
			// 500 MB, 200 seconds
			{500 * 1024 * 1024, 200},
			// 15 GB, 6000 seconds,
			{15 * 1024 * 1024 * 1024, 6000},
			// 200 GB, 10000 seconds
			{200 * 1024 * 1024 * 1024, 10000},
		}

		for _, dp := range dataPoints {
			imageSize := int64(dp[0])
			duration := dp[1]
			t.Log(imageSize, duration)
			t.Log(GetImageSizeBucket(uint64(imageSize)))
			ImagePullDuration.WithLabelValues(GetImageSizeBucket(uint64(imageSize))).Observe(duration)
		}

		wants, err := os.Open("testdata/image_pull_duration_metric")
		defer func() {
			if err := wants.Close(); err != nil {
				t.Error(err)
			}
		}()

		if err != nil {
			t.Fatal(err)
		}

		if err := testutil.GatherAndCompare(GetGather(), wants, imagePullDurationKey); err != nil {
			t.Error(err)
		}

	})
}

func clearMetrics() {
	ImagePullDuration.Reset()
}

func TestPrometheusCompliantCounterVecMetrics(t *testing.T) {
	Register()

	defer ContainerAlignedComputeResourcesFailureTotal.Reset()
	defer EventedPLEGConnErrTotal.Reset()
	defer EventedPLEGConnTotal.Reset()
	defer EvictionsTotal.Reset()
	defer PLEGDiscardEventsTotal.Reset()
	defer PodResourcesEndpointErrorsGetCountTotal.Reset()
	defer PodResourcesEndpointErrorsGetAllocatableCountTotal.Reset()
	defer PodResourcesEndpointErrorsListCountTotal.Reset()
	defer PodResourcesEndpointRequestsGetCountTotal.Reset()
	defer PodResourcesEndpointRequestsGetAllocatableCountTotal.Reset()
	defer PodResourcesEndpointRequestsListCountTotal.Reset()
	defer PreemptionsTotal.Reset()

	ContainerAlignedComputeResourcesFailureTotal.WithLabelValues(AlignScopeContainer, AlignedPhysicalCPU).Inc()
	EventedPLEGConnErrTotal.Inc()
	EventedPLEGConnTotal.Inc()
	EvictionsTotal.WithLabelValues("memory").Inc()
	PLEGDiscardEventsTotal.Inc()
	PodResourcesEndpointErrorsGetCountTotal.WithLabelValues("v1").Inc()
	PodResourcesEndpointErrorsGetAllocatableCountTotal.WithLabelValues("v1").Inc()
	PodResourcesEndpointErrorsListCountTotal.WithLabelValues("v1").Inc()
	PodResourcesEndpointRequestsGetCountTotal.WithLabelValues("v1").Inc()
	PodResourcesEndpointRequestsGetAllocatableCountTotal.WithLabelValues("v1").Inc()
	PodResourcesEndpointRequestsListCountTotal.WithLabelValues("v1").Inc()
	PreemptionsTotal.WithLabelValues("memory").Inc()

	expected := `# HELP kubelet_container_aligned_compute_resources_failure_total [ALPHA] Cumulative number of failures to allocate aligned compute resources to containers by alignment type.
# TYPE kubelet_container_aligned_compute_resources_failure_total counter
kubelet_container_aligned_compute_resources_failure_total{boundary="physical_cpu",scope="container"} 1
# HELP kubelet_evented_pleg_connection_error_total [ALPHA] The number of errors encountered during the establishment of streaming connection with the CRI runtime.
# TYPE kubelet_evented_pleg_connection_error_total counter
kubelet_evented_pleg_connection_error_total 1
# HELP kubelet_evented_pleg_connection_success_total [ALPHA] The number of times a streaming client was obtained to receive CRI Events.
# TYPE kubelet_evented_pleg_connection_success_total counter
kubelet_evented_pleg_connection_success_total 1
# HELP kubelet_evictions_total [ALPHA] Cumulative number of pod evictions by eviction signal
# TYPE kubelet_evictions_total counter
kubelet_evictions_total{eviction_signal="memory"} 1
# HELP kubelet_pleg_discard_events_total [ALPHA] The number of discard events in PLEG.
# TYPE kubelet_pleg_discard_events_total counter
kubelet_pleg_discard_events_total 1
# HELP kubelet_pod_resources_endpoint_errors_get_allocatable_total [ALPHA] Number of requests to the PodResource GetAllocatableResources endpoint which returned error. Broken down by server api version.
# TYPE kubelet_pod_resources_endpoint_errors_get_allocatable_total counter
kubelet_pod_resources_endpoint_errors_get_allocatable_total{server_api_version="v1"} 1
# HELP kubelet_pod_resources_endpoint_errors_get_total [ALPHA] Number of requests to the PodResource Get endpoint which returned error. Broken down by server api version.
# TYPE kubelet_pod_resources_endpoint_errors_get_total counter
kubelet_pod_resources_endpoint_errors_get_total{server_api_version="v1"} 1
# HELP kubelet_pod_resources_endpoint_errors_list_total [ALPHA] Number of requests to the PodResource List endpoint which returned error. Broken down by server api version.
# TYPE kubelet_pod_resources_endpoint_errors_list_total counter
kubelet_pod_resources_endpoint_errors_list_total{server_api_version="v1"} 1
# HELP kubelet_pod_resources_endpoint_requests_get_allocatable_total [ALPHA] Number of requests to the PodResource GetAllocatableResources endpoint. Broken down by server api version.
# TYPE kubelet_pod_resources_endpoint_requests_get_allocatable_total counter
kubelet_pod_resources_endpoint_requests_get_allocatable_total{server_api_version="v1"} 1
# HELP kubelet_pod_resources_endpoint_requests_get_total [ALPHA] Number of requests to the PodResource Get endpoint. Broken down by server api version.
# TYPE kubelet_pod_resources_endpoint_requests_get_total counter
kubelet_pod_resources_endpoint_requests_get_total{server_api_version="v1"} 1
# HELP kubelet_pod_resources_endpoint_requests_list_total [ALPHA] Number of requests to the PodResource List endpoint. Broken down by server api version.
# TYPE kubelet_pod_resources_endpoint_requests_list_total counter
kubelet_pod_resources_endpoint_requests_list_total{server_api_version="v1"} 1
# HELP kubelet_preemptions_total [ALPHA] Cumulative number of pod preemptions by preemption resource
# TYPE kubelet_preemptions_total counter
kubelet_preemptions_total{preemption_signal="memory"} 1
`

	metricNames := []string{
		"kubelet_container_aligned_compute_resources_failure_total",
		"kubelet_evented_pleg_connection_error_total",
		"kubelet_evented_pleg_connection_success_total",
		"kubelet_evictions_total",
		"kubelet_pleg_discard_events_total",
		"kubelet_pod_resources_endpoint_errors_get_total",
		"kubelet_pod_resources_endpoint_errors_get_allocatable_total",
		"kubelet_pod_resources_endpoint_errors_list_total",
		"kubelet_pod_resources_endpoint_requests_get_total",
		"kubelet_pod_resources_endpoint_requests_get_allocatable_total",
		"kubelet_pod_resources_endpoint_requests_list_total",
		"kubelet_preemptions_total",
	}

	if err := testutil.GatherAndCompare(
		GetGather(),
		strings.NewReader(expected),
		metricNames...,
	); err != nil {
		t.Fatal(err)
	}
}

func TestCPUManagerExclusiveCPUsAllocationMetric(t *testing.T) {
	Register()

	CPUManagerExclusiveCPUsAllocation.Set(2)

	expected := `# HELP kubelet_cpu_manager_exclusive_cpu_allocated [ALPHA] The total number of CPUs exclusively allocated to containers running on this node
# TYPE kubelet_cpu_manager_exclusive_cpu_allocated gauge
kubelet_cpu_manager_exclusive_cpu_allocated 2
`

	if err := testutil.GatherAndCompare(
		GetGather(),
		strings.NewReader(expected),
		"kubelet_cpu_manager_exclusive_cpu_allocated",
	); err != nil {
		t.Fatal(err)
	}
}

func TestPrometheusCompliantTopologyManagerAdmissionDurationMetric(t *testing.T) {
	Register()

	TopologyManagerAdmissionDurationSecond.Observe(0.1)

	expected := `# HELP kubelet_topology_manager_admission_duration_seconds [ALPHA] Duration in seconds to serve a pod admission request.
# TYPE kubelet_topology_manager_admission_duration_seconds histogram
kubelet_topology_manager_admission_duration_seconds_bucket{le="5e-05"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0001"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0002"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0004"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0008"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0016"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0032"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0064"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0128"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0256"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.0512"} 0
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.1024"} 1
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.2048"} 1
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.4096"} 1
kubelet_topology_manager_admission_duration_seconds_bucket{le="0.8192"} 1
kubelet_topology_manager_admission_duration_seconds_bucket{le="+Inf"} 1
kubelet_topology_manager_admission_duration_seconds_sum 0.1
kubelet_topology_manager_admission_duration_seconds_count 1
`

	if err := testutil.GatherAndCompare(
		GetGather(),
		strings.NewReader(expected),
		"kubelet_topology_manager_admission_duration_seconds",
	); err != nil {
		t.Fatal(err)
	}
}
