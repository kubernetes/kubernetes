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

package collectors

import (
	"context"
	"strings"
	"testing"

	"k8s.io/component-base/metrics/testutil"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
)

func TestCRIMetricsCollectorLabels(t *testing.T) {
	sandboxID := "sandbox-abc"
	containerID := "container-xyz"

	descriptors := []*runtimeapi.MetricDescriptor{
		{
			Name:      "container_cpu_usage_seconds_total",
			Help:      "CPU usage",
			LabelKeys: []string{"id", "name", "image"},
		},
	}

	podMetrics := []*runtimeapi.PodSandboxMetrics{
		{
			PodSandboxId: sandboxID,
			Metrics: []*runtimeapi.Metric{
				{
					Name:        "container_cpu_usage_seconds_total",
					MetricType:  runtimeapi.MetricType_COUNTER,
					LabelValues: []string{sandboxID, "POD", ""},
					Value:       &runtimeapi.UInt64Value{Value: 100},
				},
			},
			ContainerMetrics: []*runtimeapi.ContainerMetrics{
				{
					ContainerId: containerID,
					Metrics: []*runtimeapi.Metric{
						{
							Name:        "container_cpu_usage_seconds_total",
							MetricType:  runtimeapi.MetricType_COUNTER,
							LabelValues: []string{containerID, "k8s_nginx_my-pod_default_uid_0", "nginx:latest"},
							Value:       &runtimeapi.UInt64Value{Value: 200},
						},
					},
				},
			},
		},
	}

	podSandboxes := []*runtimeapi.PodSandbox{
		{
			Id: sandboxID,
			Metadata: &runtimeapi.PodSandboxMetadata{
				Name:      "my-pod",
				Namespace: "default",
				Uid:       "uid-123",
			},
		},
	}

	containers := []*runtimeapi.Container{
		{
			Id:           containerID,
			PodSandboxId: sandboxID,
			Metadata: &runtimeapi.ContainerMetadata{
				Name: "nginx",
			},
		},
	}

	collector := NewCRIMetricsCollector(
		context.Background(),
		func(ctx context.Context) ([]*runtimeapi.PodSandboxMetrics, error) {
			return podMetrics, nil
		},
		func(ctx context.Context) ([]*runtimeapi.MetricDescriptor, error) {
			return descriptors, nil
		},
		func(ctx context.Context, filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
			return podSandboxes, nil
		},
		func(ctx context.Context, filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
			return containers, nil
		},
	)

	err := testutil.CustomCollectAndCompare(collector, strings.NewReader(`
		# HELP container_cpu_usage_seconds_total [INTERNAL] CPU usage
		# TYPE container_cpu_usage_seconds_total counter
		container_cpu_usage_seconds_total{container="POD", id="sandbox-abc", image="", name="POD", namespace="default", pod="my-pod"} 100
		container_cpu_usage_seconds_total{container="nginx", id="container-xyz", image="nginx:latest", name="k8s_nginx_my-pod_default_uid_0", namespace="default", pod="my-pod"} 200
	`), "container_cpu_usage_seconds_total")
	if err != nil {
		t.Fatalf("unexpected metrics:\n%s", err)
	}
}

func TestCRIMetricsCollectorMissingMapping(t *testing.T) {
	descriptors := []*runtimeapi.MetricDescriptor{
		{
			Name:      "container_cpu_usage_seconds_total",
			Help:      "CPU usage",
			LabelKeys: []string{"id", "name", "image"},
		},
	}

	podMetrics := []*runtimeapi.PodSandboxMetrics{
		{
			PodSandboxId: "unknown-sandbox",
			ContainerMetrics: []*runtimeapi.ContainerMetrics{
				{
					ContainerId: "unknown-container",
					Metrics: []*runtimeapi.Metric{
						{
							Name:        "container_cpu_usage_seconds_total",
							MetricType:  runtimeapi.MetricType_COUNTER,
							LabelValues: []string{"unknown-container", "some-name", "some-image"},
							Value:       &runtimeapi.UInt64Value{Value: 1024},
						},
					},
				},
			},
		},
	}

	collector := NewCRIMetricsCollector(
		context.Background(),
		func(ctx context.Context) ([]*runtimeapi.PodSandboxMetrics, error) {
			return podMetrics, nil
		},
		func(ctx context.Context) ([]*runtimeapi.MetricDescriptor, error) {
			return descriptors, nil
		},
		func(ctx context.Context, filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
			return nil, nil
		},
		func(ctx context.Context, filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
			return nil, nil
		},
	)

	err := testutil.CustomCollectAndCompare(collector, strings.NewReader(`
		# HELP container_cpu_usage_seconds_total [INTERNAL] CPU usage
		# TYPE container_cpu_usage_seconds_total counter
		container_cpu_usage_seconds_total{container="", id="unknown-container", image="some-image", name="some-name", namespace="", pod=""} 1024
	`), "container_cpu_usage_seconds_total")
	if err != nil {
		t.Fatalf("unexpected metrics:\n%s", err)
	}
}
