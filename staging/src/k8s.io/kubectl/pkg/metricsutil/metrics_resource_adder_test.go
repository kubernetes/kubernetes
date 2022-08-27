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

package metricsutil

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/metrics/pkg/apis/metrics"
)

func getResourceQuantity(t *testing.T, quantityStr string) resource.Quantity {
	t.Helper()
	var err error
	quantity, err := resource.ParseQuantity("0")
	if err != nil {
		t.Errorf("failed when parsing 0 into resource.Quantity")
	}
	if quantityStr != "" {
		quantity, err = resource.ParseQuantity(quantityStr)
		if err != nil {
			t.Errorf("%s is not a valid resource value", quantityStr)
		}
	}
	return quantity
}

func addContainerMetricsToPodMetrics(t *testing.T, podMetrics *metrics.PodMetrics, cpuUsage, memUsage string) {
	t.Helper()
	containerMetrics := metrics.ContainerMetrics{
		Usage: corev1.ResourceList{},
	}

	containerMetrics.Usage["cpu"] = getResourceQuantity(t, cpuUsage)
	containerMetrics.Usage["memory"] = getResourceQuantity(t, memUsage)

	podMetrics.Containers = append(podMetrics.Containers, containerMetrics)
}

func addContainerResourcesToPodResources(t *testing.T, pods *corev1.Pod, cpuReq, cpuLim, memReq, memLim string) {
	t.Helper()
	container := corev1.Container{
		Resources: corev1.ResourceRequirements{
			Limits:   corev1.ResourceList{},
			Requests: corev1.ResourceList{},
		},
	}

	container.Resources.Requests[corev1.ResourceCPU] = getResourceQuantity(t, cpuReq)
	container.Resources.Limits[corev1.ResourceCPU] = getResourceQuantity(t, cpuLim)
	container.Resources.Requests[corev1.ResourceMemory] = getResourceQuantity(t, memReq)
	container.Resources.Limits[corev1.ResourceMemory] = getResourceQuantity(t, memLim)

	pods.Spec.Containers = append(pods.Spec.Containers, container)
}

func initResourceAdder() *ResourceAdder {
	resources := []corev1.ResourceName{
		corev1.ResourceCPU,
		corev1.ResourceMemory,
		corev1.ResourceLimitsCPU,
		corev1.ResourceRequestsCPU,
		corev1.ResourceLimitsMemory,
		corev1.ResourceRequestsMemory,
	}
	return NewResourceAdder(resources)
}

func TestAddPodMetrics(t *testing.T) {
	resourceAdder := initResourceAdder()

	tests := []struct {
		name             string
		cpuUsage         string
		memUsage         string
		expectedCpuUsage resource.Quantity
		expectedMemUsage resource.Quantity
	}{
		{
			name:             "initial value",
			cpuUsage:         "0",
			memUsage:         "0",
			expectedCpuUsage: getResourceQuantity(t, "0"),
			expectedMemUsage: getResourceQuantity(t, "0"),
		},
		{
			name:             "add first container metric",
			cpuUsage:         "1m",
			memUsage:         "10Mi",
			expectedCpuUsage: getResourceQuantity(t, "1m"),
			expectedMemUsage: getResourceQuantity(t, "10Mi"),
		},
		{
			name:             "add second container metric",
			cpuUsage:         "5m",
			memUsage:         "25Mi",
			expectedCpuUsage: getResourceQuantity(t, "6m"),
			expectedMemUsage: getResourceQuantity(t, "35Mi"),
		},
		{
			name:             "add third container zero metric",
			cpuUsage:         "0m",
			memUsage:         "0Mi",
			expectedCpuUsage: getResourceQuantity(t, "6m"),
			expectedMemUsage: getResourceQuantity(t, "35Mi"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podMetrics := metrics.PodMetrics{}
			addContainerMetricsToPodMetrics(t, &podMetrics, test.cpuUsage, test.memUsage)

			resourceAdder.AddPodMetrics(&podMetrics)
			cpuUsage := resourceAdder.total["cpu"]
			memUsage := resourceAdder.total["memory"]

			if !test.expectedCpuUsage.Equal(cpuUsage) {
				t.Errorf("expecting cpu usage %s but getting %s", test.expectedCpuUsage.String(), cpuUsage.String())
			}
			if !test.expectedMemUsage.Equal(memUsage) {
				t.Errorf("expecting memeory usage %s but getting %s", test.expectedMemUsage.String(), memUsage.String())
			}
		})
	}
}

func TestAddPodMetricsWithResources(t *testing.T) {
	resourceAdder := initResourceAdder()

	tests := []struct {
		name               string
		cpuUsage           string
		cpuRequest         string
		cpuLimit           string
		memUsage           string
		memRequest         string
		memLimit           string
		expectedCpuUsage   resource.Quantity
		expectedCpuRequest resource.Quantity
		expectedCpuLimit   resource.Quantity
		expectedMemUsage   resource.Quantity
		expectedMemRequest resource.Quantity
		expectedMemLimit   resource.Quantity
	}{
		{
			name:               "initial value",
			cpuUsage:           "0",
			cpuRequest:         "0",
			cpuLimit:           "0",
			memUsage:           "0",
			memRequest:         "0",
			memLimit:           "0",
			expectedCpuUsage:   getResourceQuantity(t, "0"),
			expectedCpuRequest: getResourceQuantity(t, "0"),
			expectedCpuLimit:   getResourceQuantity(t, "0"),
			expectedMemUsage:   getResourceQuantity(t, "0"),
			expectedMemRequest: getResourceQuantity(t, "0"),
			expectedMemLimit:   getResourceQuantity(t, "0"),
		},
		{
			name:               "add first container metric",
			cpuUsage:           "1m",
			cpuRequest:         "1m",
			cpuLimit:           "2m",
			memUsage:           "10Mi",
			memRequest:         "10Mi",
			memLimit:           "20Mi",
			expectedCpuUsage:   getResourceQuantity(t, "1m"),
			expectedCpuRequest: getResourceQuantity(t, "1m"),
			expectedCpuLimit:   getResourceQuantity(t, "2m"),
			expectedMemUsage:   getResourceQuantity(t, "10Mi"),
			expectedMemRequest: getResourceQuantity(t, "10Mi"),
			expectedMemLimit:   getResourceQuantity(t, "20Mi"),
		},
		{
			name:               "add second container metric",
			cpuUsage:           "5m",
			cpuRequest:         "1m",
			cpuLimit:           "10m",
			memUsage:           "25Mi",
			memRequest:         "10Mi",
			memLimit:           "30Mi",
			expectedCpuUsage:   getResourceQuantity(t, "6m"),
			expectedCpuRequest: getResourceQuantity(t, "2m"),
			expectedCpuLimit:   getResourceQuantity(t, "12m"),
			expectedMemUsage:   getResourceQuantity(t, "35Mi"),
			expectedMemRequest: getResourceQuantity(t, "20Mi"),
			expectedMemLimit:   getResourceQuantity(t, "50Mi"),
		},
		{
			name:               "add third container zero metric",
			cpuUsage:           "0m",
			cpuRequest:         "0m",
			cpuLimit:           "0m",
			memUsage:           "0Mi",
			memRequest:         "0Mi",
			memLimit:           "0Mi",
			expectedCpuUsage:   getResourceQuantity(t, "6m"),
			expectedCpuRequest: getResourceQuantity(t, "2m"),
			expectedCpuLimit:   getResourceQuantity(t, "12m"),
			expectedMemUsage:   getResourceQuantity(t, "35Mi"),
			expectedMemRequest: getResourceQuantity(t, "20Mi"),
			expectedMemLimit:   getResourceQuantity(t, "50Mi"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := corev1.Pod{}
			podMetrics := metrics.PodMetrics{}
			addContainerMetricsToPodMetrics(t, &podMetrics, test.cpuUsage, test.memUsage)
			addContainerResourcesToPodResources(t, &pod, test.cpuRequest, test.cpuLimit, test.memRequest, test.memLimit)

			containerCache := make(map[string]*ResourceContainerInfo)

			for _, c := range pod.Spec.Containers {
				containerCache[c.Name] = &ResourceContainerInfo{
					Name:      c.Name,
					Container: &c,
				}
			}

			resourceAdder.AddPodMetricsWithResources(&podMetrics, containerCache)
			cpuUsage := resourceAdder.total["cpu"]
			cpuReq := resourceAdder.total[corev1.ResourceRequestsCPU]
			cpuLim := resourceAdder.total[corev1.ResourceLimitsCPU]
			memUsage := resourceAdder.total["memory"]
			memReq := resourceAdder.total[corev1.ResourceRequestsMemory]
			memLim := resourceAdder.total[corev1.ResourceLimitsMemory]

			if !test.expectedCpuUsage.Equal(cpuUsage) {
				t.Errorf("expecting cpu usage %s but getting %s", test.expectedCpuUsage.String(), cpuUsage.String())
			}
			if !test.expectedCpuRequest.Equal(cpuReq) {
				t.Errorf("expecting cpu request %s but getting %s", test.expectedCpuRequest.String(), cpuReq.String())
			}
			if !test.expectedCpuLimit.Equal(cpuLim) {
				t.Errorf("expecting cpu limit %s but getting %s", test.expectedCpuLimit.String(), cpuLim.String())
			}
			if !test.expectedMemUsage.Equal(memUsage) {
				t.Errorf("expecting memory usage %s but getting %s", test.expectedMemUsage.String(), memUsage.String())
			}
			if !test.expectedMemRequest.Equal(memReq) {
				t.Errorf("expecting memory request %s but getting %s", test.expectedMemRequest.String(), memReq.String())
			}
			if !test.expectedMemLimit.Equal(memLim) {
				t.Errorf("expecting memory limit %s but getting %s", test.expectedMemLimit.String(), memLim.String())
			}
		})
	}
}
