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

func initResourceAdder() *ResourceAdder {
	resources := []corev1.ResourceName{
		corev1.ResourceCPU,
		corev1.ResourceMemory,
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
