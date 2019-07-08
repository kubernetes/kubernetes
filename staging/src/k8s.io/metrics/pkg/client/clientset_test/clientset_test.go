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

package clientset_test

import (
	"k8s.io/metrics/pkg/apis/metrics/v1alpha1"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/metrics/pkg/apis/metrics/v1beta1"
	"k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

// TestFakeList is a basic sanity check that makes sure the fake Clientset is working properly.
func TestFakeList(t *testing.T) {
	client := fake.NewSimpleClientset()
	if _, err := client.MetricsV1alpha1().PodMetricses("").List(metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1alpha1().NodeMetricses().List(metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1beta1().PodMetricses("").List(metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1beta1().NodeMetricses().List(metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestV1alpha1PodMetricsesList(t *testing.T) {
	expected := v1alpha1.PodMetricsList{
		Items: []v1alpha1.PodMetrics{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-1",
				},
				Containers: []v1alpha1.ContainerMetrics{
					{
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("1"),
							corev1.ResourceMemory: resource.MustParse("1G"),
						},
					},
					{
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("2G"),
						},
					},
				},
			},
		},
	}

	client := fake.NewSimpleClientset(&expected)

	actual, err := client.MetricsV1alpha1().PodMetricses("").List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if expected.String() != actual.String() {
		t.Errorf("expected PodMetricsList to be\n%s\nbut was:\n%s\n", expected.String(), actual.String())
	}
}

func TestV1beta1PodMetricsesList(t *testing.T) {
	expected := v1beta1.PodMetricsList{
		Items: []v1beta1.PodMetrics{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-1",
				},
				Containers: []v1beta1.ContainerMetrics{
					{
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("1"),
							corev1.ResourceMemory: resource.MustParse("1G"),
						},
					},
					{
						Usage: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("2G"),
						},
					},
				},
			},
		},
	}
	client := fake.NewSimpleClientset(&expected)

	actual, err := client.MetricsV1beta1().PodMetricses("").List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if expected.String() != actual.String() {
		t.Errorf("expected PodMetricsList to be\n%s\nbut was:\n%s\n", expected.String(), actual.String())
	}
}

func TestV1alpha1NodeMetricsesList(t *testing.T) {
	expected := v1alpha1.NodeMetricsList{
        Items: []v1alpha1.NodeMetrics{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-1",
				},
				Usage: corev1.ResourceList{
					corev1.ResourceCPU: resource.MustParse("1"),
					corev1.ResourceMemory: resource.MustParse("1G"),
				},
			},
        	{
        		ObjectMeta: metav1.ObjectMeta{
        			Name: "node-2",
				},
				Usage: corev1.ResourceList{
					corev1.ResourceCPU: resource.MustParse("2"),
					corev1.ResourceMemory: resource.MustParse("2G"),
				},
			},
		},
	}

	client := fake.NewSimpleClientset(&expected)

	actual, err := client.MetricsV1alpha1().NodeMetricses().List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if expected.String() != actual.String() {
		t.Errorf("expected NodeMetricsList to be\n%s\nbut was:\n%s\n", expected.String(), actual.String())
	}
}

func TestV1beta1NodeMetricsesList(t *testing.T) {
	expected := v1beta1.NodeMetricsList{
		Items: []v1beta1.NodeMetrics{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-1",
				},
				Usage: corev1.ResourceList{
					corev1.ResourceCPU: resource.MustParse("1"),
					corev1.ResourceMemory: resource.MustParse("1G"),
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-2",
				},
				Usage: corev1.ResourceList{
					corev1.ResourceCPU: resource.MustParse("2"),
					corev1.ResourceMemory: resource.MustParse("2G"),
				},
			},
		},
	}

	client := fake.NewSimpleClientset(&expected)

	actual, err := client.MetricsV1beta1().NodeMetricses().List(metav1.ListOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if expected.String() != actual.String() {
		t.Errorf("expected NodeMetricsList to be\n%s\nbut was:\n%s\n", expected.String(), actual.String())
	}
}
