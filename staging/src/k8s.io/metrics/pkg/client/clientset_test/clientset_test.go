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
	"context"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/metrics/pkg/apis/metrics/v1alpha1"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

// TestFakeList is a basic sanity check that makes sure the fake Clientset is working properly.
func TestFakeList(t *testing.T) {
	client := fake.NewSimpleClientset()
	if _, err := client.MetricsV1alpha1().PodMetricses("").List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1alpha1().NodeMetricses().List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1beta1().PodMetricses("").List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1beta1().NodeMetricses().List(context.TODO(), metav1.ListOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}
func TestFakeMetrics(t *testing.T) {
	var nodeMetrics []*v1alpha1.NodeMetrics
	var podMetrics []*v1alpha1.PodMetrics
	nodeMetrics = []*v1alpha1.NodeMetrics{
		BuildTestNodeMetrics("n1", 8000, 8000, nil),
		BuildTestNodeMetrics("n2", 2000, 2000, nil),
	}
	podMetrics = []*v1alpha1.PodMetrics{
		BuildTestPodMetrics("p1", 2000, 2000, nil),
		BuildTestPodMetrics("p2", 2000, 2000, nil),
		BuildTestPodMetrics("p3", 2000, 2000, nil),
		BuildTestPodMetrics("p4", 2000, 2000, nil),
		BuildTestPodMetrics("p5", 2000, 2000, nil),
	}
	var metricsObjs []runtime.Object
	for _, metrics := range nodeMetrics {
		metricsObjs = append(metricsObjs, metrics)
	}
	for _, metrics := range podMetrics {
		metricsObjs = append(metricsObjs, metrics)
	}
	client := fake.NewSimpleClientset(metricsObjs...)
	pm, err := client.MetricsV1alpha1().PodMetricses("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	nm, err := client.MetricsV1alpha1().NodeMetricses().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(pm.Items) != len(podMetrics) {
		t.Errorf("list pod metrics number is not match,%v,%v", len(pm.Items), len(podMetrics))
	}
	if len(nm.Items) != len(nodeMetrics) {
		t.Errorf("list node metrics number is not match,%v,%v", len(nm.Items), len(nodeMetrics))
	}
}

func BuildTestNodeMetrics(name string, millicpu, mem int64, apply func(metrics *v1alpha1.NodeMetrics)) *v1alpha1.NodeMetrics {
	metrics := &v1alpha1.NodeMetrics{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Timestamp: metav1.Now(),
		Window:    metav1.Duration{Duration: time.Second * 10},
		Usage: v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(millicpu, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(mem, resource.DecimalSI),
		},
	}
	if apply != nil {
		apply(metrics)
	}
	return metrics
}
func BuildTestPodMetrics(name string, millicpu, mem int64, apply func(metrics *v1alpha1.PodMetrics)) *v1alpha1.PodMetrics {
	var metrics = &v1alpha1.PodMetrics{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Timestamp: metav1.Now(),
		Window:    metav1.Duration{Duration: time.Second * 10},
		Containers: []v1alpha1.ContainerMetrics{
			{
				Usage: v1.ResourceList{
					v1.ResourceCPU:    *resource.NewMilliQuantity(millicpu, resource.DecimalSI),
					v1.ResourceMemory: *resource.NewQuantity(mem, resource.DecimalSI),
				},
			},
		},
	}
	if apply != nil {
		apply(metrics)
	}
	return metrics
}
