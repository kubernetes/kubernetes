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
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/metrics/pkg/apis/metrics/v1beta1"
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

// TestFakeGet checks the fake Clientset is working properly for GET method.
func TestFakeGet(t *testing.T) {
	var objects []runtime.Object
	podMetricsFoo := &v1beta1.PodMetrics{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: metav1.NamespaceDefault,
		},
		Timestamp:  metav1.Time{Time: time.Now()},
		Window:     metav1.Duration{Duration: time.Second},
		Containers: []v1beta1.ContainerMetrics{},
	}
	nodeMetricsBar := &v1beta1.NodeMetrics{
		ObjectMeta: metav1.ObjectMeta{
			Name: "bar",
		},
		Timestamp: metav1.Time{Time: time.Now()},
		Window:    metav1.Duration{Duration: time.Second},
		Usage:     v1.ResourceList{},
	}
	objects = append(objects, podMetricsFoo)
	objects = append(objects, nodeMetricsBar)
	metricsClient := fake.NewSimpleClientset(objects...)
	if _, err := metricsClient.MetricsV1beta1().PodMetricses(metav1.NamespaceDefault).Get(context.TODO(), "foo", metav1.GetOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := metricsClient.MetricsV1beta1().NodeMetricses().Get(context.TODO(), "bar", metav1.GetOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}
