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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/metrics/pkg/apis/metrics/v1beta1"
	"k8s.io/metrics/pkg/client/clientset/versioned/fake"
)

// TestFakeList is a basic sanity check that makes sure the fake Clientset is working properly.
func TestFakeList(t *testing.T) {
	nodename := "nodename"
	nodeMetrics := &v1beta1.NodeMetrics{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodename,
		},
	}
	podname := "podname"
	podMetrics := &v1beta1.PodMetrics{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podname,
			Namespace: "default",
		},
	}

	client := fake.NewSimpleClientset(nodeMetrics, podMetrics)
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
	if _, err := client.MetricsV1beta1().NodeMetricses().Get(context.TODO(), nodename, metav1.GetOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := client.MetricsV1beta1().PodMetricses(podMetrics.Namespace).Get(context.TODO(), podname, metav1.GetOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
}
