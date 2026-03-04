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
