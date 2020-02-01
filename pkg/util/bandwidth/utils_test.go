/*
Copyright 2015 The Kubernetes Authors.

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

package bandwidth

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestExtractPodBandwidthResources(t *testing.T) {
	four, _ := resource.ParseQuantity("4M")
	ten, _ := resource.ParseQuantity("10M")
	twenty, _ := resource.ParseQuantity("20M")
	fourG, _ := resource.ParseQuantity("4G")

	testPod := func(ingressRate, ingressBurst, egressRate, egressBurst string) *api.Pod {
		pod := &api.Pod{ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}}}
		if len(ingressRate) != 0 {
			pod.Annotations["kubernetes.io/ingress-bandwidth"] = ingressRate
		}
		if len(ingressBurst) != 0 {
			pod.Annotations["kubernetes.io/ingress-burst"] = ingressBurst
		}
		if len(egressRate) != 0 {
			pod.Annotations["kubernetes.io/egress-bandwidth"] = egressRate
		}
		if len(egressBurst) != 0 {
			pod.Annotations["kubernetes.io/egress-burst"] = egressBurst
		}
		return pod
	}

	tests := []struct {
		pod             *api.Pod
		expectedIngress *Limit
		expectedEgress  *Limit
		expectError     bool
	}{
		{
			pod: &api.Pod{},
		},
		{
			pod: testPod("10M", "20M", "10M", "4G"),
			expectedIngress: &Limit{
				Rate:  &ten,
				Burst: &twenty,
			},
			expectedEgress: &Limit{
				Rate:  &ten,
				Burst: &fourG,
			},
		},
		{
			pod: testPod("10M", "10M", "10M", ""),
			expectedIngress: &Limit{
				Rate:  &ten,
				Burst: &ten,
			},
			expectedEgress: &Limit{
				Rate:  &ten,
				Burst: nil,
			},
		},
		{
			pod: testPod("4M", "", "10M", "4M"),
			expectedIngress: &Limit{
				Rate:  &four,
				Burst: nil,
			},
			expectedEgress: &Limit{
				Rate:  &ten,
				Burst: &four,
			},
		},
		// lower than min bandwidth should error
		{
			pod:         testPod("0.5b", "10M", "10M", "10M"),
			expectError: true,
		},
		// greater than max bandwidth should error
		{
			pod:         testPod("2P", "4M", "20M", "10M"),
			expectError: true,
		},
		// lower than min burst should error
		{
			pod:         testPod("10M", "0.5b", "10M", "4M"),
			expectError: true,
		},
		// greater than max burst should error
		{
			pod:         testPod("10M", "5G", "10M", "4M"),
			expectError: true,
		},
		// only burst is set should error
		{
			pod:         testPod("", "4G", "", ""),
			expectError: true,
		},
		// bad bandwidth and/or burst should error
		{
			pod:         testPod("foo", "", "", ""),
			expectError: true,
		},
		{
			pod:         testPod("4M", "foo", "", ""),
			expectError: true,
		},
		{
			pod:         testPod("", "", "foo", ""),
			expectError: true,
		},
		{
			pod:         testPod("", "", "4M", "bar"),
			expectError: true,
		},
	}
	for _, test := range tests {
		ingress, egress, err := ExtractPodBandwidthResources(test.pod.Annotations)
		if test.expectError {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !reflect.DeepEqual(ingress, test.expectedIngress) {
			t.Errorf("expected: %v, saw: %v", ingress, test.expectedIngress)
		}
		if !reflect.DeepEqual(egress, test.expectedEgress) {
			t.Errorf("expected: %v, saw: %v", egress, test.expectedEgress)
		}
	}
}
