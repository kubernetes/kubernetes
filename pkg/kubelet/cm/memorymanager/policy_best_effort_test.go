/*
Copyright 2025 The Kubernetes Authors.

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

package memorymanager

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

type testBestEffortPolicy struct {
	description    string
	systemReserved systemReservedMemory
}

func initBestEffortPolicyTests(t *testing.T, testCase *testBestEffortPolicy) (Policy, error) {
	manager := topologymanager.NewFakeManager()
	return NewPolicyBestEffort(nil, testCase.systemReserved, manager)
}

func TestBestEffortCanAllocateExclusively(t *testing.T) {
	testCases := []testBestEffortPolicy{
		{
			description: "should always return false",
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb, // random legit value to make initialization pass
				},
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, err := initBestEffortPolicyTests(t, &testCase)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if p.CanAllocateExclusively() {
				t.Errorf("memory manager best effort policy should never be able to allocate exclusively")
			}
		})
	}
}
