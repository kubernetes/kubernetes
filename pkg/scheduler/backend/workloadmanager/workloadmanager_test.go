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

package workloadmanager

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestWorkloadManager_AddPod(t *testing.T) {
	p1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()
	// Assigned
	p2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").Node("node1").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()
	// Different ns
	p3 := st.MakePod().Namespace("ns2").Name("p3").UID("p3").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()
	nonWorkloadPod := st.MakePod().Namespace("ns1").Name("non-workload").Obj()

	tests := []struct {
		name             string
		initPods         []*v1.Pod
		initPodsToAssume []*v1.Pod
		podToAdd         *v1.Pod

		expectedPodGroups       int
		expectInAllPods         bool
		expectInUnscheduledPods bool
		expectInAssumedPods     bool
		expectInAssignedPods    bool
	}{
		{
			name:                    "adding an unscheduled pod",
			podToAdd:                p1,
			expectedPodGroups:       1,
			expectInAllPods:         true,
			expectInUnscheduledPods: true,
		},
		{
			name:                 "adding an assigned pod",
			podToAdd:             p2,
			expectedPodGroups:    1,
			expectInAllPods:      true,
			expectInAssignedPods: true,
		},
		{
			name:                 "adding an assigned pod that was previously assumed",
			initPods:             []*v1.Pod{p2},
			initPodsToAssume:     []*v1.Pod{p2},
			podToAdd:             p2,
			expectedPodGroups:    1,
			expectInAllPods:      true,
			expectInAssignedPods: true,
		},
		{
			name:                    "adding pod with different namespace",
			initPods:                []*v1.Pod{p1},
			podToAdd:                p3,
			expectedPodGroups:       2,
			expectInAllPods:         true,
			expectInUnscheduledPods: true,
		},
		{
			name:              "adding a non-workload pod is a no-op",
			podToAdd:          nonWorkloadPod,
			expectedPodGroups: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := New()
			for _, p := range tt.initPods {
				manager.AddPod(p)
			}
			for _, p := range tt.initPodsToAssume {
				info, err := manager.PodGroupInfo(p.Namespace, p.Spec.Workload)
				if err != nil {
					t.Fatalf("Unexpected error getting pod group info: %v", err)
				}
				info.AssumePod(p.UID)
			}

			manager.AddPod(tt.podToAdd)

			gotPodGroups := len(manager.podGroupInfos)
			if gotPodGroups != tt.expectedPodGroups {
				t.Fatalf("Expected %v pod group(s), got %v", tt.expectedPodGroups, gotPodGroups)
			}
			if gotPodGroups == 0 {
				return
			}
			info, err := manager.PodGroupInfo(tt.podToAdd.Namespace, tt.podToAdd.Spec.Workload)
			if err != nil {
				t.Fatalf("Unexpected error getting pod group info: %v", err)
			}
			if inAll := info.AllPods().Has(tt.podToAdd.UID); inAll != tt.expectInAllPods {
				t.Errorf("Unexpected AllPods state, want: %v, got: %v", tt.expectInAllPods, inAll)
			}
			if inAssumed := info.AssumedPods().Has(tt.podToAdd.UID); inAssumed != tt.expectInAssumedPods {
				t.Errorf("Unexpected AssumedPods state, want: %v, got: %v", tt.expectInAssumedPods, inAssumed)
			}
			if inAssigned := info.AssignedPods().Has(tt.podToAdd.UID); inAssigned != tt.expectInAssignedPods {
				t.Errorf("Unexpected AssignedPods state, want: %v, got: %v", tt.expectInAssignedPods, inAssigned)
			}
		})
	}
}

func TestWorkloadManager_UpdatePod(t *testing.T) {
	pod := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()
	updatedPod := st.MakePod().Namespace("ns1").Name("p1").UID("p1").Labels(map[string]string{"foo": "bar"}).
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()

	assignedPod := st.MakePod().Namespace("ns1").Name("p2").UID("p2").Node("node1").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()
	updatedAssignedPod := st.MakePod().Namespace("ns1").Name("p2").UID("p2").Node("node1").Labels(map[string]string{"foo": "bar"}).
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()

	nonWorkloadPod := st.MakePod().Namespace("ns1").Name("non-workload").Obj()
	updatedNonWorkloadPod := st.MakePod().Namespace("ns1").Name("non-workload").Labels(map[string]string{"foo": "bar"}).Obj()

	tests := []struct {
		name      string
		assumePod bool
		oldPod    *v1.Pod
		newPod    *v1.Pod

		expectInAllPods         bool
		expectInUnscheduledPods bool
		expectInAssumedPods     bool
		expectInAssignedPods    bool
	}{
		{
			name:                    "updating an unscheduled pod",
			oldPod:                  pod,
			newPod:                  updatedPod,
			expectInAllPods:         true,
			expectInUnscheduledPods: true,
		},
		{
			name:                "updating an assumed pod",
			assumePod:           true,
			oldPod:              pod,
			newPod:              updatedPod,
			expectInAllPods:     true,
			expectInAssumedPods: true,
		},
		{
			name:                 "updating an assigned pod",
			oldPod:               assignedPod,
			newPod:               updatedAssignedPod,
			expectInAllPods:      true,
			expectInAssignedPods: true,
		},
		{
			name:                 "binding an unscheduled pod",
			oldPod:               pod,
			newPod:               assignedPod,
			expectInAllPods:      true,
			expectInAssignedPods: true,
		},
		{
			name:                 "binding an assumed pod",
			assumePod:            true,
			oldPod:               pod,
			newPod:               assignedPod,
			expectInAllPods:      true,
			expectInAssignedPods: true,
		},
		{
			name:   "updating a non-workload pod is a no-op",
			oldPod: nonWorkloadPod,
			newPod: updatedNonWorkloadPod,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := New()

			manager.AddPod(tt.oldPod)
			if tt.assumePod {
				info, err := manager.PodGroupInfo(tt.oldPod.Namespace, tt.oldPod.Spec.Workload)
				if err != nil {
					t.Fatalf("Unexpected error getting pod group info: %v", err)
				}
				info.AssumePod(tt.oldPod.UID)
			}

			manager.UpdatePod(tt.oldPod, tt.newPod)

			gotPodGroups := len(manager.podGroupInfos)
			if gotPodGroups == 0 {
				if tt.expectInAllPods {
					t.Fatalf("Expected pod group, but got none")
				}
				return
			}
			if !tt.expectInAllPods {
				t.Fatalf("Expected no pod groups, but got %v", gotPodGroups)
			}
			info, err := manager.PodGroupInfo(tt.newPod.Namespace, tt.newPod.Spec.Workload)
			if err != nil {
				t.Fatalf("Unexpected error getting pod group info: %v", err)
			}
			if inAll := info.AllPods().Has(tt.newPod.UID); inAll != tt.expectInAllPods {
				t.Errorf("Unexpected AllPods state, want: %v, got: %v", tt.expectInAllPods, inAll)
			}
			if inAssumed := info.AssumedPods().Has(tt.newPod.UID); inAssumed != tt.expectInAssumedPods {
				t.Errorf("Unexpected AssumedPods state, want: %v, got: %v", tt.expectInAssumedPods, inAssumed)
			}
			if inAssigned := info.AssignedPods().Has(tt.newPod.UID); inAssigned != tt.expectInAssignedPods {
				t.Errorf("Unexpected AssignedPods state, want: %v, got: %v", tt.expectInAssignedPods, inAssigned)
			}
		})
	}
}

func TestWorkloadManager_DeletePod(t *testing.T) {
	p1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()
	p2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").
		Workload(&v1.WorkloadReference{Name: "w1", PodGroup: "pg1"}).Obj()

	tests := []struct {
		name        string
		initPods    []*v1.Pod
		podToDelete *v1.Pod

		expectedPodGroups int
	}{
		{
			name:              "deleting a pod from a group with multiple pods",
			initPods:          []*v1.Pod{p1, p2},
			podToDelete:       p1,
			expectedPodGroups: 1,
		},
		{
			name:        "deleting the last pod cleans up the info",
			initPods:    []*v1.Pod{p1},
			podToDelete: p1,
		},
		{
			name:        "deleting a non-existent pod is a no-op",
			podToDelete: p1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager := New()
			for _, p := range tt.initPods {
				manager.AddPod(p)
			}
			manager.DeletePod(tt.podToDelete)

			gotPodGroups := len(manager.podGroupInfos)
			if gotPodGroups != tt.expectedPodGroups {
				t.Fatalf("Expected %v pod group(s), got %v", tt.expectedPodGroups, gotPodGroups)
			}
			if gotPodGroups == 0 {
				return
			}
			info, err := manager.PodGroupInfo(tt.podToDelete.Namespace, tt.podToDelete.Spec.Workload)
			if err != nil {
				t.Fatalf("Unexpected error getting pod group info: %v", err)
			}
			if len(info.AllPods()) == 0 {
				t.Errorf("Expected AllPods to be non-empty")
			}
			if info.AllPods().Has(p1.UID) {
				t.Errorf("Expected pod to be deleted")
			}
		})
	}
}
