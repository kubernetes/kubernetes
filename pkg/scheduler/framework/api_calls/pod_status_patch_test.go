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

package apicalls

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
)

func TestPodStatusPatchCall_IsNoOp(t *testing.T) {
	podWithNode := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "uid",
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node-a",
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionFalse,
				},
			},
		},
	}

	tests := []struct {
		name           string
		pod            *v1.Pod
		conditions     []*v1.PodCondition
		nominatingInfo *fwk.NominatingInfo
		want           bool
	}{
		{
			name:           "No-op when condition and node name match",
			pod:            podWithNode,
			conditions:     []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionFalse}},
			nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeOverride},
			want:           true,
		},
		{
			name:           "Not no-op when condition is different",
			pod:            podWithNode,
			conditions:     []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionTrue}},
			nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeOverride},
			want:           false,
		},
		{
			name:           "Not no-op when nominated node name is different",
			pod:            podWithNode,
			conditions:     []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionFalse}},
			nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node-b", NominatingMode: fwk.ModeOverride},
			want:           false,
		},
		{
			name:           "No-op when condition is nil and node name matches",
			pod:            podWithNode,
			conditions:     nil,
			nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeOverride},
			want:           true,
		},
		{
			name:           "Not no-op when condition is nil but node name differs",
			pod:            podWithNode,
			conditions:     nil,
			nominatingInfo: &fwk.NominatingInfo{NominatedNodeName: "node-b", NominatingMode: fwk.ModeOverride},
			want:           false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			call := NewPodStatusPatchCall(test.pod, test.conditions, test.nominatingInfo)
			if got := call.IsNoOp(); got != test.want {
				t.Errorf("Expected IsNoOp() to return %v, but got %v", test.want, got)
			}
		})
	}
}

func TestPodStatusPatchCall_Merge(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "uid",
		},
	}

	t.Run("Merges nominating info and condition from the old call", func(t *testing.T) {
		oldCall := NewPodStatusPatchCall(pod, []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionFalse}},
			&fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeOverride},
		)
		newCall := NewPodStatusPatchCall(pod, nil, &fwk.NominatingInfo{NominatingMode: fwk.ModeNoop})

		if err := newCall.Merge(oldCall); err != nil {
			t.Fatalf("Unexpected error returned by Merge(): %v", err)
		}
		if newCall.nominatingInfo.NominatedNodeName != "node-a" {
			t.Errorf("Expected NominatedNodeName to be node-a, but got: %v", newCall.nominatingInfo.NominatedNodeName)
		}
		if len(newCall.newConditions) == 0 || newCall.newConditions[0].Type != v1.PodScheduled {
			t.Errorf("Expected PodScheduled condition, but got: %v", newCall.newConditions)
		}
	})

	t.Run("Doesn't overwrite nominating info and condition of a new call", func(t *testing.T) {
		oldCall := NewPodStatusPatchCall(pod, nil, &fwk.NominatingInfo{NominatingMode: fwk.ModeNoop})
		newCall := NewPodStatusPatchCall(pod, []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionFalse}},
			&fwk.NominatingInfo{NominatedNodeName: "node-b", NominatingMode: fwk.ModeOverride})

		if err := newCall.Merge(oldCall); err != nil {
			t.Fatalf("Unexpected error returned by Merge(): %v", err)
		}
		if newCall.nominatingInfo.NominatedNodeName != "node-b" {
			t.Errorf("Expected NominatedNodeName to be node-b, but got: %v", newCall.nominatingInfo.NominatedNodeName)
		}
		if len(newCall.newConditions) == 0 || newCall.newConditions[0].Type != v1.PodScheduled {
			t.Errorf("Expected PodScheduled condition, but got: %v", newCall.newConditions)
		}
	})

	t.Run("Overwrite nominating info and condition of a old call", func(t *testing.T) {
		oldCall := NewPodStatusPatchCall(pod, []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionFalse}}, nil)
		newCall := NewPodStatusPatchCall(pod, []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionTrue}, {Type: v1.PodInitialized, Status: v1.ConditionTrue}}, nil)

		if err := newCall.Merge(oldCall); err != nil {
			t.Fatalf("Unexpected error returned by Merge(): %v", err)
		}
		if newCall.newConditions[0].Type != v1.PodScheduled || newCall.newConditions[0].Status != v1.ConditionTrue {
			t.Errorf("Expected PodScheduled condition, but got: %v", newCall.newConditions)
		}
		if newCall.newConditions[1].Type != v1.PodInitialized || newCall.newConditions[1].Status != v1.ConditionTrue {
			t.Errorf("Expected PodScheduled condition, but got: %v", newCall.newConditions)
		}
	})
}

func TestPodStatusPatchCall_Sync(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "uid",
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node-a",
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodScheduled,
					Status: v1.ConditionFalse,
				},
			},
		},
	}

	t.Run("Syncs the status before execution and updates the pod", func(t *testing.T) {
		call := NewPodStatusPatchCall(pod, nil,
			&fwk.NominatingInfo{NominatedNodeName: "node-c", NominatingMode: fwk.ModeOverride})

		updatedPod := pod.DeepCopy()
		updatedPod.Status.NominatedNodeName = "node-b"

		syncedObj, err := call.Sync(updatedPod)
		if err != nil {
			t.Fatalf("Unexpected error returned by Sync(): %v", err)
		}
		if call.podStatus.NominatedNodeName != "node-b" {
			t.Errorf("Expected podStatus NominatedNodeName to be node-b, but got: %v", call.podStatus.NominatedNodeName)
		}
		syncedPod := syncedObj.(*v1.Pod)
		if syncedPod.Status.NominatedNodeName != "node-c" {
			t.Errorf("Expected synced pod's NominatedNodeName to be node-c, but got: %v", syncedPod.Status.NominatedNodeName)
		}
	})

	t.Run("Doesn't sync internal status during or after execution, but updates the pod", func(t *testing.T) {
		call := NewPodStatusPatchCall(pod, nil,
			&fwk.NominatingInfo{NominatedNodeName: "node-c", NominatingMode: fwk.ModeOverride})
		call.executed = true

		updatedPod := pod.DeepCopy()
		updatedPod.Status.NominatedNodeName = "node-b"

		syncedObj, err := call.Sync(updatedPod)
		if err != nil {
			t.Fatalf("Unexpected error returned by Sync(): %v", err)
		}
		if call.podStatus.NominatedNodeName != "node-a" {
			t.Errorf("Expected podStatus NominatedNodeName to be node-a, but got: %v", call.podStatus.NominatedNodeName)
		}
		syncedPod := syncedObj.(*v1.Pod)
		if syncedPod.Status.NominatedNodeName != "node-c" {
			t.Errorf("Expected synced pod's NominatedNodeName to be node-c, but got: %v", syncedPod.Status.NominatedNodeName)
		}
	})
}

func TestSyncStatus(t *testing.T) {
	tests := []struct {
		name                    string
		podStatusBefore         *v1.PodStatus
		conditions              []*v1.PodCondition
		nominatingInfo          *fwk.NominatingInfo
		expectedResult          bool
		expectedNominatedNode   string
		expectedConditionStatus map[v1.PodConditionType]v1.ConditionStatus
	}{
		{
			name: "No changes needed - empty conditions and same node name",
			podStatusBefore: &v1.PodStatus{
				NominatedNodeName: "node-a",
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
			conditions:              nil,
			nominatingInfo:          &fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeOverride},
			expectedResult:          false,
			expectedNominatedNode:   "node-a",                                                                       // should remain unchanged
			expectedConditionStatus: map[v1.PodConditionType]v1.ConditionStatus{v1.PodScheduled: v1.ConditionFalse}, // should remain unchanged
		},
		{
			name: "Condition needs update",
			podStatusBefore: &v1.PodStatus{
				NominatedNodeName: "node-a",
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
			conditions:              []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionTrue}},
			nominatingInfo:          &fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeNoop},
			expectedResult:          true,
			expectedNominatedNode:   "node-a",                                                                      // should remain unchanged
			expectedConditionStatus: map[v1.PodConditionType]v1.ConditionStatus{v1.PodScheduled: v1.ConditionTrue}, // should be updated
		},
		{
			name: "Nominated node name needs update",
			podStatusBefore: &v1.PodStatus{
				NominatedNodeName: "node-a",
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
			conditions:              nil,
			nominatingInfo:          &fwk.NominatingInfo{NominatedNodeName: "node-b", NominatingMode: fwk.ModeOverride},
			expectedResult:          true,
			expectedNominatedNode:   "node-b",                                                                       // should be updated
			expectedConditionStatus: map[v1.PodConditionType]v1.ConditionStatus{v1.PodScheduled: v1.ConditionFalse}, // should remain unchanged
		},
		{
			name: "Both condition and node name need update",
			podStatusBefore: &v1.PodStatus{
				NominatedNodeName: "node-a",
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
			conditions:              []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionTrue}},
			nominatingInfo:          &fwk.NominatingInfo{NominatedNodeName: "node-b", NominatingMode: fwk.ModeOverride},
			expectedResult:          true,
			expectedNominatedNode:   "node-b",                                                                      // should be updated
			expectedConditionStatus: map[v1.PodConditionType]v1.ConditionStatus{v1.PodScheduled: v1.ConditionTrue}, // should be updated
		},
		{
			name: "Multiple conditions need update",
			podStatusBefore: &v1.PodStatus{
				NominatedNodeName: "node-original",
			},
			conditions: []*v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.ContainersReady, Status: v1.ConditionFalse},
			},
			nominatingInfo:          &fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeNoop},
			expectedResult:          true,
			expectedNominatedNode:   "node-original",                                                                                                      // should remain unchanged because ModeNoop
			expectedConditionStatus: map[v1.PodConditionType]v1.ConditionStatus{v1.PodScheduled: v1.ConditionTrue, v1.ContainersReady: v1.ConditionFalse}, // should be updated
		},
		{
			name: "No-op nominating mode with different node name",
			podStatusBefore: &v1.PodStatus{
				NominatedNodeName: "node-a",
				Conditions: []v1.PodCondition{
					{
						Type:   v1.PodScheduled,
						Status: v1.ConditionFalse,
					},
				},
			},
			conditions:              nil,
			nominatingInfo:          &fwk.NominatingInfo{NominatedNodeName: "node-b", NominatingMode: fwk.ModeNoop},
			expectedResult:          false,
			expectedNominatedNode:   "node-a",                                                                       // should remain unchanged
			expectedConditionStatus: map[v1.PodConditionType]v1.ConditionStatus{v1.PodScheduled: v1.ConditionFalse}, // should remain unchanged
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			podStatus := test.podStatusBefore.DeepCopy()
			result := syncStatus(podStatus, test.conditions, test.nominatingInfo)
			if result != test.expectedResult {
				t.Errorf("Expected syncStatus to return %v, but got %v", test.expectedResult, result)
			}
			// Check if the nominated node name was updated correctly
			if podStatus.NominatedNodeName != test.expectedNominatedNode {
				t.Errorf("Expected NominatedNodeName to be %s, but got %s", test.expectedNominatedNode, podStatus.NominatedNodeName)
			}
			// Check if the conditions were updated correctly
			for expectedType, expectedStatus := range test.expectedConditionStatus {
				found := false
				for _, condition := range podStatus.Conditions {
					if condition.Type == expectedType {
						found = true
						if condition.Status != expectedStatus {
							t.Errorf("Expected condition %s to have status %s, but got %s", expectedType, expectedStatus, condition.Status)
						}
						break
					}
				}
				if !found {
					t.Errorf("Expected condition %s not found in pod status", expectedType)
				}
			}
		})
	}
}

func TestPodStatusPatchCall_Execute(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "uid",
			Name:      "pod",
			Namespace: "ns",
		},
		Status: v1.PodStatus{
			NominatedNodeName: "node-a",
		},
	}

	t.Run("Successful patch", func(t *testing.T) {
		client := fake.NewClientset()
		patched := false
		client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			patched = true
			return true, nil, nil
		})

		call := NewPodStatusPatchCall(pod, []*v1.PodCondition{{Type: v1.PodScheduled, Status: v1.ConditionFalse}},
			&fwk.NominatingInfo{NominatingMode: fwk.ModeNoop})
		if err := call.Execute(ctx, client); err != nil {
			t.Fatalf("Unexpected error returned by Execute(): %v", err)
		}
		if !patched {
			t.Error("Expected patch API to be called")
		}
		if !call.executed {
			t.Error("Expected 'executed' flag to be set during execution")
		}
	})

	t.Run("Skip API call if patch is not needed", func(t *testing.T) {
		client := fake.NewClientset()
		patched := false
		client.PrependReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
			patched = true
			return true, nil, nil
		})

		noOpCall := NewPodStatusPatchCall(pod, nil,
			&fwk.NominatingInfo{NominatedNodeName: "node-a", NominatingMode: fwk.ModeOverride})
		if err := noOpCall.Execute(ctx, client); err != nil {
			t.Fatalf("Unexpected error returned by Execute(): %v", err)
		}
		if patched {
			t.Error("Expected patch API not to be called if the call is no-op")
		}
	})
}
