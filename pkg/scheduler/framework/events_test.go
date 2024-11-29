/*
Copyright 2024 The Kubernetes Authors.

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

package framework

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

func TestNodeAllocatableChange(t *testing.T) {
	newQuantity := func(value int64) resource.Quantity {
		return *resource.NewQuantity(value, resource.BinarySI)
	}
	for _, test := range []struct {
		name string
		// changed is true if it's expected that the function detects the change and returns event.
		changed        bool
		oldAllocatable v1.ResourceList
		newAllocatable v1.ResourceList
	}{
		{
			name:           "no allocatable resources changed",
			changed:        false,
			oldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			newAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
		},
		{
			name:           "new node has more allocatable resources",
			changed:        true,
			oldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			newAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024), v1.ResourceStorage: newQuantity(1024)},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldNode := &v1.Node{Status: v1.NodeStatus{Allocatable: test.oldAllocatable}}
			newNode := &v1.Node{Status: v1.NodeStatus{Allocatable: test.newAllocatable}}
			changed := extractNodeAllocatableChange(newNode, oldNode) != none
			if changed != test.changed {
				t.Errorf("nodeAllocatableChanged should be %t, got %t", test.changed, changed)
			}
		})
	}
}

func TestNodeLabelsChange(t *testing.T) {
	for _, test := range []struct {
		name string
		// changed is true if it's expected that the function detects the change and returns event.
		changed   bool
		oldLabels map[string]string
		newLabels map[string]string
	}{
		{
			name:      "no labels changed",
			changed:   false,
			oldLabels: map[string]string{"foo": "bar"},
			newLabels: map[string]string{"foo": "bar"},
		},
		// Labels changed.
		{
			name:      "new object has more labels",
			changed:   true,
			oldLabels: map[string]string{"foo": "bar"},
			newLabels: map[string]string{"foo": "bar", "test": "value"},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.oldLabels}}
			newNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.newLabels}}
			changed := extractNodeLabelsChange(newNode, oldNode) != none
			if changed != test.changed {
				t.Errorf("Test case %q failed: should be %t, got %t", test.name, test.changed, changed)
			}
		})
	}
}

func TestNodeTaintsChange(t *testing.T) {
	for _, test := range []struct {
		name string
		// changed is true if it's expected that the function detects the change and returns event.
		changed   bool
		oldTaints []v1.Taint
		newTaints []v1.Taint
	}{
		{
			name:      "no taint changed",
			changed:   false,
			oldTaints: []v1.Taint{{Key: "key", Value: "value"}},
			newTaints: []v1.Taint{{Key: "key", Value: "value"}},
		},
		{
			name:      "taint value changed",
			changed:   true,
			oldTaints: []v1.Taint{{Key: "key", Value: "value1"}},
			newTaints: []v1.Taint{{Key: "key", Value: "value2"}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldNode := &v1.Node{Spec: v1.NodeSpec{Taints: test.oldTaints}}
			newNode := &v1.Node{Spec: v1.NodeSpec{Taints: test.newTaints}}
			changed := extractNodeTaintsChange(newNode, oldNode) != none
			if changed != test.changed {
				t.Errorf("Test case %q failed: should be %t, not %t", test.name, test.changed, changed)
			}
		})
	}
}

func TestNodeConditionsChange(t *testing.T) {
	nodeConditionType := reflect.TypeOf(v1.NodeCondition{})
	if nodeConditionType.NumField() != 6 {
		t.Errorf("NodeCondition type has changed. The nodeConditionsChange() function must be reevaluated.")
	}

	for _, test := range []struct {
		name string
		// changed is true if it's expected that the function detects the change and returns event.
		changed       bool
		oldConditions []v1.NodeCondition
		newConditions []v1.NodeCondition
	}{
		{
			name:          "no condition changed",
			changed:       false,
			oldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
			newConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
		},
		{
			name:          "only LastHeartbeatTime changed",
			changed:       false,
			oldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(1, 0)}},
			newConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(2, 0)}},
		},
		{
			name:          "new node has more healthy conditions",
			changed:       true,
			oldConditions: []v1.NodeCondition{},
			newConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
		{
			name:          "new node has less unhealthy conditions",
			changed:       true,
			oldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
			newConditions: []v1.NodeCondition{},
		},
		{
			name:          "condition status changed",
			changed:       true,
			oldConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			newConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldNode := &v1.Node{Status: v1.NodeStatus{Conditions: test.oldConditions}}
			newNode := &v1.Node{Status: v1.NodeStatus{Conditions: test.newConditions}}
			changed := extractNodeConditionsChange(newNode, oldNode) != none
			if changed != test.changed {
				t.Errorf("Test case %q failed: should be %t, got %t", test.name, test.changed, changed)
			}
		})
	}
}

func TestNodeSchedulingPropertiesChange(t *testing.T) {
	testCases := []struct {
		name       string
		newNode    *v1.Node
		oldNode    *v1.Node
		wantEvents []ClusterEvent
	}{
		{
			name:       "no specific changed applied",
			newNode:    st.MakeNode().Unschedulable(false).Obj(),
			oldNode:    st.MakeNode().Unschedulable(false).Obj(),
			wantEvents: nil,
		},
		{
			name:       "only node spec unavailable changed",
			newNode:    st.MakeNode().Unschedulable(false).Obj(),
			oldNode:    st.MakeNode().Unschedulable(true).Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeTaint}},
		},
		{
			name: "only node allocatable changed",
			newNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:                     "1000m",
				v1.ResourceMemory:                  "100m",
				v1.ResourceName("example.com/foo"): "1"},
			).Obj(),
			oldNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:                     "1000m",
				v1.ResourceMemory:                  "100m",
				v1.ResourceName("example.com/foo"): "2"},
			).Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeAllocatable}},
		},
		{
			name:       "only node label changed",
			newNode:    st.MakeNode().Label("foo", "bar").Obj(),
			oldNode:    st.MakeNode().Label("foo", "fuz").Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeLabel}},
		},
		{
			name: "only node taint changed",
			newNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			oldNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "foo", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeTaint}},
		},
		{
			name:       "only node annotation changed",
			newNode:    st.MakeNode().Annotation("foo", "bar").Obj(),
			oldNode:    st.MakeNode().Annotation("foo", "fuz").Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeAnnotation}},
		},
		{
			name:    "only node condition changed",
			newNode: st.MakeNode().Obj(),
			oldNode: st.MakeNode().Condition(
				v1.NodeReady,
				v1.ConditionTrue,
				"Ready",
				"Ready",
			).Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeCondition}},
		},
		{
			name: "both node label and node taint changed",
			newNode: st.MakeNode().
				Label("foo", "bar").
				Taints([]v1.Taint{
					{Key: v1.TaintNodeUnschedulable, Value: "", Effect: v1.TaintEffectNoSchedule},
				}).Obj(),
			oldNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "foo", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			wantEvents: []ClusterEvent{{Resource: Node, ActionType: UpdateNodeLabel}, {Resource: Node, ActionType: UpdateNodeTaint}},
		},
	}

	for _, tc := range testCases {
		gotEvents := NodeSchedulingPropertiesChange(tc.newNode, tc.oldNode)
		if diff := cmp.Diff(tc.wantEvents, gotEvents, cmpopts.EquateComparable(ClusterEvent{})); diff != "" {
			t.Errorf("unexpected event (-want, +got):\n%s", diff)
		}
	}
}

func Test_podSchedulingPropertiesChange(t *testing.T) {
	podWithBigRequest := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "app",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("101m")},
					},
				},
			},
		},
		Status: v1.PodStatus{
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "app",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: resource.MustParse("101m")},
				},
			},
		},
	}
	podWithSmallRequestAndLabel := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"foo": "bar"},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "app",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
					},
				},
			},
		},
		Status: v1.PodStatus{
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "app",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
				},
			},
		},
	}
	podWithSmallRequest := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "app",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
					},
				},
			},
		},
		Status: v1.PodStatus{
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               "app",
					AllocatedResources: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
				},
			},
		},
	}
	claimStatusA := v1.PodResourceClaimStatus{
		Name:              "my-claim",
		ResourceClaimName: ptr.To("claim"),
	}
	claimStatusB := v1.PodResourceClaimStatus{
		Name:              "my-claim-2",
		ResourceClaimName: ptr.To("claim-2"),
	}
	tests := []struct {
		name        string
		newPod      *v1.Pod
		oldPod      *v1.Pod
		draDisabled bool
		want        []ClusterEvent
	}{
		{
			name:   "assigned pod is updated",
			newPod: st.MakePod().Label("foo", "bar").Node("node").Obj(),
			oldPod: st.MakePod().Label("foo", "bar2").Node("node").Obj(),
			want:   []ClusterEvent{{Resource: assignedPod, ActionType: UpdatePodLabel}},
		},
		{
			name:   "only label is updated",
			newPod: st.MakePod().Label("foo", "bar").Obj(),
			oldPod: st.MakePod().Label("foo", "bar2").Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodLabel}},
		},
		{
			name:   "pod's resource request is scaled down",
			oldPod: podWithBigRequest,
			newPod: podWithSmallRequest,
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodScaleDown}},
		},
		{
			name:   "pod's resource request is scaled up",
			oldPod: podWithSmallRequest,
			newPod: podWithBigRequest,
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: updatePodOther}},
		},
		{
			name:   "both pod's resource request and label are updated",
			oldPod: podWithBigRequest,
			newPod: podWithSmallRequestAndLabel,
			want: []ClusterEvent{
				{Resource: unschedulablePod, ActionType: UpdatePodLabel},
				{Resource: unschedulablePod, ActionType: UpdatePodScaleDown},
			},
		},
		{
			name:   "untracked properties of pod is updated",
			newPod: st.MakePod().Annotation("foo", "bar").Obj(),
			oldPod: st.MakePod().Annotation("foo", "bar2").Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: updatePodOther}},
		},
		{
			name:   "scheduling gate is eliminated",
			newPod: st.MakePod().SchedulingGates([]string{}).Obj(),
			oldPod: st.MakePod().SchedulingGates([]string{"foo"}).Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodSchedulingGatesEliminated}},
		},
		{
			name:   "scheduling gate is removed, but not completely eliminated",
			newPod: st.MakePod().SchedulingGates([]string{"foo"}).Obj(),
			oldPod: st.MakePod().SchedulingGates([]string{"foo", "bar"}).Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: updatePodOther}},
		},
		{
			name:   "pod's tolerations are updated",
			newPod: st.MakePod().Toleration("key").Toleration("key2").Obj(),
			oldPod: st.MakePod().Toleration("key").Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodToleration}},
		},
		{
			name:        "pod claim statuses change, feature disabled",
			draDisabled: true,
			newPod:      st.MakePod().ResourceClaimStatuses(claimStatusA).Obj(),
			oldPod:      st.MakePod().Obj(),
			want:        []ClusterEvent{{Resource: unschedulablePod, ActionType: updatePodOther}},
		},
		{
			name:   "pod claim statuses change, feature enabled",
			newPod: st.MakePod().ResourceClaimStatuses(claimStatusA).Obj(),
			oldPod: st.MakePod().Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodGeneratedResourceClaim}},
		},
		{
			name:   "pod claim statuses swapped",
			newPod: st.MakePod().ResourceClaimStatuses(claimStatusA, claimStatusB).Obj(),
			oldPod: st.MakePod().ResourceClaimStatuses(claimStatusB, claimStatusA).Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodGeneratedResourceClaim}},
		},
		{
			name:   "pod claim statuses extended",
			newPod: st.MakePod().ResourceClaimStatuses(claimStatusA, claimStatusB).Obj(),
			oldPod: st.MakePod().ResourceClaimStatuses(claimStatusA).Obj(),
			want:   []ClusterEvent{{Resource: unschedulablePod, ActionType: UpdatePodGeneratedResourceClaim}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, !tt.draDisabled)
			got := PodSchedulingPropertiesChange(tt.newPod, tt.oldPod)
			if diff := cmp.Diff(tt.want, got, cmpopts.EquateComparable(ClusterEvent{})); diff != "" {
				t.Errorf("unexpected event is returned from podSchedulingPropertiesChange (-want, +got):\n%s", diff)
			}
		})
	}
}
