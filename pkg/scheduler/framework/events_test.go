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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
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
			changed := extractNodeAllocatableChange(newNode, oldNode) != nil
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
			changed := extractNodeLabelsChange(newNode, oldNode) != nil
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
			changed := extractNodeTaintsChange(newNode, oldNode) != nil
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
			changed := extractNodeConditionsChange(newNode, oldNode) != nil
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
			wantEvents: []ClusterEvent{NodeSpecUnschedulableChange},
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
			wantEvents: []ClusterEvent{NodeAllocatableChange},
		},
		{
			name:       "only node label changed",
			newNode:    st.MakeNode().Label("foo", "bar").Obj(),
			oldNode:    st.MakeNode().Label("foo", "fuz").Obj(),
			wantEvents: []ClusterEvent{NodeLabelChange},
		},
		{
			name: "only node taint changed",
			newNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			oldNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "foo", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			wantEvents: []ClusterEvent{NodeTaintChange},
		},
		{
			name:       "only node annotation changed",
			newNode:    st.MakeNode().Annotation("foo", "bar").Obj(),
			oldNode:    st.MakeNode().Annotation("foo", "fuz").Obj(),
			wantEvents: []ClusterEvent{NodeAnnotationChange},
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
			wantEvents: []ClusterEvent{NodeConditionChange},
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
			wantEvents: []ClusterEvent{NodeLabelChange, NodeTaintChange},
		},
	}

	for _, tc := range testCases {
		gotEvents := NodeSchedulingPropertiesChange(tc.newNode, tc.oldNode)
		if diff := cmp.Diff(tc.wantEvents, gotEvents); diff != "" {
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
	tests := []struct {
		name   string
		newPod *v1.Pod
		oldPod *v1.Pod
		want   []ClusterEvent
	}{
		{
			name:   "only label is updated",
			newPod: st.MakePod().Label("foo", "bar").Obj(),
			oldPod: st.MakePod().Label("foo", "bar2").Obj(),
			want:   []ClusterEvent{PodLabelChange},
		},
		{
			name:   "pod's resource request is scaled down",
			oldPod: podWithBigRequest,
			newPod: podWithSmallRequest,
			want:   []ClusterEvent{PodRequestScaledDown},
		},
		{
			name:   "pod's resource request is scaled up",
			oldPod: podWithSmallRequest,
			newPod: podWithBigRequest,
			want:   []ClusterEvent{assignedPodOtherUpdate},
		},
		{
			name:   "both pod's resource request and label are updated",
			oldPod: podWithBigRequest,
			newPod: podWithSmallRequestAndLabel,
			want:   []ClusterEvent{PodLabelChange, PodRequestScaledDown},
		},
		{
			name:   "untracked properties of pod is updated",
			newPod: st.MakePod().Annotation("foo", "bar").Obj(),
			oldPod: st.MakePod().Annotation("foo", "bar2").Obj(),
			want:   []ClusterEvent{assignedPodOtherUpdate},
		},
		{
			name:   "scheduling gate is eliminated",
			newPod: st.MakePod().SchedulingGates([]string{}).Obj(),
			oldPod: st.MakePod().SchedulingGates([]string{"foo"}).Obj(),
			want:   []ClusterEvent{PodSchedulingGateEliminatedChange},
		},
		{
			name:   "scheduling gate is removed, but not completely eliminated",
			newPod: st.MakePod().SchedulingGates([]string{"foo"}).Obj(),
			oldPod: st.MakePod().SchedulingGates([]string{"foo", "bar"}).Obj(),
			want:   []ClusterEvent{assignedPodOtherUpdate},
		},
		{
			name:   "pod's tolerations are updated",
			newPod: st.MakePod().Toleration("key").Toleration("key2").Obj(),
			oldPod: st.MakePod().Toleration("key").Obj(),
			want:   []ClusterEvent{PodTolerationChange},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PodSchedulingPropertiesChange(tt.newPod, tt.oldPod)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("unexpected event is returned from podSchedulingPropertiesChange (-want, +got):\n%s", diff)
			}
		})
	}
}
