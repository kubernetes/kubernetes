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

package queue

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestNodeAllocatableChanged(t *testing.T) {
	newQuantity := func(value int64) resource.Quantity {
		return *resource.NewQuantity(value, resource.BinarySI)
	}
	for _, test := range []struct {
		Name           string
		Changed        bool
		OldAllocatable v1.ResourceList
		NewAllocatable v1.ResourceList
	}{
		{
			Name:           "no allocatable resources changed",
			Changed:        false,
			OldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			NewAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
		},
		{
			Name:           "new node has more allocatable resources",
			Changed:        true,
			OldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			NewAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024), v1.ResourceStorage: newQuantity(1024)},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{Status: v1.NodeStatus{Allocatable: test.OldAllocatable}}
			newNode := &v1.Node{Status: v1.NodeStatus{Allocatable: test.NewAllocatable}}
			changed := nodeAllocatableChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("nodeAllocatableChanged should be %t, got %t", test.Changed, changed)
			}
		})
	}
}

func TestNodeLabelsChanged(t *testing.T) {
	for _, test := range []struct {
		Name      string
		Changed   bool
		OldLabels map[string]string
		NewLabels map[string]string
	}{
		{
			Name:      "no labels changed",
			Changed:   false,
			OldLabels: map[string]string{"foo": "bar"},
			NewLabels: map[string]string{"foo": "bar"},
		},
		// Labels changed.
		{
			Name:      "new node has more labels",
			Changed:   true,
			OldLabels: map[string]string{"foo": "bar"},
			NewLabels: map[string]string{"foo": "bar", "test": "value"},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.OldLabels}}
			newNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.NewLabels}}
			changed := nodeLabelsChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("Test case %q failed: should be %t, got %t", test.Name, test.Changed, changed)
			}
		})
	}
}

func TestNodeTaintsChanged(t *testing.T) {
	for _, test := range []struct {
		Name      string
		Changed   bool
		OldTaints []v1.Taint
		NewTaints []v1.Taint
	}{
		{
			Name:      "no taint changed",
			Changed:   false,
			OldTaints: []v1.Taint{{Key: "key", Value: "value"}},
			NewTaints: []v1.Taint{{Key: "key", Value: "value"}},
		},
		{
			Name:      "taint value changed",
			Changed:   true,
			OldTaints: []v1.Taint{{Key: "key", Value: "value1"}},
			NewTaints: []v1.Taint{{Key: "key", Value: "value2"}},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{Spec: v1.NodeSpec{Taints: test.OldTaints}}
			newNode := &v1.Node{Spec: v1.NodeSpec{Taints: test.NewTaints}}
			changed := nodeTaintsChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("Test case %q failed: should be %t, not %t", test.Name, test.Changed, changed)
			}
		})
	}
}

func TestNodeConditionsChanged(t *testing.T) {
	nodeConditionType := reflect.TypeOf(v1.NodeCondition{})
	if nodeConditionType.NumField() != 6 {
		t.Errorf("NodeCondition type has changed. The nodeConditionsChanged() function must be reevaluated.")
	}

	for _, test := range []struct {
		Name          string
		Changed       bool
		OldConditions []v1.NodeCondition
		NewConditions []v1.NodeCondition
	}{
		{
			Name:          "no condition changed",
			Changed:       false,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
		},
		{
			Name:          "only LastHeartbeatTime changed",
			Changed:       false,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(1, 0)}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(2, 0)}},
		},
		{
			Name:          "new node has more healthy conditions",
			Changed:       true,
			OldConditions: []v1.NodeCondition{},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
		{
			Name:          "new node has less unhealthy conditions",
			Changed:       true,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeDiskPressure, Status: v1.ConditionTrue}},
			NewConditions: []v1.NodeCondition{},
		},
		{
			Name:          "condition status changed",
			Changed:       true,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	} {
		t.Run(test.Name, func(t *testing.T) {
			oldNode := &v1.Node{Status: v1.NodeStatus{Conditions: test.OldConditions}}
			newNode := &v1.Node{Status: v1.NodeStatus{Conditions: test.NewConditions}}
			changed := nodeConditionsChanged(newNode, oldNode)
			if changed != test.Changed {
				t.Errorf("Test case %q failed: should be %t, got %t", test.Name, test.Changed, changed)
			}
		})
	}
}

func TestNodeSchedulingPropertiesChange(t *testing.T) {
	testCases := []struct {
		name       string
		newNode    *v1.Node
		oldNode    *v1.Node
		wantEvents []framework.ClusterEvent
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
			wantEvents: []framework.ClusterEvent{NodeSpecUnschedulableChange},
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
			wantEvents: []framework.ClusterEvent{NodeAllocatableChange},
		},
		{
			name:       "only node label changed",
			newNode:    st.MakeNode().Label("foo", "bar").Obj(),
			oldNode:    st.MakeNode().Label("foo", "fuz").Obj(),
			wantEvents: []framework.ClusterEvent{NodeLabelChange},
		},
		{
			name: "only node taint changed",
			newNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			oldNode: st.MakeNode().Taints([]v1.Taint{
				{Key: v1.TaintNodeUnschedulable, Value: "foo", Effect: v1.TaintEffectNoSchedule},
			}).Obj(),
			wantEvents: []framework.ClusterEvent{NodeTaintChange},
		},
		{
			name:       "only node annotation changed",
			newNode:    st.MakeNode().Annotation("foo", "bar").Obj(),
			oldNode:    st.MakeNode().Annotation("foo", "fuz").Obj(),
			wantEvents: []framework.ClusterEvent{NodeAnnotationChange},
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
			wantEvents: []framework.ClusterEvent{NodeConditionChange},
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
			wantEvents: []framework.ClusterEvent{NodeLabelChange, NodeTaintChange},
		},
	}

	for _, tc := range testCases {
		gotEvents := NodeSchedulingPropertiesChange(tc.newNode, tc.oldNode)
		if diff := cmp.Diff(tc.wantEvents, gotEvents); diff != "" {
			t.Errorf("unexpected event (-want, +got):\n%s", diff)
		}
	}
}
