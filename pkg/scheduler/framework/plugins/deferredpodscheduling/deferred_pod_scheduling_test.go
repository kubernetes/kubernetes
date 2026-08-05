/*
Copyright The Kubernetes Authors.

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

package deferredpodscheduling

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestDeferredPodScheduling_PreFilter(t *testing.T) {
	tests := []struct {
		name       string
		pod        *v1.Pod
		gateOn     bool
		wantStatus *fwk.Status
	}{
		{
			name: "feature gate off, deferred pod -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gateOn:     false,
			wantStatus: fwk.NewStatus(fwk.Skip),
		},
		{
			name:       "feature gate on, non-deferred pod -> skip",
			pod:        st.MakePod().Name("pod1").Node("node1").Obj(),
			gateOn:     true,
			wantStatus: fwk.NewStatus(fwk.Skip),
		},
		{
			name: "feature gate on, deferred pod -> success/nil",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gateOn:     true,
			wantStatus: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DeferredPodScheduling{
				enableInPlacePodVerticalScalingSchedulerPreemption: tt.gateOn,
			}
			_, status := pl.PreFilter(context.Background(), nil, tt.pod, nil)
			if !statusEqual(status, tt.wantStatus) {
				t.Errorf("PreFilter status = %v, want %v", status, tt.wantStatus)
			}
		})
	}
}

func TestDeferredPodScheduling_Filter(t *testing.T) {
	tests := []struct {
		name       string
		pod        *v1.Pod
		node       *v1.Node
		gateOn     bool
		wantStatus *fwk.Status
	}{
		{
			name: "feature gate off -> success (ignore disabled policy)",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec: v1.NodeSpec{
					PodPreemptionPolicy: &v1.NodePodPreemptionPolicy{
						DisableResizePreemption: []string{"policy1"},
					},
				},
			},
			gateOn:     false,
			wantStatus: nil,
		},
		{
			name: "feature gate on, non-deferred pod -> success (ignore disabled policy)",
			pod:  st.MakePod().Name("pod1").Node("node1").Obj(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec: v1.NodeSpec{
					PodPreemptionPolicy: &v1.NodePodPreemptionPolicy{
						DisableResizePreemption: []string{"policy1"},
					},
				},
			},
			gateOn:     true,
			wantStatus: nil,
		},
		{
			name: "node has nil preemption policy -> success",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			node:       st.MakeNode().Name("node1").Obj(),
			gateOn:     true,
			wantStatus: nil,
		},
		{
			name: "node has empty disable preemption policy -> success",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec: v1.NodeSpec{
					PodPreemptionPolicy: &v1.NodePodPreemptionPolicy{
						DisableResizePreemption: []string{},
					},
				},
			},
			gateOn:     true,
			wantStatus: nil,
		},
		{
			name: "node has disabled preemption policy -> unschedulable",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Spec: v1.NodeSpec{
					PodPreemptionPolicy: &v1.NodePodPreemptionPolicy{
						DisableResizePreemption: []string{"policy1"},
					},
				},
			},
			gateOn:     true,
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonNodeDisablesResizePreemption),
		},
		{
			name: "node name does not match pod assigned node -> unschedulable",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			node:       st.MakeNode().Name("node2").Obj(),
			gateOn:     true,
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod assigned to different node"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DeferredPodScheduling{
				enableInPlacePodVerticalScalingSchedulerPreemption: tt.gateOn,
			}
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(tt.node)
			status := pl.Filter(context.Background(), nil, tt.pod, nodeInfo)
			if !statusEqual(status, tt.wantStatus) {
				t.Errorf("Filter status = %v, want %v", status, tt.wantStatus)
			}
		})
	}
}

func TestDeferredPodScheduling_isSchedulableAfterNodeChange(t *testing.T) {
	nodeEnabled := st.MakeNode().Name("node1").Obj()
	nodeDisabled := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
		Spec: v1.NodeSpec{
			PodPreemptionPolicy: &v1.NodePodPreemptionPolicy{
				DisableResizePreemption: []string{"policy1"},
			},
		},
	}
	nodeOther := st.MakeNode().Name("node2").Obj()

	tests := []struct {
		name     string
		pod      *v1.Pod
		oldObj   interface{}
		newObj   interface{}
		gateOn   bool
		wantHint fwk.QueueingHint
	}{
		{
			name: "gate off, deferred pod, transition to enabled -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			oldObj:   nodeDisabled,
			newObj:   nodeEnabled,
			gateOn:   false,
			wantHint: fwk.QueueSkip,
		},
		{
			name:     "gate on, non-deferred pod, transition to enabled -> skip",
			pod:      st.MakePod().Name("pod1").Node("node1").Obj(),
			oldObj:   nodeDisabled,
			newObj:   nodeEnabled,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
		{
			name: "gate on, deferred pod, node is not pod's assigned node -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			oldObj:   nodeOther,
			newObj:   nodeOther,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
		{
			name: "gate on, deferred pod, policy transition disabled -> enabled -> queue",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			oldObj:   nodeDisabled,
			newObj:   nodeEnabled,
			gateOn:   true,
			wantHint: fwk.Queue,
		},
		{
			name: "gate on, deferred pod, policy transition enabled -> disabled -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			oldObj:   nodeEnabled,
			newObj:   nodeDisabled,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
		{
			name: "gate on, deferred pod, other node field updated -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			oldObj:   nodeEnabled,
			newObj:   nodeEnabled,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			pl := &DeferredPodScheduling{
				enableInPlacePodVerticalScalingSchedulerPreemption: tt.gateOn,
			}
			hint, err := pl.isSchedulableAfterNodeChange(logger, tt.pod, tt.oldObj, tt.newObj)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if hint != tt.wantHint {
				t.Errorf("isSchedulableAfterNodeChange hint = %v, want %v", hint, tt.wantHint)
			}
		})
	}
}

func TestDeferredPodScheduling_isSchedulableAfterNodeAdd(t *testing.T) {
	nodeEnabled := st.MakeNode().Name("node1").Obj()
	nodeDisabled := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node1"},
		Spec: v1.NodeSpec{
			PodPreemptionPolicy: &v1.NodePodPreemptionPolicy{
				DisableResizePreemption: []string{"policy1"},
			},
		},
	}
	nodeOther := st.MakeNode().Name("node2").Obj()

	tests := []struct {
		name     string
		pod      *v1.Pod
		newObj   interface{}
		gateOn   bool
		wantHint fwk.QueueingHint
	}{
		{
			name: "gate off, deferred pod, add enabled node -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			newObj:   nodeEnabled,
			gateOn:   false,
			wantHint: fwk.QueueSkip,
		},
		{
			name:     "gate on, non-deferred pod, add enabled node -> skip",
			pod:      st.MakePod().Name("pod1").Node("node1").Obj(),
			newObj:   nodeEnabled,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
		{
			name: "gate on, deferred pod, node is not pod's assigned node -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			newObj:   nodeOther,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
		{
			name: "gate on, deferred pod, add enabled node -> queue",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			newObj:   nodeEnabled,
			gateOn:   true,
			wantHint: fwk.Queue,
		},
		{
			name: "gate on, deferred pod, add disabled node -> skip",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			newObj:   nodeDisabled,
			gateOn:   true,
			wantHint: fwk.QueueSkip,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			pl := &DeferredPodScheduling{
				enableInPlacePodVerticalScalingSchedulerPreemption: tt.gateOn,
			}
			hint, err := pl.isSchedulableAfterNodeAdd(logger, tt.pod, nil, tt.newObj)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if hint != tt.wantHint {
				t.Errorf("isSchedulableAfterNodeAdd hint = %v, want %v", hint, tt.wantHint)
			}
		})
	}
}

func TestDeferredPodScheduling_Permit(t *testing.T) {
	tests := []struct {
		name       string
		pod        *v1.Pod
		gateOn     bool
		wantStatus *fwk.Status
	}{
		{
			name: "feature gate off, deferred pod -> success/nil",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gateOn:     false,
			wantStatus: nil,
		},
		{
			name:       "feature gate on, non-deferred pod -> success/nil",
			pod:        st.MakePod().Name("pod1").Node("node1").Obj(),
			gateOn:     true,
			wantStatus: nil,
		},
		{
			name: "feature gate on, deferred pod -> reject with UnschedulableAndUnresolvable",
			pod: st.MakePod().Name("pod1").Node("node1").
				Condition(v1.PodResizePending, v1.ConditionTrue, v1.PodReasonDeferred).Obj(),
			gateOn:     true,
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod resize fits, waiting for Kubelet actuation"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pl := &DeferredPodScheduling{
				enableInPlacePodVerticalScalingSchedulerPreemption: tt.gateOn,
			}
			status, _ := pl.Permit(context.Background(), nil, tt.pod, "node1")
			if !statusEqual(status, tt.wantStatus) {
				t.Errorf("Permit status = %v, want %v", status, tt.wantStatus)
			}
		})
	}
}

func statusEqual(s1, s2 *fwk.Status) bool {
	if s1 == nil && s2 == nil {
		return true
	}
	if s1 == nil || s2 == nil {
		return false
	}
	return s1.Code() == s2.Code() && s1.Message() == s2.Message()
}
