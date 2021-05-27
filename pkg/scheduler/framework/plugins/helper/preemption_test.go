/*
Copyright 2021 The Kubernetes Authors.

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

package helper

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

const (
	// NodeAffinityErrReasonPod is the reason for Pod's node affinity/selector not matching.
	NodeAffinityErrReasonPod = "node(s) didn't match Pod's node affinity/selector"
	// NodeNameErrReason returned when node name doesn't match.
	NodeNameErrReason = "node(s) didn't match the requested node name"
	// TaintTolerationErrReasonNotMatch is the Filter reason status when not matching.
	TaintTolerationErrReasonNotMatch = "node(s) had taints that the pod didn't tolerate"
	// NodeLabelErrReasonPresenceViolated is used for CheckNodeLabelPresence filter error.
	NodeLabelErrReasonPresenceViolated = "node(s) didn't have the requested labels"
	// NodeUnschedulableErrReasonUnschedulable is used for NodeUnschedulable filter error.
	NodeUnschedulableErrReasonUnschedulable = "node(s) were unschedulable"
	// NodeUnschedulableErrReasonUnknownCondition is used for NodeUnknownCondition filter error.
	NodeUnschedulableErrReasonUnknownCondition = "node(s) had unknown conditions"
	// ErrReasonAffinityRulesNotMatch is used for PodAffinityRulesNotMatch filter error.
	ErrReasonAffinityRulesNotMatch = "node(s) didn't match pod affinity rules"
	// ErrReasonAntiAffinityRulesNotMatch is used for PodAntiAffinityRulesNotMatch predicate error.
	ErrReasonAntiAffinityRulesNotMatch = "node(s) didn't match pod anti-affinity rules"
	// VolumeRestrictionsErrReasonDiskConflict is used for NoDiskConflict filter error.
	VolumeRestrictionsErrReasonDiskConflict = "node(s) had no available disk"
	// VolumeZoneErrReasonConflict is used for NoVolumeZoneConflict filter error.
	VolumeZoneErrReasonConflict = "node(s) had no available volume zone"
	// VolumeSchedulingErrReasonNodeConflict is used for VolumeNodeAffinityConflict filter error.
	VolumeSchedulingErrReasonNodeConflict = "node(s) had volume node affinity conflict"
	// PodTopologySpreadErrReasonConstraintsNotMatch is used for PodTopologySpread filter error.
	PodTopologySpreadErrReasonConstraintsNotMatch = "node(s) didn't match pod topology spread constraints"
	// PodTopologySpreadErrReasonNodeLabelNotMatch is used when the node doesn't hold the required label.
	PodTopologySpreadErrReasonNodeLabelNotMatch = PodTopologySpreadErrReasonConstraintsNotMatch + " (missing required label)"
)

var lowPriority, highPriority = int32(0), int32(1000)

func TestNodesWherePreemptionMightHelp(t *testing.T) {
	// Prepare 4 nodes names.
	nodeNames := []string{"node1", "node2", "node3", "node4"}
	tests := []struct {
		name          string
		nodesStatuses framework.NodeToStatusMap
		expected      sets.String // set of expected node names.
	}{
		{
			name: "No node should be attempted",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeAffinityErrReasonPod),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeNameErrReason),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, TaintTolerationErrReasonNotMatch),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeLabelErrReasonPresenceViolated),
			},
			expected: sets.NewString(),
		},
		{
			name: "ErrReasonAntiAffinityRulesNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, ErrReasonAntiAffinityRulesNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeNameErrReason),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeUnschedulableErrReasonUnschedulable),
			},
			expected: sets.NewString("node1", "node4"),
		},
		{
			name: "ErrReasonAffinityRulesNotMatch should not be tried as it indicates that the pod is unschedulable due to inter-pod affinity, but ErrReasonAffinityNotMatch should be tried as it indicates that the pod is unschedulable due to inter-pod affinity or anti-affinity",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonAffinityRulesNotMatch),
				"node2": framework.NewStatus(framework.Unschedulable, ErrReasonAntiAffinityRulesNotMatch),
			},
			expected: sets.NewString("node2", "node3", "node4"),
		},
		{
			name: "Mix of failed filters works fine",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, VolumeRestrictionsErrReasonDiskConflict),
				"node2": framework.NewStatus(framework.Unschedulable, fmt.Sprintf("Insufficient %v", v1.ResourceMemory)),
			},
			expected: sets.NewString("node2", "node3", "node4"),
		},
		{
			name: "Node condition errors should be considered unresolvable",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeUnschedulableErrReasonUnknownCondition),
			},
			expected: sets.NewString("node2", "node3", "node4"),
		},
		{
			name: "ErrVolume... errors should not be tried as it indicates that the pod is unschedulable due to no matching volumes for pod on node",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.UnschedulableAndUnresolvable, VolumeZoneErrReasonConflict),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(VolumeSchedulingErrReasonNodeConflict)),
				"node3": framework.NewStatus(framework.UnschedulableAndUnresolvable, string(VolumeSchedulingErrReasonNodeConflict)),
			},
			expected: sets.NewString("node4"),
		},
		{
			name: "ErrReasonConstraintsNotMatch should be tried as it indicates that the pod is unschedulable due to topology spread constraints",
			nodesStatuses: framework.NodeToStatusMap{
				"node1": framework.NewStatus(framework.Unschedulable, PodTopologySpreadErrReasonConstraintsNotMatch),
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, NodeNameErrReason),
				"node3": framework.NewStatus(framework.Unschedulable, PodTopologySpreadErrReasonConstraintsNotMatch),
			},
			expected: sets.NewString("node1", "node3", "node4"),
		},
		{
			name: "UnschedulableAndUnresolvable status should be skipped but Unschedulable should be tried",
			nodesStatuses: framework.NodeToStatusMap{
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
				"node3": framework.NewStatus(framework.Unschedulable, ""),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
			},
			expected: sets.NewString("node1", "node3"),
		},
		{
			name: "ErrReasonNodeLabelNotMatch should not be tried as it indicates that the pod is unschedulable due to node doesn't have the required label",
			nodesStatuses: framework.NodeToStatusMap{
				"node2": framework.NewStatus(framework.UnschedulableAndUnresolvable, PodTopologySpreadErrReasonNodeLabelNotMatch),
				"node3": framework.NewStatus(framework.Unschedulable, ""),
				"node4": framework.NewStatus(framework.UnschedulableAndUnresolvable, ""),
			},
			expected: sets.NewString("node1", "node3"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var nodeInfos []*framework.NodeInfo
			for _, name := range nodeNames {
				ni := framework.NewNodeInfo()
				ni.SetNode(st.MakeNode().Name(name).Obj())
				nodeInfos = append(nodeInfos, ni)
			}
			nodes, _ := nodesWherePreemptionMightHelp(nodeInfos, tt.nodesStatuses)
			if len(tt.expected) != len(nodes) {
				t.Errorf("number of nodes is not the same as expected. exptectd: %d, got: %d. Nodes: %v", len(tt.expected), len(nodes), nodes)
			}
			for _, node := range nodes {
				name := node.Node().Name
				if _, found := tt.expected[name]; !found {
					t.Errorf("node %v is not expected.", name)
				}
			}
		})
	}
}

func TestPodEligibleToPreemptOthers(t *testing.T) {
	tests := []struct {
		name                string
		pod                 *v1.Pod
		pods                []*v1.Pod
		nodes               []string
		nominatedNodeStatus *framework.Status
		expected            bool
	}{
		{
			name:                "Pod with nominated node",
			pod:                 st.MakePod().Name("p_with_nominated_node").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			pods:                []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(lowPriority).Node("node1").Terminating().Obj()},
			nodes:               []string{"node1"},
			nominatedNodeStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, TaintTolerationErrReasonNotMatch),
			expected:            true,
		},
		{
			name:                "Pod with nominated node, but without nominated node status",
			pod:                 st.MakePod().Name("p_without_status").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj(),
			pods:                []*v1.Pod{st.MakePod().Name("p1").UID("p1").Priority(lowPriority).Node("node1").Terminating().Obj()},
			nodes:               []string{"node1"},
			nominatedNodeStatus: nil,
			expected:            false,
		},
		{
			name:                "Pod without nominated node",
			pod:                 st.MakePod().Name("p_without_nominated_node").UID("p").Priority(highPriority).Obj(),
			pods:                []*v1.Pod{},
			nodes:               []string{},
			nominatedNodeStatus: nil,
			expected:            true,
		},
		{
			name:                "Pod with 'PreemptNever' preemption policy",
			pod:                 st.MakePod().Name("p_with_preempt_never_policy").UID("p").Priority(highPriority).PreemptionPolicy(v1.PreemptNever).Obj(),
			pods:                []*v1.Pod{},
			nodes:               []string{},
			nominatedNodeStatus: nil,
			expected:            false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var nodes []*v1.Node
			for _, n := range test.nodes {
				nodes = append(nodes, st.MakeNode().Name(n).Obj())
			}
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			if got := PodEligibleToPreemptOthers(test.pod, snapshot.NodeInfos(), test.nominatedNodeStatus); got != test.expected {
				t.Errorf("expected %t, got %t for pod: %s", test.expected, got, test.pod.Name)
			}
		})
	}
}
