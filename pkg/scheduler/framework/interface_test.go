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

package framework

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type nodeInfoLister []fwk.NodeInfo

func (nodes nodeInfoLister) Get(nodeName string) (fwk.NodeInfo, error) {
	for _, node := range nodes {
		if node != nil && node.Node().Name == nodeName {
			return node, nil
		}
	}
	return nil, fmt.Errorf("unable to find node: %s", nodeName)
}

func (nodes nodeInfoLister) List() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

func (nodes nodeInfoLister) HavePodsWithAffinityList() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

func (nodes nodeInfoLister) HavePodsWithRequiredAntiAffinityList() ([]fwk.NodeInfo, error) {
	return nodes, nil
}

func TestNodesForStatusCode(t *testing.T) {
	// Prepare 4 nodes names.
	nodeNames := []string{"node1", "node2", "node3", "node4"}
	tests := []struct {
		name          string
		nodesStatuses *NodeToStatus
		code          fwk.Code
		expected      sets.Set[string] // set of expected node names.
	}{
		{
			name: "No node should be attempted",
			nodesStatuses: NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node2": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node3": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node4": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			code:     fwk.Unschedulable,
			expected: sets.New[string](),
		},
		{
			name: "All nodes should be attempted",
			nodesStatuses: NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node2": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node3": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node4": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			code:     fwk.UnschedulableAndUnresolvable,
			expected: sets.New("node1", "node2", "node3", "node4"),
		},
		{
			name:          "No node should be attempted, as all are implicitly not matching the code",
			nodesStatuses: NewDefaultNodeToStatus(),
			code:          fwk.Unschedulable,
			expected:      sets.New[string](),
		},
		{
			name:          "All nodes should be attempted, as all are implicitly matching the code",
			nodesStatuses: NewDefaultNodeToStatus(),
			code:          fwk.UnschedulableAndUnresolvable,
			expected:      sets.New("node1", "node2", "node3", "node4"),
		},
		{
			name: "UnschedulableAndUnresolvable status should be skipped but Unschedulable should be tried",
			nodesStatuses: NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node3": fwk.NewStatus(fwk.Unschedulable),
				// node4 is UnschedulableAndUnresolvable by absence
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			code:     fwk.Unschedulable,
			expected: sets.New("node1", "node3"),
		},
		{
			name: "Unschedulable status should be skipped but UnschedulableAndUnresolvable should be tried",
			nodesStatuses: NewNodeToStatus(map[string]*fwk.Status{
				"node1": fwk.NewStatus(fwk.Unschedulable),
				"node2": fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
				"node3": fwk.NewStatus(fwk.Unschedulable),
				// node4 is UnschedulableAndUnresolvable by absence
			}, fwk.NewStatus(fwk.UnschedulableAndUnresolvable)),
			code:     fwk.UnschedulableAndUnresolvable,
			expected: sets.New("node2", "node4"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var nodeInfos nodeInfoLister
			for _, name := range nodeNames {
				ni := NewNodeInfo()
				ni.SetNode(st.MakeNode().Name(name).Obj())
				nodeInfos = append(nodeInfos, ni)
			}
			nodes, err := tt.nodesStatuses.NodesForStatusCode(nodeInfos, tt.code)
			if err != nil {
				t.Fatalf("Failed to get nodes for status code: %s", err)
			}
			if len(tt.expected) != len(nodes) {
				t.Errorf("Number of nodes is not the same as expected. expected: %d, got: %d. Nodes: %v", len(tt.expected), len(nodes), nodes)
			}
			for _, node := range nodes {
				name := node.Node().Name
				if _, found := tt.expected[name]; !found {
					t.Errorf("Node %v is not expected", name)
				}
			}
		})
	}
}
