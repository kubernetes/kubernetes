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
	"errors"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

var errorStatus = NewStatus(Error, "internal error")
var statusWithErr = AsStatus(errors.New("internal error"))

func TestStatus(t *testing.T) {
	tests := []struct {
		name              string
		status            *Status
		expectedCode      Code
		expectedMessage   string
		expectedIsSuccess bool
		expectedIsWait    bool
		expectedIsSkip    bool
		expectedAsError   error
	}{
		{
			name:              "success status",
			status:            NewStatus(Success, ""),
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedIsWait:    false,
			expectedIsSkip:    false,
			expectedAsError:   nil,
		},
		{
			name:              "wait status",
			status:            NewStatus(Wait, ""),
			expectedCode:      Wait,
			expectedMessage:   "",
			expectedIsSuccess: false,
			expectedIsWait:    true,
			expectedIsSkip:    false,
			expectedAsError:   nil,
		},
		{
			name:              "error status",
			status:            NewStatus(Error, "unknown error"),
			expectedCode:      Error,
			expectedMessage:   "unknown error",
			expectedIsSuccess: false,
			expectedIsWait:    false,
			expectedIsSkip:    false,
			expectedAsError:   errors.New("unknown error"),
		},
		{
			name:              "skip status",
			status:            NewStatus(Skip, ""),
			expectedCode:      Skip,
			expectedMessage:   "",
			expectedIsSuccess: false,
			expectedIsWait:    false,
			expectedIsSkip:    true,
			expectedAsError:   nil,
		},
		{
			name:              "nil status",
			status:            nil,
			expectedCode:      Success,
			expectedMessage:   "",
			expectedIsSuccess: true,
			expectedIsSkip:    false,
			expectedAsError:   nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.status.Code() != test.expectedCode {
				t.Errorf("expect status.Code() returns %v, but %v", test.expectedCode, test.status.Code())
			}

			if test.status.Message() != test.expectedMessage {
				t.Errorf("expect status.Message() returns %v, but %v", test.expectedMessage, test.status.Message())
			}

			if test.status.IsSuccess() != test.expectedIsSuccess {
				t.Errorf("expect status.IsSuccess() returns %v, but %v", test.expectedIsSuccess, test.status.IsSuccess())
			}

			if test.status.IsWait() != test.expectedIsWait {
				t.Errorf("status.IsWait() returns %v, but want %v", test.status.IsWait(), test.expectedIsWait)
			}

			if test.status.IsSkip() != test.expectedIsSkip {
				t.Errorf("status.IsSkip() returns %v, but want %v", test.status.IsSkip(), test.expectedIsSkip)
			}

			if test.status.AsError() == test.expectedAsError {
				return
			}

			if test.status.AsError().Error() != test.expectedAsError.Error() {
				t.Errorf("expect status.AsError() returns %v, but %v", test.expectedAsError, test.status.AsError())
			}
		})
	}
}

func TestPreFilterResultMerge(t *testing.T) {
	tests := map[string]struct {
		receiver *PreFilterResult
		in       *PreFilterResult
		want     *PreFilterResult
	}{
		"all nil": {},
		"nil receiver empty input": {
			in:   &PreFilterResult{NodeNames: sets.New[string]()},
			want: &PreFilterResult{NodeNames: sets.New[string]()},
		},
		"empty receiver nil input": {
			receiver: &PreFilterResult{NodeNames: sets.New[string]()},
			want:     &PreFilterResult{NodeNames: sets.New[string]()},
		},
		"empty receiver empty input": {
			receiver: &PreFilterResult{NodeNames: sets.New[string]()},
			in:       &PreFilterResult{NodeNames: sets.New[string]()},
			want:     &PreFilterResult{NodeNames: sets.New[string]()},
		},
		"nil receiver populated input": {
			in:   &PreFilterResult{NodeNames: sets.New("node1")},
			want: &PreFilterResult{NodeNames: sets.New("node1")},
		},
		"empty receiver populated input": {
			receiver: &PreFilterResult{NodeNames: sets.New[string]()},
			in:       &PreFilterResult{NodeNames: sets.New("node1")},
			want:     &PreFilterResult{NodeNames: sets.New[string]()},
		},

		"populated receiver nil input": {
			receiver: &PreFilterResult{NodeNames: sets.New("node1")},
			want:     &PreFilterResult{NodeNames: sets.New("node1")},
		},
		"populated receiver empty input": {
			receiver: &PreFilterResult{NodeNames: sets.New("node1")},
			in:       &PreFilterResult{NodeNames: sets.New[string]()},
			want:     &PreFilterResult{NodeNames: sets.New[string]()},
		},
		"populated receiver and input": {
			receiver: &PreFilterResult{NodeNames: sets.New("node1", "node2")},
			in:       &PreFilterResult{NodeNames: sets.New("node2", "node3")},
			want:     &PreFilterResult{NodeNames: sets.New("node2")},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			got := test.receiver.Merge(test.in)
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("unexpected diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestIsStatusEqual(t *testing.T) {
	tests := []struct {
		name string
		x, y *Status
		want bool
	}{
		{
			name: "two nil should be equal",
			x:    nil,
			y:    nil,
			want: true,
		},
		{
			name: "nil should be equal to success status",
			x:    nil,
			y:    NewStatus(Success),
			want: true,
		},
		{
			name: "nil should not be equal with status except success",
			x:    nil,
			y:    NewStatus(Error, "internal error"),
			want: false,
		},
		{
			name: "one status should be equal to itself",
			x:    errorStatus,
			y:    errorStatus,
			want: true,
		},
		{
			name: "same type statuses without reasons should be equal",
			x:    NewStatus(Success),
			y:    NewStatus(Success),
			want: true,
		},
		{
			name: "statuses with same message should be equal",
			x:    NewStatus(Unschedulable, "unschedulable"),
			y:    NewStatus(Unschedulable, "unschedulable"),
			want: true,
		},
		{
			name: "error statuses with same message should be equal",
			x:    NewStatus(Error, "error"),
			y:    NewStatus(Error, "error"),
			want: true,
		},
		{
			name: "statuses with different reasons should not be equal",
			x:    NewStatus(Unschedulable, "unschedulable"),
			y:    NewStatus(Unschedulable, "unschedulable", "injected filter status"),
			want: false,
		},
		{
			name: "statuses with different codes should not be equal",
			x:    NewStatus(Error, "internal error"),
			y:    NewStatus(Unschedulable, "internal error"),
			want: false,
		},
		{
			name: "wrap error status should be equal with original one",
			x:    statusWithErr,
			y:    AsStatus(fmt.Errorf("error: %w", statusWithErr.AsError())),
			want: true,
		},
		{
			name: "statues with different errors that have the same message shouldn't be equal",
			x:    AsStatus(errors.New("error")),
			y:    AsStatus(errors.New("error")),
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.x.Equal(tt.y); got != tt.want {
				t.Errorf("cmp.Equal() = %v, want %v", got, tt.want)
			}
		})
	}
}

type nodeInfoLister []*NodeInfo

func (nodes nodeInfoLister) Get(nodeName string) (*NodeInfo, error) {
	for _, node := range nodes {
		if node != nil && node.Node().Name == nodeName {
			return node, nil
		}
	}
	return nil, fmt.Errorf("unable to find node: %s", nodeName)
}

func (nodes nodeInfoLister) List() ([]*NodeInfo, error) {
	return nodes, nil
}

func (nodes nodeInfoLister) HavePodsWithAffinityList() ([]*NodeInfo, error) {
	return nodes, nil
}

func (nodes nodeInfoLister) HavePodsWithRequiredAntiAffinityList() ([]*NodeInfo, error) {
	return nodes, nil
}

func TestNodesForStatusCode(t *testing.T) {
	// Prepare 4 nodes names.
	nodeNames := []string{"node1", "node2", "node3", "node4"}
	tests := []struct {
		name          string
		nodesStatuses *NodeToStatus
		code          Code
		expected      sets.Set[string] // set of expected node names.
	}{
		{
			name: "No node should be attempted",
			nodesStatuses: NewNodeToStatus(map[string]*Status{
				"node1": NewStatus(UnschedulableAndUnresolvable),
				"node2": NewStatus(UnschedulableAndUnresolvable),
				"node3": NewStatus(UnschedulableAndUnresolvable),
				"node4": NewStatus(UnschedulableAndUnresolvable),
			}, NewStatus(UnschedulableAndUnresolvable)),
			code:     Unschedulable,
			expected: sets.New[string](),
		},
		{
			name: "All nodes should be attempted",
			nodesStatuses: NewNodeToStatus(map[string]*Status{
				"node1": NewStatus(UnschedulableAndUnresolvable),
				"node2": NewStatus(UnschedulableAndUnresolvable),
				"node3": NewStatus(UnschedulableAndUnresolvable),
				"node4": NewStatus(UnschedulableAndUnresolvable),
			}, NewStatus(UnschedulableAndUnresolvable)),
			code:     UnschedulableAndUnresolvable,
			expected: sets.New[string]("node1", "node2", "node3", "node4"),
		},
		{
			name:          "No node should be attempted, as all are implicitly not matching the code",
			nodesStatuses: NewDefaultNodeToStatus(),
			code:          Unschedulable,
			expected:      sets.New[string](),
		},
		{
			name:          "All nodes should be attempted, as all are implicitly matching the code",
			nodesStatuses: NewDefaultNodeToStatus(),
			code:          UnschedulableAndUnresolvable,
			expected:      sets.New[string]("node1", "node2", "node3", "node4"),
		},
		{
			name: "UnschedulableAndUnresolvable status should be skipped but Unschedulable should be tried",
			nodesStatuses: NewNodeToStatus(map[string]*Status{
				"node1": NewStatus(Unschedulable),
				"node2": NewStatus(UnschedulableAndUnresolvable),
				"node3": NewStatus(Unschedulable),
				// node4 is UnschedulableAndUnresolvable by absence
			}, NewStatus(UnschedulableAndUnresolvable)),
			code:     Unschedulable,
			expected: sets.New("node1", "node3"),
		},
		{
			name: "Unschedulable status should be skipped but UnschedulableAndUnresolvable should be tried",
			nodesStatuses: NewNodeToStatus(map[string]*Status{
				"node1": NewStatus(Unschedulable),
				"node2": NewStatus(UnschedulableAndUnresolvable),
				"node3": NewStatus(Unschedulable),
				// node4 is UnschedulableAndUnresolvable by absence
			}, NewStatus(UnschedulableAndUnresolvable)),
			code:     UnschedulableAndUnresolvable,
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
