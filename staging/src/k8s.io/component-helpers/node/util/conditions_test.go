/*
Copyright 2026 The Kubernetes Authors.

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

package util

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestIsNodeInGracefulShutdown(t *testing.T) {
	testCases := []struct {
		name string
		node *v1.Node
		want bool
	}{
		{
			name: "nil node",
		},
		{
			name: "ready node",
			node: nodeWithReadyCondition(v1.ConditionTrue, "KubeletReady", "kubelet is posting ready status"),
		},
		{
			name: "not ready without shutdown signal",
			node: nodeWithReadyCondition(v1.ConditionFalse, "KubeletNotReady", "runtime error"),
		},
		{
			name: "shutdown by message",
			node: nodeWithReadyCondition(v1.ConditionFalse, "KubeletNotReady", NodeShutdownMessage),
			want: true,
		},
		{
			name: "shutdown by reason",
			node: nodeWithReadyCondition(v1.ConditionFalse, NodeShutdownMessage, ""),
			want: true,
		},
		{
			name: "shutdown message with aggregated errors",
			node: nodeWithReadyCondition(v1.ConditionFalse, "KubeletNotReady", "[runtime error, "+NodeShutdownMessage+"]"),
			want: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := IsNodeInGracefulShutdown(tc.node); got != tc.want {
				t.Fatalf("IsNodeInGracefulShutdown() = %v, want %v", got, tc.want)
			}
		})
	}
}

func nodeWithReadyCondition(status v1.ConditionStatus, reason, message string) *v1.Node {
	return &v1.Node{
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:    v1.NodeReady,
					Status:  status,
					Reason:  reason,
					Message: message,
				},
			},
		},
	}
}
