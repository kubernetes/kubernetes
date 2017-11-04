/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNodesHaveSameTopologyKey(t *testing.T) {
	tests := []struct {
		name         string
		nodeA, nodeB *v1.Node
		topologyKey  string
		expected     bool
	}{
		{
			name: "nodeA{'a':'a'} vs. empty label in nodeB",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			nodeB:       &v1.Node{},
			expected:    false,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':'a'} vs. nodeB{'a':'a'}",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			expected:    true,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':''} vs. empty label in nodeB",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			nodeB:       &v1.Node{},
			expected:    false,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':''} vs. nodeB{'a':''}",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "",
					},
				},
			},
			expected:    true,
			topologyKey: "a",
		},
		{
			name: "nodeA{'a':'a'} vs. nodeB{'a':'a'} by key{'b'}",
			nodeA: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			nodeB: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"a": "a",
					},
				},
			},
			expected:    false,
			topologyKey: "b",
		},
	}

	for _, test := range tests {
		got := NodesHaveSameTopologyKey(test.nodeA, test.nodeB, test.topologyKey)
		if test.expected != got {
			t.Errorf("%v: expected %t, got %t", test.name, test.expected, got)
		}
	}
}
