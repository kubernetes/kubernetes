/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
)

func TestNodesHaveSameTopologyKeyInternal(t *testing.T) {
	for i, test := range []struct {
		labelsA, labelsB map[string]string
		equal            bool
	}{
		{
			labelsA: nil,
			labelsB: nil,
			equal:   true,
		},
		{
			labelsA: nil,
			labelsB: map[string]string{"foo": "0"},
			equal:   false,
		},
		{
			labelsA: nil,
			labelsB: map[string]string{"bar": "0"},
			equal:   true,
		},
		{
			labelsA: map[string]string{"abc": "def"},
			labelsB: map[string]string{"bar": "0"},
			equal:   true,
		},
		{
			labelsA: map[string]string{"abc": "def"},
			labelsB: map[string]string{"foo": "0"},
			equal:   false,
		},
		{
			labelsA: map[string]string{"foo": "1"},
			labelsB: map[string]string{"foo": "0"},
			equal:   false,
		},
		{
			labelsA: map[string]string{"foo": "0"},
			labelsB: map[string]string{"foo": "0"},
			equal:   true,
		},
	} {
		nodeA := api.Node{
			ObjectMeta: api.ObjectMeta{
				Labels: test.labelsA,
			},
		}
		nodeB := api.Node{
			ObjectMeta: api.ObjectMeta{
				Labels: test.labelsB,
			},
		}
		if got, expected := nodesHaveSameTopologyKeyInternal(&nodeA, &nodeB, "foo"), test.equal; got != expected {
			t.Fatalf("test %d: wrong result for A==B: got=%v expected=%v", i, got, expected)
		}
		if got, expected := nodesHaveSameTopologyKeyInternal(&nodeB, &nodeA, "foo"), test.equal; got != expected {
			t.Fatalf("test %d: wrong result for B==A: got=%v expected=%v", i, got, expected)
		}
	}
}
