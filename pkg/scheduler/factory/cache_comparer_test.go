/*
Copyright 2018 The Kubernetes Authors.

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

package factory

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
)

func TestCompareNodes(t *testing.T) {
	compare := compareStrategy{}

	tests := []struct {
		actual    []string
		cached    []string
		missing   []string
		redundant []string
	}{
		{
			actual:    []string{"foo", "bar"},
			cached:    []string{"bar", "foo", "foobar"},
			missing:   []string{},
			redundant: []string{"foobar"},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foo"},
			missing:   []string{"foobar"},
			redundant: []string{},
		},
		{
			actual:    []string{"foo", "bar", "foobar"},
			cached:    []string{"bar", "foobar", "foo"},
			missing:   []string{},
			redundant: []string{},
		},
	}

	for _, test := range tests {
		nodes := []*v1.Node{}
		for _, nodeName := range test.actual {
			node := &v1.Node{}
			node.Name = nodeName
			nodes = append(nodes, node)
		}

		nodeInfo := make(map[string]*schedulercache.NodeInfo)
		for _, nodeName := range test.cached {
			nodeInfo[nodeName] = &schedulercache.NodeInfo{}
		}

		m, r := compare.CompareNodes(nodes, nodeInfo)

		if !reflect.DeepEqual(m, test.missing) {
			t.Errorf("missing expected to be %s; got %s", test.missing, m)
		}

		if !reflect.DeepEqual(r, test.redundant) {
			t.Errorf("redundant expected to be %s; got %s", test.redundant, r)
		}
	}
}
