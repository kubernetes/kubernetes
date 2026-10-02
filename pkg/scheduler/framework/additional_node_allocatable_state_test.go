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

package framework

import (
	"fmt"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func makeAdditionalNodeAllocatableResource(name, cpu string) v1.AdditionalNodeAllocatableResource {
	return v1.AdditionalNodeAllocatableResource{
		Source:     v1.AdditionalNodeAllocatableReference{APIGroup: "resource.k8s.io", Kind: "ResourceClaim", Name: name},
		Containers: []string{"c1"},
		Mapping: []v1.NodeAllocatableMappedResources{{
			Name:     v1.ResourceCPU,
			Quantity: new(resource.MustParse(cpu)),
		}},
	}
}

func TestAdditionalNodeAllocatableResourcesStateSetGet(t *testing.T) {
	a := makeAdditionalNodeAllocatableResource("claim-a", "1")
	b := makeAdditionalNodeAllocatableResource("claim-b", "2")
	c := makeAdditionalNodeAllocatableResource("claim-c", "3")

	tests := []struct {
		name  string
		ops   func(s *AdditionalNodeAllocatableResourcesState)
		node  string
		want  []v1.AdditionalNodeAllocatableResource
		want2 bool
	}{
		{
			name: "empty store",
			ops:  func(s *AdditionalNodeAllocatableResourcesState) {},
			node: "node-1",
		},
		{
			name: "single plugin",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{a})
			},
			node:  "node-1",
			want:  []v1.AdditionalNodeAllocatableResource{a},
			want2: true,
		},
		{
			name: "other node is not visible",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-2", "PluginA", []v1.AdditionalNodeAllocatableResource{a})
			},
			node: "node-1",
		},
		{
			name: "set replaces the same plugin's contribution",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{a})
				s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{b})
			},
			node:  "node-1",
			want:  []v1.AdditionalNodeAllocatableResource{b},
			want2: true,
		},
		{
			name: "empty set removes the contribution",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{a})
				s.Set("node-1", "PluginA", nil)
			},
			node: "node-1",
		},
		{
			name: "empty set on unknown node is a no-op",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-1", "PluginA", nil)
			},
			node: "node-1",
		},
		{
			name: "multiple plugins accumulate on the same node",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-1", "PluginB", []v1.AdditionalNodeAllocatableResource{c})
				s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{a, b})
			},
			node:  "node-1",
			want:  []v1.AdditionalNodeAllocatableResource{a, b, c},
			want2: true,
		},
		{
			name: "removing one plugin keeps the others",
			ops: func(s *AdditionalNodeAllocatableResourcesState) {
				s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{a})
				s.Set("node-1", "PluginB", []v1.AdditionalNodeAllocatableResource{c})
				s.Set("node-1", "PluginA", nil)
			},
			node:  "node-1",
			want:  []v1.AdditionalNodeAllocatableResource{c},
			want2: true,
		},
	}

	sortByClaimName := cmpopts.SortSlices(func(x, y v1.AdditionalNodeAllocatableResource) bool {
		return x.Source.Name < y.Source.Name
	})
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := NewAdditionalNodeAllocatableResourcesState()
			tc.ops(s)
			if diff := cmp.Diff(tc.want, s.Get(tc.node), sortByClaimName); diff != "" {
				t.Errorf("Get() mismatch (-want +got):\n%s", diff)
			}
			if got := s.Has(tc.node); got != tc.want2 {
				t.Errorf("Has() = %v, want %v", got, tc.want2)
			}
		})
	}
}

func TestAdditionalNodeAllocatableResourcesStateGetReturnsCopy(t *testing.T) {
	original := makeAdditionalNodeAllocatableResource("claim-a", "1")
	s := NewAdditionalNodeAllocatableResourcesState()
	s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{*original.DeepCopy()})

	got := s.Get("node-1")
	got[0].Source.Name = "mutated-by-caller"
	got[0].Mapping[0].Quantity.Add(resource.MustParse("1"))

	if diff := cmp.Diff([]v1.AdditionalNodeAllocatableResource{original}, s.Get("node-1")); diff != "" {
		t.Errorf("mutating Get() result changed the store (-want +got):\n%s", diff)
	}
}

func TestAdditionalNodeAllocatableResourcesStateNilSafe(t *testing.T) {
	var s *AdditionalNodeAllocatableResourcesState
	s.Set("node-1", "PluginA", []v1.AdditionalNodeAllocatableResource{makeAdditionalNodeAllocatableResource("claim-a", "1")})
	if got := s.Get("node-1"); got != nil {
		t.Errorf("Get() on nil store = %v, want nil", got)
	}
	if s.Has("node-1") {
		t.Error("Has() on nil store = true, want false")
	}
}

func TestGetAdditionalNodeAllocatableResourcesState(t *testing.T) {
	if got := GetAdditionalNodeAllocatableResourcesState(nil); got != nil {
		t.Errorf("GetAdditionalNodeAllocatableResourcesState(nil) = %v, want nil", got)
	}

	cs := NewCycleState()
	if got := GetAdditionalNodeAllocatableResourcesState(cs); got != nil {
		t.Errorf("GetAdditionalNodeAllocatableResourcesState() before create = %v, want nil", got)
	}

	created := GetOrCreateAdditionalNodeAllocatableResourcesState(cs)
	if created == nil {
		t.Fatal("GetOrCreateAdditionalNodeAllocatableResourcesState() = nil")
	}
	if got := GetOrCreateAdditionalNodeAllocatableResourcesState(cs); got != created {
		t.Error("GetOrCreateAdditionalNodeAllocatableResourcesState() replaced an existing store")
	}
	if got := GetAdditionalNodeAllocatableResourcesState(cs); got != created {
		t.Error("GetAdditionalNodeAllocatableResourcesState() did not return the created store")
	}

	if got := GetAdditionalNodeAllocatableResourcesState(cs.Clone()); got != created {
		t.Error("store in cloned CycleState is not the same object")
	}
}

func TestAdditionalNodeAllocatableResourcesStateConcurrentNodes(t *testing.T) {
	const numNodes = 100
	s := NewAdditionalNodeAllocatableResourcesState()

	// One goroutine per node, as in Filter.
	var wg sync.WaitGroup
	for i := range numNodes {
		wg.Go(func() {
			node := fmt.Sprintf("node-%d", i)
			s.Set(node, "PluginA", []v1.AdditionalNodeAllocatableResource{makeAdditionalNodeAllocatableResource(node, "1")})
			_ = s.Get(node)
			_ = s.Has(fmt.Sprintf("node-%d", (i+1)%numNodes))
		})
	}
	wg.Wait()

	for i := range numNodes {
		node := fmt.Sprintf("node-%d", i)
		got := s.Get(node)
		if len(got) != 1 || got[0].Source.Name != node {
			t.Errorf("Get(%q) = %v, want one entry for %q", node, got, node)
		}
	}
}
