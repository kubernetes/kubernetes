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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func makeExtraNodeAllocatableResource(claimName string) v1.ExtraNodeAllocatableResource {
	return v1.ExtraNodeAllocatableResource{
		Source:     v1.ExtraNodeAllocatableReference{APIGroup: "resource.k8s.io", Kind: "ResourceClaim", Name: claimName},
		Containers: []string{"c1"},
		Mapping: []v1.NodeAllocatableMappedResources{{
			Name:     v1.ResourceCPU,
			Quantity: new(resource.MustParse("1")),
		}},
	}
}

func TestExtraNodeAllocatableResourcesState(t *testing.T) {
	const node = "node-1"
	a1 := makeExtraNodeAllocatableResource("a1")
	a2 := makeExtraNodeAllocatableResource("a2")
	b1 := makeExtraNodeAllocatableResource("b1")
	c1 := makeExtraNodeAllocatableResource("c1")

	tests := []struct {
		name                    string
		setup                   func(state *ExtraNodeAllocatableResourcesState)
		wantGet                 []v1.ExtraNodeAllocatableResource
		wantGetExcludingPluginA []v1.ExtraNodeAllocatableResource
	}{
		{
			name: "empty store returns nothing",
		},
		{
			name: "set records the plugin's entries",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
			},
			wantGet: []v1.ExtraNodeAllocatableResource{a1},
		},
		{
			name: "entries of another node are not returned",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set("node-2", "PluginA", []v1.ExtraNodeAllocatableResource{a1})
			},
		},
		{
			name: "set replaces the plugin's entries",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a2})
			},
			wantGet: []v1.ExtraNodeAllocatableResource{a2},
		},
		{
			name: "set without entries removes the plugin's entries",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginA", nil)
			},
		},
		{
			name: "entries are sorted by plugin name and keep each plugin's order",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginC", []v1.ExtraNodeAllocatableResource{c1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{b1})
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a2, a1})
			},
			wantGet:                 []v1.ExtraNodeAllocatableResource{a2, a1, b1, c1},
			wantGetExcludingPluginA: []v1.ExtraNodeAllocatableResource{b1, c1},
		},
		{
			name: "removing one plugin's entries keeps the others",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{b1})
				state.Set(node, "PluginA", nil)
			},
			wantGet:                 []v1.ExtraNodeAllocatableResource{b1},
			wantGetExcludingPluginA: []v1.ExtraNodeAllocatableResource{b1},
		},
		{
			name: "changing the set input does not change the store",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				entries := []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()}
				state.Set(node, "PluginA", entries)
				entries[0].Source.Name = "changed"
				entries[0].Mapping[0].Quantity.Add(resource.MustParse("1"))
			},
			wantGet: []v1.ExtraNodeAllocatableResource{a1},
		},
		{
			name: "changing a get result does not change the store",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()})
				got := state.Get(node)
				got[0].Source.Name = "changed"
				got[0].Mapping[0].Quantity.Add(resource.MustParse("1"))
			},
			wantGet: []v1.ExtraNodeAllocatableResource{a1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			state := NewExtraNodeAllocatableResourcesState()
			if tc.setup != nil {
				tc.setup(state)
			}
			if diff := cmp.Diff(tc.wantGet, state.Get(node)); diff != "" {
				t.Errorf("Get() mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantGetExcludingPluginA, state.GetExcludingPlugin(node, "PluginA")); diff != "" {
				t.Errorf("GetExcludingPlugin() mismatch (-want +got):\n%s", diff)
			}
			if got, want := state.Has(node), len(tc.wantGet) > 0; got != want {
				t.Errorf("Has() = %v, want %v", got, want)
			}
		})
	}
}

// TestExtraNodeAllocatableResourcesStateConcurrent relies on the race detector
// (make test runs unit tests with -race). The goroutines only create concurrent
// access to the shared map; if a method skips the lock, -race fails the test.
func TestExtraNodeAllocatableResourcesStateConcurrent(t *testing.T) {
	const numNodes = 100
	state := NewExtraNodeAllocatableResourcesState()

	// One goroutine per node, as in Filter.
	var wg sync.WaitGroup
	for i := range numNodes {
		wg.Go(func() {
			node := fmt.Sprintf("node-%d", i)
			state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{makeExtraNodeAllocatableResource(node)})
			_ = state.Get(node)
			_ = state.GetExcludingPlugin(node, "PluginA")
			_ = state.Has(fmt.Sprintf("node-%d", (i+1)%numNodes))
		})
	}
	wg.Wait()

	for i := range numNodes {
		node := fmt.Sprintf("node-%d", i)
		if diff := cmp.Diff([]v1.ExtraNodeAllocatableResource{makeExtraNodeAllocatableResource(node)}, state.Get(node)); diff != "" {
			t.Errorf("Get(%q) mismatch (-want +got):\n%s", node, diff)
		}
	}
}

func TestGetExtraNodeAllocatableResourcesState(t *testing.T) {
	if got := GetExtraNodeAllocatableResourcesState(nil); got != nil {
		t.Errorf("GetExtraNodeAllocatableResourcesState(nil) = %v, want nil", got)
	}
	cs := NewCycleState()
	if got := GetExtraNodeAllocatableResourcesState(cs); got != nil {
		t.Errorf("GetExtraNodeAllocatableResourcesState() before create = %v, want nil", got)
	}
	created := GetOrCreateExtraNodeAllocatableResourcesState(cs)
	if created == nil {
		t.Fatal("GetOrCreateExtraNodeAllocatableResourcesState() = nil")
	}
	if got := GetOrCreateExtraNodeAllocatableResourcesState(cs); got != created {
		t.Error("GetOrCreateExtraNodeAllocatableResourcesState() replaced an existing store")
	}
	if got := GetExtraNodeAllocatableResourcesState(cs); got != created {
		t.Error("GetExtraNodeAllocatableResourcesState() did not return the created store")
	}
	if got := GetExtraNodeAllocatableResourcesState(cs.Clone()); got != created {
		t.Error("store in cloned CycleState is not the same object")
	}
}
