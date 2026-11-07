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
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func makeExtraNodeAllocatableResource(containerName string) v1.ExtraNodeAllocatableResource {
	return v1.ExtraNodeAllocatableResource{
		ContainerName: containerName,
		Requests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
		Limits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
	}
}

func TestExtraNodeAllocatableResourcesState(t *testing.T) {
	const node = "node-1"
	a1 := makeExtraNodeAllocatableResource("a1")
	a2 := makeExtraNodeAllocatableResource("a2")
	b1 := makeExtraNodeAllocatableResource("b1")
	c1 := makeExtraNodeAllocatableResource("c1")

	tests := []struct {
		name                   string
		setup                  func(state *ExtraNodeAllocatableResourcesState)
		wantGet                []v1.ExtraNodeAllocatableResource
		wantGetExcludingPlugin []v1.ExtraNodeAllocatableResource
		wantGetFromPlugin      []v1.ExtraNodeAllocatableResource
	}{
		{
			name: "empty store returns nothing",
		},
		{
			name: "set records the plugin's entries",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
			},
			wantGet:           []v1.ExtraNodeAllocatableResource{a1},
			wantGetFromPlugin: []v1.ExtraNodeAllocatableResource{a1},
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
			wantGet:           []v1.ExtraNodeAllocatableResource{a2},
			wantGetFromPlugin: []v1.ExtraNodeAllocatableResource{a2},
		},
		{
			name: "set without entries removes the plugin's entries",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginA", nil)
			},
			wantGet:           nil,
			wantGetFromPlugin: nil,
		},
		{
			name: "entries are sorted by plugin name and keep each plugin's order",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginC", []v1.ExtraNodeAllocatableResource{c1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{b1})
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a2, a1})
			},
			wantGet:                []v1.ExtraNodeAllocatableResource{a2, a1, b1, c1},
			wantGetExcludingPlugin: []v1.ExtraNodeAllocatableResource{b1, c1},
			wantGetFromPlugin:      []v1.ExtraNodeAllocatableResource{a2, a1},
		},
		{
			name: "entries of different plugins for the same container are added up",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{makeExtraNodeAllocatableResource(""), c1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{
					makeExtraNodeAllocatableResource(""),
					{ContainerName: "c1", Limits: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")}},
					b1,
				})
			},
			wantGet: []v1.ExtraNodeAllocatableResource{
				{
					ContainerName: "",
					Requests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
					Limits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
				},
				{
					ContainerName: "c1",
					Requests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					Limits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("1Gi")},
				},
				b1,
			},
			wantGetExcludingPlugin: []v1.ExtraNodeAllocatableResource{
				makeExtraNodeAllocatableResource(""),
				{ContainerName: "c1", Limits: v1.ResourceList{v1.ResourceMemory: resource.MustParse("1Gi")}},
				b1,
			},
			wantGetFromPlugin: []v1.ExtraNodeAllocatableResource{makeExtraNodeAllocatableResource(""), c1},
		},
		{
			name: "removing one plugin's entries keeps the others",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{b1})
				state.Set(node, "PluginA", nil)
			},
			wantGet:                []v1.ExtraNodeAllocatableResource{b1},
			wantGetExcludingPlugin: []v1.ExtraNodeAllocatableResource{b1},
		},
		{
			name: "changing the set input does not change the store",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				entries := []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()}
				state.Set(node, "PluginA", entries)
				entries[0].ContainerName = "changed"
				entries[0].Requests[v1.ResourceCPU] = resource.MustParse("2")
			},
			wantGet:           []v1.ExtraNodeAllocatableResource{a1},
			wantGetFromPlugin: []v1.ExtraNodeAllocatableResource{a1},
		},
		{
			name: "changing a get result does not change the store",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()})
				got := state.Get(node)
				got[0].ContainerName = "changed"
				got[0].Requests[v1.ResourceCPU] = resource.MustParse("2")
			},
			wantGet:           []v1.ExtraNodeAllocatableResource{a1},
			wantGetFromPlugin: []v1.ExtraNodeAllocatableResource{a1},
		},
		{
			name: "changing a GetFromPlugin or GetPersisted result does not change the store",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()})
				_ = state.UpdatePersisted(node, "PluginA", func([]v1.ExtraNodeAllocatableResource) error { return nil })
				for _, got := range [][]v1.ExtraNodeAllocatableResource{state.GetFromPlugin(node, "PluginA"), state.GetPersisted(node)} {
					got[0].ContainerName = "changed"
					got[0].Requests[v1.ResourceCPU] = resource.MustParse("2")
				}
			},
			wantGet:           []v1.ExtraNodeAllocatableResource{a1},
			wantGetFromPlugin: []v1.ExtraNodeAllocatableResource{a1},
		},
		{
			name: "adding up entries does not change the store",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{*a1.DeepCopy()})
				_ = state.Get(node)
			},
			wantGet: []v1.ExtraNodeAllocatableResource{{
				ContainerName: "a1",
				Requests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
				Limits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
			}},
			wantGetExcludingPlugin: []v1.ExtraNodeAllocatableResource{a1},
			wantGetFromPlugin:      []v1.ExtraNodeAllocatableResource{a1},
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
			if diff := cmp.Diff(tc.wantGetExcludingPlugin, state.GetExcludingPlugin(node, "PluginA")); diff != "" {
				t.Errorf("GetExcludingPlugin() mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantGetFromPlugin, state.GetFromPlugin(node, "PluginA")); diff != "" {
				t.Errorf("GetFromPlugin() mismatch (-want +got):\n%s", diff)
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
			_ = state.GetFromPlugin(node, "PluginA")
			_ = state.UpdatePersisted(node, "PluginA", func([]v1.ExtraNodeAllocatableResource) error { return nil })
			_ = state.GetPersisted(node)
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

func TestExtraNodeAllocatableResourcesStatePersisted(t *testing.T) {
	const node = "node-1"
	// persist marks the entries of pluginName as persisted, as the PreBind of the plugin does once it wrote them.
	persist := func(state *ExtraNodeAllocatableResourcesState, nodeName, pluginName string) {
		_ = state.UpdatePersisted(nodeName, pluginName, func([]v1.ExtraNodeAllocatableResource) error { return nil })
	}
	a1 := makeExtraNodeAllocatableResource("a1")
	b1 := makeExtraNodeAllocatableResource("b1")
	c1 := makeExtraNodeAllocatableResource("c1")

	tests := []struct {
		name          string
		setup         func(state *ExtraNodeAllocatableResourcesState)
		wantPersisted []v1.ExtraNodeAllocatableResource
	}{
		{
			name: "nothing persisted",
		},
		{
			name: "recorded entries are not persisted",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
			},
		},
		{
			name: "entries of persisted plugins, in plugin name order",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{b1})
				state.Set(node, "PluginC", []v1.ExtraNodeAllocatableResource{c1})
				persist(state, node, "PluginB")
				persist(state, node, "PluginA")
			},
			wantPersisted: []v1.ExtraNodeAllocatableResource{a1, b1},
		},
		{
			name: "entries of persisted plugins for the same container are added up",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{a1})
				persist(state, node, "PluginA")
				persist(state, node, "PluginB")
			},
			wantPersisted: []v1.ExtraNodeAllocatableResource{{
				ContainerName: "a1",
				Requests:      v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
				Limits:        v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
			}},
		},
		{
			name: "entries of another node are not returned",
			setup: func(state *ExtraNodeAllocatableResourcesState) {
				state.Set("node-2", "PluginA", []v1.ExtraNodeAllocatableResource{a1})
				persist(state, "node-2", "PluginA")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			state := NewExtraNodeAllocatableResourcesState()
			if tc.setup != nil {
				tc.setup(state)
			}
			if diff := cmp.Diff(tc.wantPersisted, state.GetPersisted(node)); diff != "" {
				t.Errorf("GetPersisted() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtraNodeAllocatableResourcesStateUpdatePersisted(t *testing.T) {
	const node = "node-1"
	a1 := makeExtraNodeAllocatableResource("a1")
	b1 := makeExtraNodeAllocatableResource("b1")

	tests := []struct {
		name          string
		updateErr     error
		wantUpdateGot []v1.ExtraNodeAllocatableResource
		wantPersisted []v1.ExtraNodeAllocatableResource
	}{
		{
			name:          "update gets the entries of persisted plugins and the plugin's entries become persisted",
			wantUpdateGot: []v1.ExtraNodeAllocatableResource{a1},
			wantPersisted: []v1.ExtraNodeAllocatableResource{a1, b1},
		},
		{
			name:          "the plugin's entries do not become persisted if update fails",
			updateErr:     errors.New("update failed"),
			wantUpdateGot: []v1.ExtraNodeAllocatableResource{a1},
			wantPersisted: []v1.ExtraNodeAllocatableResource{a1},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			state := NewExtraNodeAllocatableResourcesState()
			state.Set(node, "PluginA", []v1.ExtraNodeAllocatableResource{a1})
			state.Set(node, "PluginB", []v1.ExtraNodeAllocatableResource{b1})
			_ = state.UpdatePersisted(node, "PluginA", func([]v1.ExtraNodeAllocatableResource) error { return nil })

			var updateGot []v1.ExtraNodeAllocatableResource
			err := state.UpdatePersisted(node, "PluginB", func(persisted []v1.ExtraNodeAllocatableResource) error {
				updateGot = persisted
				return tc.updateErr
			})
			if !errors.Is(err, tc.updateErr) {
				t.Errorf("UpdatePersisted() error = %v, want %v", err, tc.updateErr)
			}
			if diff := cmp.Diff(tc.wantUpdateGot, updateGot); diff != "" {
				t.Errorf("update got unexpected persisted entries (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantPersisted, state.GetPersisted(node)); diff != "" {
				t.Errorf("GetPersisted() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestExtraNodeAllocatableResourcesStateUpdatePersistedParallel(t *testing.T) {
	const node = "node-1"
	const numPlugins = 10
	state := NewExtraNodeAllocatableResourcesState()
	var want []v1.ExtraNodeAllocatableResource
	for i := range numPlugins {
		entry := makeExtraNodeAllocatableResource(fmt.Sprintf("c%d", i))
		state.Set(node, fmt.Sprintf("Plugin%d", i), []v1.ExtraNodeAllocatableResource{entry})
		want = append(want, entry)
	}

	// podStatus stands for the field in the pod status. Each write replaces it, as in PreBind.
	var podStatus []v1.ExtraNodeAllocatableResource
	var wg sync.WaitGroup
	for i := range numPlugins {
		wg.Go(func() {
			pluginName := fmt.Sprintf("Plugin%d", i)
			entries := state.GetFromPlugin(node, pluginName)
			if err := state.UpdatePersisted(node, pluginName, func(persisted []v1.ExtraNodeAllocatableResource) error {
				podStatus = MergeExtraNodeAllocatableResources(persisted, entries)
				return nil
			}); err != nil {
				t.Errorf("UpdatePersisted() error = %v", err)
			}
		})
	}
	wg.Wait()

	// Each write kept what the earlier ones wrote, so the last one has the entries of all plugins.
	slices.SortFunc(podStatus, func(a, b v1.ExtraNodeAllocatableResource) int {
		return strings.Compare(a.ContainerName, b.ContainerName)
	})
	if diff := cmp.Diff(want, podStatus); diff != "" {
		t.Errorf("pod status mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(want, state.GetPersisted(node)); diff != "" {
		t.Errorf("GetPersisted() mismatch (-want +got):\n%s", diff)
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
