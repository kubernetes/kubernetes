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

package runtime

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type mergeData struct {
	value string
}

func (d *mergeData) Clone() fwk.StateData {
	return &mergeData{value: d.value}
}

// mergeTestPlugin is a PlacementGeneratePlugin used to exercise multi-plugin merging.
type mergeTestPlugin struct {
	name          string
	placements    []*fwk.Placement
	returnParent  bool
	includeParent bool
	statusCode    fwk.Code
	parent        *fwk.Placement
	parentState   map[fwk.StateKey]fwk.StateData
	// state maps placement pointer -> (state key -> value) written during generation.
	state map[*fwk.Placement]map[fwk.StateKey]fwk.StateData
}

func (p *mergeTestPlugin) Name() string { return p.name }

func (p *mergeTestPlugin) GeneratePlacements(_ context.Context, state fwk.PodGroupCycleState, _ fwk.PodGroupInfo, parent *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	if p.statusCode != fwk.Success {
		return nil, fwk.NewStatus(p.statusCode, "injected")
	}
	p.parent = parent
	if len(p.parentState) > 0 {
		parentState := state.PlacementState(parent)
		for k, v := range p.parentState {
			parentState.Write(k, v)
		}
	}
	if p.returnParent {
		return &fwk.GeneratePlacementsResult{Placements: []*fwk.Placement{parent}}, nil
	}
	for pl, kvs := range p.state {
		ps := state.PlacementState(pl)
		for k, v := range kvs {
			ps.Write(k, v)
		}
	}
	placements := p.placements
	if p.includeParent {
		placements = append([]*fwk.Placement{parent}, placements...)
	}
	return &fwk.GeneratePlacementsResult{Placements: placements}, nil
}

func buildMergeFramework(t *testing.T, ctx context.Context, plugins []*mergeTestPlugin) framework.Framework {
	t.Helper()
	r := make(Registry)
	pluginSet := config.PluginSet{}
	for _, pl := range plugins {
		pluginSet.Enabled = append(pluginSet.Enabled, config.Plugin{Name: pl.name})
		if err := r.Register(pl.name, func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
			return pl, nil
		}); err != nil {
			t.Fatalf("failed to register plugin %q: %v", pl.name, err)
		}
	}
	profile := config.KubeSchedulerProfile{Plugins: &config.Plugins{PlacementGenerate: pluginSet}}
	fw, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
	if err != nil {
		t.Fatalf("unexpected error building framework: %v", err)
	}
	return fw
}

func TestRunPlacementGeneratePluginsMerge(t *testing.T) {
	nodeNames := []string{"n1", "n2", "n3", "n4"}
	nodes := make(map[string]fwk.NodeInfo, len(nodeNames))
	allNodes := make([]fwk.NodeInfo, 0, len(nodeNames))
	for _, name := range nodeNames {
		ni := framework.NewNodeInfo()
		ni.SetNode(st.MakeNode().Name(name).Obj())
		nodes[name] = ni
		allNodes = append(allNodes, ni)
	}
	placement := func(name string, ns ...string) *fwk.Placement {
		p := &fwk.Placement{Name: name}
		for _, n := range ns {
			p.Nodes = append(p.Nodes, nodes[n])
		}
		return p
	}

	tests := map[string]struct {
		plugins        []*mergeTestPlugin
		wantStatusCode fwk.Code
		wantPlacements []*fwk.Placement
	}{
		"two single-placement plugins intersect": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a", "n1", "n2", "n3")}},
				{name: "b", placements: []*fwk.Placement{placement("b", "n2", "n3", "n4")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("a/b", "n2", "n3")},
		},
		"cross product drops empty intersections": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a1", "n1", "n2"), placement("a2", "n3", "n4")}},
				{name: "b", placements: []*fwk.Placement{placement("b1", "n2", "n3")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("a1/b1", "n2"), placement("a2/b1", "n3")},
		},
		"three plugins fold-merge": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a", "n1", "n2", "n3")}},
				{name: "b", placements: []*fwk.Placement{placement("b", "n2", "n3", "n4")}},
				{name: "c", placements: []*fwk.Placement{placement("c", "n3", "n4")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("a/b/c", "n3")},
		},
		"no overlap is unschedulable": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a", "n1")}},
				{name: "b", placements: []*fwk.Placement{placement("b", "n2")}},
			},
			wantStatusCode: fwk.Unschedulable,
		},
		"unconstrained plugin is skipped": {
			plugins: []*mergeTestPlugin{
				{name: "a", returnParent: true},
				{name: "b", placements: []*fwk.Placement{placement("b", "n2", "n3")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("b", "n2", "n3")},
		},
		"all unconstrained returns input placement": {
			plugins: []*mergeTestPlugin{
				{name: "a", returnParent: true},
				{name: "b", returnParent: true},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("", "n1", "n2", "n3", "n4")},
		},
		"same placement name from different plugins succeeds": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("dup", "n1", "n2")}},
				{name: "b", placements: []*fwk.Placement{placement("dup", "n2", "n3")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("dup/dup", "n2")},
		},
		"empty placement name succeeds": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("zone-a", "n1", "n2")}},
				{name: "b", placements: []*fwk.Placement{placement("", "n2", "n3")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("zone-a/", "n2")},
		},
		"placement name containing the separator is an error": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("zone-a/rack-1", "n1", "n2")}},
			},
			wantStatusCode: fwk.Error,
		},
		"duplicate placement name from one plugin is an error": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("dup", "n1"), placement("dup", "n2")}},
			},
			wantStatusCode: fwk.Error,
		},
		"nil placement is an error": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{nil}},
			},
			wantStatusCode: fwk.Error,
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			fw := buildMergeFramework(t, ctx, tt.plugins)
			got, status := fw.RunPlacementGeneratePlugins(ctx, framework.NewCycleState(), nil, allNodes)
			if status.Code() != tt.wantStatusCode {
				t.Fatalf("unexpected status code: want %v, got %v (%v)", tt.wantStatusCode, status.Code(), status)
			}
			if tt.wantStatusCode != fwk.Success {
				return
			}
			if diff := cmp.Diff(tt.wantPlacements, got, cmp.AllowUnexported(framework.NodeInfo{})); diff != "" {
				t.Errorf("unexpected placements (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestRunPlacementGeneratePluginsStateMerge(t *testing.T) {
	n1 := framework.NewNodeInfo()
	n1.SetNode(st.MakeNode().Name("n1").Obj())
	n2 := framework.NewNodeInfo()
	n2.SetNode(st.MakeNode().Name("n2").Obj())
	allNodes := []fwk.NodeInfo{n1, n2}

	t.Run("disjoint plugin state is combined, cloned, and parent states are cleaned up", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		sourceDataA := &mergeData{value: "from-a"}
		sourceDataB := &mergeData{value: "from-b"}
		pA := &fwk.Placement{Name: "a", Nodes: []fwk.NodeInfo{n1, n2}}
		pB := &fwk.Placement{Name: "b", Nodes: []fwk.NodeInfo{n1, n2}}
		plugins := []*mergeTestPlugin{
			{
				name:       "a",
				placements: []*fwk.Placement{pA},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pA: {"tas": sourceDataA}},
			},
			{
				name:       "b",
				placements: []*fwk.Placement{pB},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pB: {"dra": sourceDataB}},
			},
		}
		fw := buildMergeFramework(t, ctx, plugins)
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if len(got) != 1 || got[0].Name != "a/b" {
			t.Fatalf("expected a single merged placement %q, got %v", "a/b", got)
		}
		// Mutate source data to verify merged placement holds cloned StateData.
		sourceDataA.value = "mutated"

		mergedState := state.PlacementState(got[0])
		vA, err := mergedState.Read("tas")
		if err != nil || vA.(*mergeData).value != "from-a" {
			t.Errorf("tas = %v, %v; want from-a", vA, err)
		}
		vB, err := mergedState.Read("dra")
		if err != nil || vB.(*mergeData).value != "from-b" {
			t.Errorf("dra = %v, %v; want from-b", vB, err)
		}

		// Verify intermediate parent placement states were deleted from PodGroupCycleState.
		if _, err := state.PlacementState(pA).Read("tas"); !errors.Is(err, fwk.ErrNotFound) {
			t.Errorf("expected pA state to be deleted, got err=%v", err)
		}
		if _, err := state.PlacementState(pB).Read("dra"); !errors.Is(err, fwk.ErrNotFound) {
			t.Errorf("expected pB state to be deleted, got err=%v", err)
		}
	})

	t.Run("conflicting plugin state keys produce an error", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		pA := &fwk.Placement{Name: "a", Nodes: []fwk.NodeInfo{n1, n2}}
		pB := &fwk.Placement{Name: "b", Nodes: []fwk.NodeInfo{n1, n2}}
		plugins := []*mergeTestPlugin{
			{
				name:       "a",
				placements: []*fwk.Placement{pA},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pA: {"shared": &mergeData{value: "from-a"}}},
			},
			{
				name:       "b",
				placements: []*fwk.Placement{pB},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pB: {"shared": &mergeData{value: "from-b"}}},
			},
		}
		fw := buildMergeFramework(t, ctx, plugins)
		_, status := fw.RunPlacementGeneratePlugins(ctx, framework.NewCycleState(), nil, allNodes)
		if status.Code() != fwk.Error {
			t.Errorf("expected Error status for conflicting state keys, got %v", status)
		}
	})

	t.Run("single constraining plugin preserves placement state", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		pA := &fwk.Placement{Name: "zone-a", Nodes: []fwk.NodeInfo{n1}}
		plugins := []*mergeTestPlugin{
			{
				name:       "a",
				placements: []*fwk.Placement{pA},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pA: {"tas": &mergeData{value: "solo"}}},
			},
			{
				name:         "b",
				returnParent: true,
			},
		}
		fw := buildMergeFramework(t, ctx, plugins)
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if len(got) != 1 || got[0] != pA {
			t.Fatalf("expected pA unchanged, got %v", got)
		}
		v, err := state.PlacementState(got[0]).Read("tas")
		if err != nil || v.(*mergeData).value != "solo" {
			t.Errorf("tas = %v, %v; want solo", v, err)
		}
	})

	t.Run("unconstrained plugin state is merged into every constrained placement", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		globalData := &mergeData{value: "global"}
		pB1 := &fwk.Placement{Name: "b1", Nodes: []fwk.NodeInfo{n1}}
		pB2 := &fwk.Placement{Name: "b2", Nodes: []fwk.NodeInfo{n2}}
		unconstrained := &mergeTestPlugin{
			name:         "a",
			returnParent: true,
			parentState:  map[fwk.StateKey]fwk.StateData{"global": globalData},
		}
		constrained := &mergeTestPlugin{
			name:       "b",
			placements: []*fwk.Placement{pB1, pB2},
		}
		fw := buildMergeFramework(t, ctx, []*mergeTestPlugin{unconstrained, constrained})
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if unconstrained.parent == constrained.parent {
			t.Fatal("plugins received the same parent placement pointer")
		}
		globalData.value = "mutated"
		for _, placement := range got {
			data, err := state.PlacementState(placement).Read("global")
			if err != nil || data.(*mergeData).value != "global" {
				t.Errorf("placement %q global state = %v, %v; want global", placement.Name, data, err)
			}
		}
		if _, err := state.PlacementState(unconstrained.parent).Read("global"); !errors.Is(err, fwk.ErrNotFound) {
			t.Errorf("expected unconstrained parent state to be deleted, got err=%v", err)
		}
	})

	t.Run("all unconstrained plugin states are combined", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		pluginA := &mergeTestPlugin{
			name:         "a",
			returnParent: true,
			parentState:  map[fwk.StateKey]fwk.StateData{"a": &mergeData{value: "from-a"}},
		}
		pluginB := &mergeTestPlugin{
			name:         "b",
			returnParent: true,
			parentState:  map[fwk.StateKey]fwk.StateData{"b": &mergeData{value: "from-b"}},
		}
		fw := buildMergeFramework(t, ctx, []*mergeTestPlugin{pluginA, pluginB})
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if pluginA.parent == pluginB.parent {
			t.Fatal("plugins received the same parent placement pointer")
		}
		if len(got) != 1 {
			t.Fatalf("expected one input placement, got %d", len(got))
		}
		for key, want := range map[fwk.StateKey]string{"a": "from-a", "b": "from-b"} {
			data, err := state.PlacementState(got[0]).Read(key)
			if err != nil || data.(*mergeData).value != want {
				t.Errorf("state %q = %v, %v; want %q", key, data, err, want)
			}
		}
	})

	t.Run("conflicting unconstrained plugin state keys produce an error", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		plugins := []*mergeTestPlugin{
			{
				name:         "a",
				returnParent: true,
				parentState:  map[fwk.StateKey]fwk.StateData{"shared": &mergeData{value: "from-a"}},
			},
			{
				name:         "b",
				returnParent: true,
				parentState:  map[fwk.StateKey]fwk.StateData{"shared": &mergeData{value: "from-b"}},
			},
		}
		fw := buildMergeFramework(t, ctx, plugins)
		_, status := fw.RunPlacementGeneratePlugins(ctx, framework.NewCycleState(), nil, allNodes)
		if status.Code() != fwk.Error {
			t.Errorf("expected Error status for conflicting state keys, got %v", status)
		}
	})

	t.Run("plugins returning parent alternatives keep parent state isolated", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		pluginA := &mergeTestPlugin{
			name:          "a",
			includeParent: true,
			placements:    []*fwk.Placement{{Name: "a", Nodes: []fwk.NodeInfo{n1, n2}}},
			parentState:   map[fwk.StateKey]fwk.StateData{"from-a": &mergeData{value: "a"}},
		}
		pluginB := &mergeTestPlugin{
			name:          "b",
			includeParent: true,
			placements:    []*fwk.Placement{{Name: "b", Nodes: []fwk.NodeInfo{n1, n2}}},
			parentState:   map[fwk.StateKey]fwk.StateData{"from-b": &mergeData{value: "b"}},
		}
		fw := buildMergeFramework(t, ctx, []*mergeTestPlugin{pluginA, pluginB})
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if pluginA.parent == pluginB.parent {
			t.Fatal("plugins received the same parent placement pointer")
		}
		var parentsMerged *fwk.Placement
		for _, placement := range got {
			if placement.Name == "/" {
				parentsMerged = placement
				break
			}
		}
		if parentsMerged == nil {
			t.Fatalf("merged parent placement not found in %v", got)
		}
		for _, key := range []fwk.StateKey{"from-a", "from-b"} {
			if _, err := state.PlacementState(parentsMerged).Read(key); err != nil {
				t.Errorf("merged parent state missing key %q: %v", key, err)
			}
		}
	})

	t.Run("same placement names from different plugins do not collide in state merge", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		pA := &fwk.Placement{Name: "dup", Nodes: []fwk.NodeInfo{n1, n2}}
		pB := &fwk.Placement{Name: "dup", Nodes: []fwk.NodeInfo{n2}}
		plugins := []*mergeTestPlugin{
			{
				name:       "a",
				placements: []*fwk.Placement{pA},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pA: {"tas": &mergeData{value: "from-a"}}},
			},
			{
				name:       "b",
				placements: []*fwk.Placement{pB},
				state:      map[*fwk.Placement]map[fwk.StateKey]fwk.StateData{pB: {"dra": &mergeData{value: "from-b"}}},
			},
		}
		fw := buildMergeFramework(t, ctx, plugins)
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if len(got) != 1 || got[0].Name != "dup/dup" {
			t.Fatalf("unexpected placements: %+v", got)
		}
		mergedState := state.PlacementState(got[0])
		if v1, err := mergedState.Read("tas"); err != nil || v1.(*mergeData).value != "from-a" {
			t.Errorf("tas = %v, %v; want from-a", v1, err)
		}
		if v2, err := mergedState.Read("dra"); err != nil || v2.(*mergeData).value != "from-b" {
			t.Errorf("dra = %v, %v; want from-b", v2, err)
		}
	})
}

var _ fwk.PlacementGeneratePlugin = &mergeTestPlugin{}
