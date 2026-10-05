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
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type mergeData struct {
	value string
}

func (d *mergeData) Clone() fwk.StateData {
	return &mergeData{value: d.value}
}

// mergeTestPlugin is a PlacementGeneratePlugin used to exercise multi-plugin merging.
type mergeTestPlugin struct {
	name         string
	placements   []*fwk.Placement
	returnParent bool
	statusCode   fwk.Code
	// state maps placement pointer -> (state key -> value) written during generation.
	state map[*fwk.Placement]map[fwk.StateKey]fwk.StateData
}

func (p *mergeTestPlugin) Name() string { return p.name }

func (p *mergeTestPlugin) GeneratePlacements(_ context.Context, state fwk.PodGroupCycleState, _ fwk.PodGroupInfo, parent *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	if p.statusCode != fwk.Success {
		return nil, fwk.NewStatus(p.statusCode, "injected")
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
	return &fwk.GeneratePlacementsResult{Placements: p.placements}, nil
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
		"empty and slash-containing placement names succeed": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("zone-a/rack-1", "n1", "n2")}},
				{name: "b", placements: []*fwk.Placement{placement("", "n2", "n3")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: []*fwk.Placement{placement("zone-a/rack-1/", "n2")},
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

	t.Run("duplicate and slash-containing placement names do not collide in state merge", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		pA := &fwk.Placement{Name: "dup/slash", Nodes: []fwk.NodeInfo{n1, n2}}
		pB := &fwk.Placement{Name: "dup/slash", Nodes: []fwk.NodeInfo{n2}}
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
		if len(got) != 1 || got[0].Name != "dup/slash/dup/slash" {
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
