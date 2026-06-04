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
	"fmt"
	"sort"
	"testing"

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

func (d mergeData) Clone() fwk.StateData { return d }

// mergeTestPlugin is a PlacementGeneratePlugin used to exercise multi-plugin merging. It returns
// configured placements, optionally returns the parent placement unchanged, and optionally writes
// per-placement state keyed by placement name.
type mergeTestPlugin struct {
	name string
	// placements are returned by GeneratePlacements (parent is substituted for nil entries).
	placements []*fwk.Placement
	// returnParent makes the plugin return the parent placement unchanged (no constraint).
	returnParent bool
	statusCode   fwk.Code
	// state maps placement name -> (state key -> value) written during generation.
	state map[string]map[fwk.StateKey]string
}

func (p *mergeTestPlugin) Name() string { return p.name }

func (p *mergeTestPlugin) GeneratePlacements(_ context.Context, state fwk.PodGroupCycleState, _ fwk.PodGroupInfo, parent *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	for name, kv := range p.state {
		ps := framework.NewCycleState()
		for k, v := range kv {
			ps.Write(k, mergeData{value: v})
		}
		state.SetPlacementCycleStateForName(name, ps)
	}
	if p.statusCode != fwk.Success {
		return nil, fwk.NewStatus(p.statusCode, "injected")
	}
	if p.returnParent {
		return &fwk.GeneratePlacementsResult{Placements: []*fwk.Placement{parent}}, nil
	}
	return &fwk.GeneratePlacementsResult{Placements: p.placements}, nil
}

func placementNodeNames(p *fwk.Placement) []string {
	names := make([]string, 0, len(p.Nodes))
	for _, n := range p.Nodes {
		names = append(names, n.Node().Name)
	}
	sort.Strings(names)
	return names
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
		// wantPlacements maps expected placement name -> sorted node names.
		wantPlacements map[string][]string
	}{
		"two single-placement plugins intersect": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a", "n1", "n2", "n3")}},
				{name: "b", placements: []*fwk.Placement{placement("b", "n2", "n3", "n4")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: map[string][]string{"a/b": {"n2", "n3"}},
		},
		"cross product drops empty intersections": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a1", "n1", "n2"), placement("a2", "n3", "n4")}},
				{name: "b", placements: []*fwk.Placement{placement("b1", "n2", "n3")}},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: map[string][]string{"a1/b1": {"n2"}, "a2/b1": {"n3"}},
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
			wantPlacements: map[string][]string{"b": {"n2", "n3"}},
		},
		"all unconstrained returns input placement": {
			plugins: []*mergeTestPlugin{
				{name: "a", returnParent: true},
				{name: "b", returnParent: true},
			},
			wantStatusCode: fwk.Success,
			wantPlacements: map[string][]string{"": {"n1", "n2", "n3", "n4"}},
		},
		"duplicate placement names across plugins is an error": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("dup", "n1", "n2")}},
				{name: "b", placements: []*fwk.Placement{placement("dup", "n2", "n3")}},
			},
			wantStatusCode: fwk.Error,
		},
		"constrained placement without a name is an error": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a", "n1", "n2")}},
				{name: "b", placements: []*fwk.Placement{placement("", "n2", "n3")}},
			},
			wantStatusCode: fwk.Error,
		},
		"placement name containing the separator is an error": {
			plugins: []*mergeTestPlugin{
				{name: "a", placements: []*fwk.Placement{placement("a/b", "n1", "n2")}},
				{name: "b", placements: []*fwk.Placement{placement("b", "n2", "n3")}},
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
			gotMap := make(map[string][]string, len(got))
			for _, p := range got {
				gotMap[p.Name] = placementNodeNames(p)
			}
			if fmt.Sprint(gotMap) != fmt.Sprint(tt.wantPlacements) {
				t.Errorf("unexpected placements:\nwant %v\ngot  %v", tt.wantPlacements, gotMap)
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

	makePlugins := func(keyA, keyB fwk.StateKey) []*mergeTestPlugin {
		return []*mergeTestPlugin{
			{
				name:       "a",
				placements: []*fwk.Placement{{Name: "a", Nodes: []fwk.NodeInfo{n1, n2}}},
				state:      map[string]map[fwk.StateKey]string{"a": {keyA: "from-a"}},
			},
			{
				name:       "b",
				placements: []*fwk.Placement{{Name: "b", Nodes: []fwk.NodeInfo{n1, n2}}},
				state:      map[string]map[fwk.StateKey]string{"b": {keyB: "from-b"}},
			},
		}
	}

	t.Run("disjoint plugin state is combined under the merged placement", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		fw := buildMergeFramework(t, ctx, makePlugins("tas", "dra"))
		state := framework.NewCycleState()
		got, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, allNodes)
		if !status.IsSuccess() {
			t.Fatalf("unexpected status: %v", status)
		}
		if len(got) != 1 || got[0].Name != "a/b" {
			t.Fatalf("expected a single merged placement %q, got %v", "a/b", got)
		}
		merged := state.GetPlacementCycleStateForName("a/b")
		if merged == nil {
			t.Fatal("merged placement has no cycle state")
		}
		for _, k := range []fwk.StateKey{"tas", "dra"} {
			if _, err := merged.Read(k); err != nil {
				t.Errorf("merged state missing key %q: %v", k, err)
			}
		}
	})

	t.Run("conflicting plugin state keys produce an error", func(t *testing.T) {
		_, ctx := ktesting.NewTestContext(t)
		fw := buildMergeFramework(t, ctx, makePlugins("shared", "shared"))
		_, status := fw.RunPlacementGeneratePlugins(ctx, framework.NewCycleState(), nil, allNodes)
		if status.Code() != fwk.Error {
			t.Errorf("expected Error status for conflicting state keys, got %v", status)
		}
	})
}

var _ fwk.PlacementGeneratePlugin = &mergeTestPlugin{}
