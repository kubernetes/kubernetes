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
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type mergeData struct {
	value string
}

func (d *mergeData) Clone() fwk.StateData {
	return &mergeData{value: d.value}
}

func TestMergePlacements(t *testing.T) {
	nodes := BuildNodeInfos([]*v1.Node{
		st.MakeNode().Name("node1").Obj(),
		st.MakeNode().Name("node2").Obj(),
		st.MakeNode().Name("node3").Obj(),
	})
	placement := func(name string, nodeIndices ...int) *fwk.Placement {
		p := &fwk.Placement{Name: name}
		for _, i := range nodeIndices {
			p.Nodes = append(p.Nodes, nodes[i])
		}
		return p
	}
	a := placement("a", 0, 1)
	b := placement("b", 0, 1)
	b1 := placement("b1", 0)
	b2 := placement("b2", 1)
	disjoint := placement("disjoint", 2)
	dupA := placement("dup", 0, 1)
	dupB := placement("dup", 1)
	tests := map[string]struct {
		as             []*fwk.Placement
		bs             []*fwk.Placement
		sourceState    map[*fwk.Placement]map[fwk.StateKey]string
		wantPlacements []*fwk.Placement
		wantStatusCode fwk.Code
		wantStates     []map[fwk.StateKey]string
	}{
		"intersection preserves left node order and removes duplicates": {
			as:             []*fwk.Placement{placement("a", 2, 1, 1, 0)},
			bs:             []*fwk.Placement{placement("b", 1, 2, 2)},
			wantPlacements: []*fwk.Placement{placement("a/b", 2, 1)},
		},
		"cross product drops pairs with no common nodes": {
			as:             []*fwk.Placement{placement("a1", 0), placement("a2", 1)},
			bs:             []*fwk.Placement{b1, b2, disjoint},
			wantPlacements: []*fwk.Placement{placement("a1/b1", 0), placement("a2/b2", 1)},
		},
		"disjoint state is combined and cloned": {
			as:             []*fwk.Placement{a},
			bs:             []*fwk.Placement{b},
			sourceState:    map[*fwk.Placement]map[fwk.StateKey]string{a: {"tas": "from-a"}, b: {"dra": "from-b"}},
			wantPlacements: []*fwk.Placement{placement("a/b", 0, 1)},
			wantStates:     []map[fwk.StateKey]string{{"tas": "from-a", "dra": "from-b"}},
		},
		"siblings have independent state": {
			as:             []*fwk.Placement{a},
			bs:             []*fwk.Placement{b1, b2},
			sourceState:    map[*fwk.Placement]map[fwk.StateKey]string{a: {"tas": "from-a"}, b1: {"dra": "from-b1"}, b2: {"dra": "from-b2"}},
			wantPlacements: []*fwk.Placement{placement("a/b1", 0), placement("a/b2", 1)},
			wantStates:     []map[fwk.StateKey]string{{"tas": "from-a", "dra": "from-b1"}, {"tas": "from-a", "dra": "from-b2"}},
		},
		"same names from different plugins do not collide in state": {
			as:             []*fwk.Placement{dupA},
			bs:             []*fwk.Placement{dupB},
			sourceState:    map[*fwk.Placement]map[fwk.StateKey]string{dupA: {"tas": "from-a"}, dupB: {"dra": "from-b"}},
			wantPlacements: []*fwk.Placement{placement("dup/dup", 1)},
			wantStates:     []map[fwk.StateKey]string{{"tas": "from-a", "dra": "from-b"}},
		},
		"conflicting state keys return an error": {
			as:             []*fwk.Placement{a},
			bs:             []*fwk.Placement{b},
			sourceState:    map[*fwk.Placement]map[fwk.StateKey]string{a: {"shared": "from-a"}, b: {"shared": "from-b"}},
			wantStatusCode: fwk.Error,
		},
		"empty intersections still clean up source state": {
			as:          []*fwk.Placement{a},
			bs:          []*fwk.Placement{disjoint},
			sourceState: map[*fwk.Placement]map[fwk.StateKey]string{a: {"tas": "from-a"}, disjoint: {"dra": "from-b"}},
		},
		"empty left set": {
			bs:          []*fwk.Placement{b},
			sourceState: map[*fwk.Placement]map[fwk.StateKey]string{b: {"dra": "from-b"}},
		},
		"empty right set": {
			as:          []*fwk.Placement{a},
			sourceState: map[*fwk.Placement]map[fwk.StateKey]string{a: {"tas": "from-a"}},
		},
		"empty placement name": {
			as:             []*fwk.Placement{placement("", 0)},
			bs:             []*fwk.Placement{b},
			wantPlacements: []*fwk.Placement{placement("/b", 0)},
		},
	}

	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			state := framework.NewCycleState()
			stateKeys := make(map[fwk.StateKey]struct{})
			var originals []*mergeData
			for p, values := range tt.sourceState {
				for key, value := range values {
					data := &mergeData{value: value}
					originals = append(originals, data)
					state.PlacementState(p).Write(key, data)
					stateKeys[key] = struct{}{}
				}
			}

			got, status := mergePlacements(tt.as, tt.bs, state)
			if status.Code() != tt.wantStatusCode {
				t.Fatalf("unexpected status: got %v, want %v", status, tt.wantStatusCode)
			}
			if diff := cmp.Diff(tt.wantPlacements, got, cmp.Comparer(func(a, b fwk.NodeInfo) bool { return a == b })); diff != "" {
				t.Errorf("unexpected placements (-want,+got):\n%s", diff)
			}
			if !status.IsSuccess() {
				return
			}
			for _, original := range originals {
				original.value = "mutated source"
			}
			if len(tt.wantStates) > 0 && len(tt.wantStates) != len(got) {
				t.Fatalf("expected state for %d placements, got %d placements", len(tt.wantStates), len(got))
			}
			for i, p := range got {
				var want map[fwk.StateKey]string
				if len(tt.wantStates) > 0 {
					want = tt.wantStates[i]
				}
				for key := range stateKeys {
					data, err := state.PlacementState(p).Read(key)
					value, present := want[key]
					if !present {
						if !errors.Is(err, fwk.ErrNotFound) {
							t.Errorf("placement %q unexpectedly has state %q: %v, %v", p.Name, key, data, err)
						}
						continue
					}
					if err != nil {
						t.Fatalf("placement %q missing state %q: %v", p.Name, key, err)
					}
					gotData := data.(*mergeData)
					if gotData.value != value {
						t.Errorf("placement %q state %q = %q, want %q", p.Name, key, gotData.value, value)
					}
					// Later siblings must not observe changes to this placement's state.
					gotData.value = "mutated sibling"
				}
			}
			for p, values := range tt.sourceState {
				for key := range values {
					if _, err := state.PlacementState(p).Read(key); !errors.Is(err, fwk.ErrNotFound) {
						t.Errorf("source placement %q state %q was not deleted: %v", p.Name, key, err)
					}
				}
			}
		})
	}
}

func TestValidatePlacementNames(t *testing.T) {
	tests := map[string]struct {
		placements []*fwk.Placement
		wantError  string
	}{
		"no placements": {},
		"unique names": {
			placements: []*fwk.Placement{{Name: "a"}, {Name: "b"}},
		},
		"empty name is valid": {
			placements: []*fwk.Placement{{Name: ""}, {Name: "a"}},
		},
		"duplicate names": {
			placements: []*fwk.Placement{{Name: "dup"}, {Name: "dup"}},
			wantError:  `plugin returned duplicate placement name "dup"`,
		},
		"duplicate empty names": {
			placements: []*fwk.Placement{{}, {}},
			wantError:  `plugin returned duplicate placement name ""`,
		},
		"separator in a name": {
			placements: []*fwk.Placement{{Name: "zone/rack"}},
			wantError:  `placement name "zone/rack" must not contain "/"`,
		},
		"nil placement": {
			placements: []*fwk.Placement{nil},
			wantError:  "plugin returned a nil placement",
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			err := validatePlacementNames(tt.placements)
			if tt.wantError == "" {
				if err != nil {
					t.Fatalf("unexpected validation error: %v", err)
				}
				return
			}
			if err == nil || err.Error() != tt.wantError {
				t.Errorf("validation error = %v, want %q", err, tt.wantError)
			}
		})
	}
}
