/*
Copyright 2019 The Kubernetes Authors.

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
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

type fakeData struct {
	data string
}

func (f *fakeData) Clone() fwk.StateData {
	copy := &fakeData{
		data: f.data,
	}
	return copy
}

var key fwk.StateKey = "fakedata_key"

// createCycleStateWithFakeData creates *CycleState with fakeData.
// The given data is used in stored fakeData.
func createCycleStateWithFakeData(data string, recordPluginMetrics bool, skipAllPostFilterPlugins bool, skipPlugins ...[]string) *CycleState {
	c := NewCycleState()
	c.Write(key, &fakeData{
		data: data,
	})
	c.SetRecordPluginMetrics(recordPluginMetrics)
	if len(skipPlugins) > 0 {
		c.SetSkipFilterPlugins(sets.New(skipPlugins[0]...))
	}
	if len(skipPlugins) > 1 {
		c.SetSkipScorePlugins(sets.New(skipPlugins[1]...))
	}
	c.SetSkipAllPostFilterPlugins(skipAllPostFilterPlugins)
	return c
}

// isCycleStateEqual returns whether two CycleState, which has fakeData in storage, is equal or not.
// And if they are not equal, returns message which shows why not equal.
func isCycleStateEqual(a, b *CycleState) (bool, string) {
	if a == nil && b == nil {
		return true, ""
	}
	if a == nil || b == nil {
		return false, fmt.Sprintf("one CycleState is nil, but another one is not nil. A: %v, B: %v", a, b)
	}

	if a.recordPluginMetrics != b.recordPluginMetrics {
		return false, fmt.Sprintf("CycleState A and B have a different recordPluginMetrics. A: %v, B: %v", a.recordPluginMetrics, b.recordPluginMetrics)
	}
	if diff := cmp.Diff(a.skipFilterPlugins, b.skipFilterPlugins); diff != "" {
		return false, fmt.Sprintf("CycleState A and B have different SkipFilterPlugin sets. -wanted,+got:\n%s", diff)
	}
	if diff := cmp.Diff(a.skipScorePlugins, b.skipScorePlugins); diff != "" {
		return false, fmt.Sprintf("CycleState A and B have different SkipScorePlugins sets. -wanted,+got:\n%s", diff)
	}
	if diff := cmp.Diff(a.skipAllPostFilterPlugins, b.skipAllPostFilterPlugins); diff != "" {
		return false, fmt.Sprintf("CycleState A and B have different SkipAllPostFilterPlugins sets. -wanted,+got:\n%s", diff)
	}

	var msg string
	isEqual := true
	countA := 0
	a.storage.Range(func(k, v1 interface{}) bool {
		countA++
		v2, ok := b.storage.Load(k)
		if !ok {
			isEqual = false
			msg = fmt.Sprintf("CycleState B doesn't have the data which CycleState A has. key: %v, data: %v", k, v1)
			return false
		}

		typed1, ok1 := v1.(*fakeData)
		typed2, ok2 := v2.(*fakeData)
		if !ok1 || !ok2 {
			isEqual = false
			msg = "CycleState has the data which is not type *fakeData."
			return false
		}

		if typed1.data != typed2.data {
			isEqual = false
			msg = fmt.Sprintf("CycleState B has a different data on key %v. A: %v, B: %v", k, typed1.data, typed2.data)
			return false
		}

		return true
	})

	if !isEqual {
		return false, msg
	}

	countB := 0
	b.storage.Range(func(k, _ interface{}) bool {
		countB++
		return true
	})

	if countA != countB {
		return false, fmt.Sprintf("two Cyclestates have different numbers of data. A: %v, B: %v", countA, countB)
	}

	return true, ""
}

func TestCycleStateClone(t *testing.T) {
	tests := []struct {
		name            string
		state           *CycleState
		wantClonedState *CycleState
	}{
		{
			name:            "clone with recordPluginMetrics true",
			state:           createCycleStateWithFakeData("data", true, false),
			wantClonedState: createCycleStateWithFakeData("data", true, false),
		},
		{
			name:            "clone with recordPluginMetrics false",
			state:           createCycleStateWithFakeData("data", false, false),
			wantClonedState: createCycleStateWithFakeData("data", false, false),
		},
		{
			name:            "clone with SkipFilterPlugins",
			state:           createCycleStateWithFakeData("data", true, false, []string{"p1", "p2", "p3"}),
			wantClonedState: createCycleStateWithFakeData("data", true, false, []string{"p1", "p2", "p3"}),
		},
		{
			name:            "clone with SkipScorePlugins",
			state:           createCycleStateWithFakeData("data", false, false, []string{}, []string{"p1", "p2", "p3"}),
			wantClonedState: createCycleStateWithFakeData("data", false, false, []string{}, []string{"p1", "p2", "p3"}),
		},
		{
			name:            "clone with SkipScorePlugins and SkipFilterPlugins",
			state:           createCycleStateWithFakeData("data", true, false, []string{"p0"}, []string{"p1", "p2", "p3"}),
			wantClonedState: createCycleStateWithFakeData("data", true, false, []string{"p0"}, []string{"p1", "p2", "p3"}),
		},
		{
			name:            "clone with SkipAllPostFilterPlugins",
			state:           createCycleStateWithFakeData("data", true, true, []string{"p0"}, []string{"p1", "p2", "p3"}),
			wantClonedState: createCycleStateWithFakeData("data", true, true, []string{"p0"}, []string{"p1", "p2", "p3"}),
		},
		{
			name:            "clone with nil CycleState",
			state:           nil,
			wantClonedState: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := tt.state
			copy := state.Clone()
			var stateCopy *CycleState
			if copy != nil {
				stateCopy = copy.(*CycleState)
			}

			if isEqual, msg := isCycleStateEqual(stateCopy, tt.wantClonedState); !isEqual {
				t.Errorf("unexpected cloned state: %v", msg)
			}

			if state == nil || stateCopy == nil {
				// not need to run the rest check in this case.
				return
			}

			stateCopy.Write(key, &fakeData{data: "modified"})
			if isEqual, _ := isCycleStateEqual(state, stateCopy); isEqual {
				t.Errorf("the change for a cloned state should not affect the original state.")
			}
		})
	}
}

func TestPlacementCycleState(t *testing.T) {
	t.Run("nil by default", func(t *testing.T) {
		state := NewCycleState()
		if state.GetPlacementCycleState() != nil {
			t.Errorf("expected nil PlacementCycleState on fresh CycleState")
		}
	})

	t.Run("set and get", func(t *testing.T) {
		state := NewCycleState()
		podGroupState := NewCycleState()
		placementState := NewCycleState()
		placementState.SetPodGroupSchedulingCycle(podGroupState)
		placementState.Write("testkey", &fakeData{data: "placementdata"})

		state.SetPlacementCycleState(placementState)

		got := state.GetPlacementCycleState()
		if got == nil {
			t.Fatal("expected non-nil PlacementCycleState after Set")
		}

		data, err := got.Read("testkey")
		if err != nil {
			t.Fatalf("unexpected error reading from PlacementCycleState: %v", err)
		}
		if data.(*fakeData).data != "placementdata" {
			t.Errorf("expected 'placementdata', got %q", data.(*fakeData).data)
		}
		if got.GetPodGroupSchedulingCycle() != podGroupState {
			t.Errorf("expected PlacementCycleState to expose its PodGroupCycleState")
		}
	})

	t.Run("set to nil clears", func(t *testing.T) {
		state := NewCycleState()
		state.SetPlacementCycleState(NewCycleState())
		state.SetPlacementCycleState(nil)

		if state.GetPlacementCycleState() != nil {
			t.Errorf("expected nil PlacementCycleState after setting to nil")
		}
	})

	t.Run("clone preserves reference", func(t *testing.T) {
		state := NewCycleState()
		state.Write(key, &fakeData{data: "pod-data"})

		placementState := NewCycleState()
		placementState.Write("pkey", &fakeData{data: "placement-data"})
		state.SetPlacementCycleState(placementState)

		cloned := state.Clone().(*CycleState)

		// The cloned state should reference the same PlacementCycleState.
		if cloned.GetPlacementCycleState() == nil {
			t.Fatal("cloned state should have non-nil PlacementCycleState")
		}

		data, err := cloned.GetPlacementCycleState().Read("pkey")
		if err != nil {
			t.Fatalf("unexpected error reading from cloned PlacementCycleState: %v", err)
		}
		if data.(*fakeData).data != "placement-data" {
			t.Errorf("expected 'placement-data', got %q", data.(*fakeData).data)
		}

		// Writes to the PlacementCycleState via the clone should be visible from the original,
		// since it's a shared reference (same as podGroupCycleState behavior).
		cloned.GetPlacementCycleState().Write("newkey", &fakeData{data: "new"})
		newData, err := state.GetPlacementCycleState().Read("newkey")
		if err != nil {
			t.Fatalf("write via clone's PlacementCycleState should be visible from original: %v", err)
		}
		if newData.(*fakeData).data != "new" {
			t.Errorf("expected 'new', got %q", newData.(*fakeData).data)
		}
	})

	t.Run("clone with nil placement state", func(t *testing.T) {
		state := NewCycleState()
		state.Write(key, &fakeData{data: "data"})
		// Do not set PlacementCycleState — leave nil.

		cloned := state.Clone().(*CycleState)
		if cloned.GetPlacementCycleState() != nil {
			t.Errorf("cloned state should have nil PlacementCycleState when original has nil")
		}
	})
}

func TestPlacementCycleStateForName(t *testing.T) {
	c := NewCycleState()

	if got := c.GetPlacementCycleStateForName("missing"); got != nil {
		t.Errorf("expected nil for unknown placement name, got %v", got)
	}

	state := NewCycleState()
	state.Write(key, &fakeData{data: "v"})
	c.SetPlacementCycleStateForName("p1", state)

	if got := c.GetPlacementCycleStateForName("p1"); got != state {
		t.Errorf("expected to read back the registered state, got %v", got)
	}

	c.DeletePlacementCycleStateForName("p1")
	if got := c.GetPlacementCycleStateForName("p1"); got != nil {
		t.Errorf("expected nil after delete, got %v", got)
	}
}

func TestMergePlacementStatesInto(t *testing.T) {
	keyA := fwk.StateKey("a")
	keyB := fwk.StateKey("b")

	t.Run("disjoint keys are combined", func(t *testing.T) {
		c := NewCycleState()
		sa := NewCycleState()
		sa.Write(keyA, &fakeData{data: "va"})
		sb := NewCycleState()
		sb.Write(keyB, &fakeData{data: "vb"})
		c.SetPlacementCycleStateForName("a", sa)
		c.SetPlacementCycleStateForName("b", sb)

		if err := c.MergePlacementStatesInto("a/b", "a", "b"); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		merged := c.GetPlacementCycleStateForName("a/b").(*CycleState)
		for _, k := range []fwk.StateKey{keyA, keyB} {
			if _, err := merged.Read(k); err != nil {
				t.Errorf("merged state missing key %q: %v", k, err)
			}
		}
	})

	t.Run("conflicting keys return an error", func(t *testing.T) {
		c := NewCycleState()
		sa := NewCycleState()
		sa.Write(keyA, &fakeData{data: "va"})
		sb := NewCycleState()
		sb.Write(keyA, &fakeData{data: "vb"})
		c.SetPlacementCycleStateForName("a", sa)
		c.SetPlacementCycleStateForName("b", sb)

		if err := c.MergePlacementStatesInto("a/b", "a", "b"); err == nil {
			t.Errorf("expected an error for conflicting keys, got nil")
		}
	})

	t.Run("merged state is isolated from sources", func(t *testing.T) {
		c := NewCycleState()
		sa := NewCycleState()
		original := &fakeData{data: "va"}
		sa.Write(keyA, original)
		c.SetPlacementCycleStateForName("a", sa)

		if err := c.MergePlacementStatesInto("a/b", "a"); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Mutate the source value; the merged copy must not change.
		original.data = "mutated"

		merged := c.GetPlacementCycleStateForName("a/b").(*CycleState)
		got, err := merged.Read(keyA)
		if err != nil {
			t.Fatalf("merged state missing key: %v", err)
		}
		if got.(*fakeData).data != "va" {
			t.Errorf("merged state shares mutable data with source: got %q", got.(*fakeData).data)
		}
	})
}

func TestCopyPlacementDataInto(t *testing.T) {
	src := NewCycleState()
	original := &fakeData{data: "v"}
	src.Write(key, original)

	dst := NewCycleState()
	src.CopyPlacementDataInto(dst)

	original.data = "mutated"
	got, err := dst.Read(key)
	if err != nil {
		t.Fatalf("dst missing copied key: %v", err)
	}
	if got.(*fakeData).data != "v" {
		t.Errorf("copied data shares mutable state with source: got %q", got.(*fakeData).data)
	}
}

func TestCloneCopiesPlacementStates(t *testing.T) {
	c := NewCycleState()
	src := NewCycleState()
	original := &fakeData{data: "v"}
	src.Write(key, original)
	c.SetPlacementCycleStateForName("p1", src)

	cloned := c.Clone().(*CycleState)
	original.data = "mutated"

	got := cloned.GetPlacementCycleStateForName("p1")
	if got == nil {
		t.Fatal("cloned state lost the named placement state")
	}
	data, err := got.Read(key)
	if err != nil {
		t.Fatalf("cloned named state missing key: %v", err)
	}
	if data.(*fakeData).data != "v" {
		t.Errorf("cloned named state shares mutable data with source: got %q", data.(*fakeData).data)
	}
}
