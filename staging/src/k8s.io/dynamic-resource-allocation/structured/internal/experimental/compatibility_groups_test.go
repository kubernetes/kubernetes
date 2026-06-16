/*
Copyright 2026 The Kubernetes Authors.

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

package experimental

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

// TestCompatibilityGroupSet verifies that nil and empty lists both produce a nil
// set, i.e. "[] is treated as nil".
func TestCompatibilityGroupSet(t *testing.T) {
	if got := compatibilityGroupSet(nil); got != nil {
		t.Errorf("compatibilityGroupSet(nil) = %v, want nil", got)
	}
	if got := compatibilityGroupSet([]string{}); got != nil {
		t.Errorf("compatibilityGroupSet([]) = %v, want nil", got)
	}
	if got := compatibilityGroupSet([]string{"mig"}); got.Len() != 1 || !got.Has("mig") {
		t.Errorf("compatibilityGroupSet([mig]) = %v, want {mig}", got)
	}
}

// TestCompatibilityGroupIntersectionAdmits exercises the admission predicate for
// a single device against an existing intersection state, covering the empty,
// nil, single and multiple group cases and the strict nil-matching rule.
func TestCompatibilityGroupIntersectionAdmits(t *testing.T) {
	set := func(groups ...string) sets.Set[string] {
		if len(groups) == 0 {
			return nil
		}
		return sets.New[string](groups...)
	}

	for name, tc := range map[string]struct {
		intersection compatibilityGroupIntersection
		candidate    sets.Set[string]
		want         bool
	}{
		"empty-intersection-admits-grouped": {
			intersection: compatibilityGroupIntersection{},
			candidate:    set("mig"),
			want:         true,
		},
		"empty-intersection-admits-nil": {
			intersection: compatibilityGroupIntersection{},
			candidate:    set(),
			want:         true,
		},
		"no-group-members-admit-no-group-candidate (nil-vs-nil)": {
			intersection: compatibilityGroupIntersection{}.add(set()),
			candidate:    set(),
			want:         true,
		},
		"no-group-members-reject-grouped-candidate (nil-vs-set)": {
			intersection: compatibilityGroupIntersection{}.add(set()),
			candidate:    set("mig"),
			want:         false,
		},
		"grouped-members-reject-no-group-candidate (set-vs-nil)": {
			intersection: compatibilityGroupIntersection{}.add(set("mig")),
			candidate:    set(),
			want:         false,
		},
		"grouped-members-admit-overlapping-candidate (set-vs-set)": {
			intersection: compatibilityGroupIntersection{}.add(set("mig")),
			candidate:    set("mig"),
			want:         true,
		},
		"grouped-members-reject-disjoint-candidate": {
			intersection: compatibilityGroupIntersection{}.add(set("mig")),
			candidate:    set("vgpu"),
			want:         false,
		},
		"multi-group-candidate-overlaps": {
			intersection: compatibilityGroupIntersection{}.add(set("foo", "foobar")),
			candidate:    set("bar", "foobar"),
			want:         true,
		},
		"multi-group-candidate-disjoint": {
			intersection: compatibilityGroupIntersection{}.add(set("foo", "foobar")),
			candidate:    set("baz"),
			want:         false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			if got := tc.intersection.admits(tc.candidate); got != tc.want {
				t.Errorf("admits() = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestCompatibilityGroupIntersectionRolling verifies the rolling-intersection
// behaviour as successive devices are folded in, mirroring KEP example 4 (a
// device may belong to multiple groups, and the intersection narrows as devices
// are added).
func TestCompatibilityGroupIntersectionRolling(t *testing.T) {
	set := func(groups ...string) sets.Set[string] { return sets.New[string](groups...) }

	// Start empty: foo device {foo, foobar} is admitted and seeds the set.
	i := compatibilityGroupIntersection{}
	if !i.admits(set("foo", "foobar")) {
		t.Fatal("expected empty intersection to admit {foo, foobar}")
	}
	i = i.add(set("foo", "foobar"))

	// bar device {bar, foobar} shares foobar -> admitted, intersection narrows to {foobar}.
	if !i.admits(set("bar", "foobar")) {
		t.Fatal("expected {foo, foobar} to admit {bar, foobar} via foobar")
	}
	i = i.add(set("bar", "foobar"))
	if i.groups.Len() != 1 || !i.groups.Has("foobar") {
		t.Fatalf("expected rolling intersection {foobar}, got %v", i.groups)
	}

	// baz device {baz} no longer shares anything -> rejected.
	if i.admits(set("baz")) {
		t.Error("expected narrowed intersection {foobar} to reject {baz}")
	}
	// A further foobar-sharing device is still admitted.
	if !i.admits(set("qux", "foobar")) {
		t.Error("expected narrowed intersection {foobar} to admit {qux, foobar}")
	}
}

// TestCompatibilityGroupIntersectionNoGroupsStacking verifies that any number of
// no-group devices stack freely, but a grouped device is rejected once a
// no-group device is present (and vice versa).
func TestCompatibilityGroupIntersectionNoGroupsStacking(t *testing.T) {
	i := compatibilityGroupIntersection{}
	for n := 0; n < 3; n++ {
		if !i.admits(nil) {
			t.Fatalf("no-group device #%d should be admitted", n)
		}
		i = i.add(nil)
	}
	if i.admits(sets.New[string]("mig")) {
		t.Error("grouped device must be rejected once no-group devices are present")
	}
}
