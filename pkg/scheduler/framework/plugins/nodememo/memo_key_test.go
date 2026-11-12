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

package nodememo

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

// TestWriteSelectorDistinguishesNothingFromEverything pins the one distinction that String() cannot
// make, at the source both plugins share. A missing labelSelector becomes labels.Nothing() and an
// empty one becomes labels.Everything() - opposite meanings, identical rendering - and a memo key
// built on the rendering alone gives the two shapes one entry, so one pod is filtered and scored on
// the other's counts.
func TestWriteSelectorDistinguishesNothingFromEverything(t *testing.T) {
	asSelector := func(ls *metav1.LabelSelector) labels.Selector {
		sel, err := metav1.LabelSelectorAsSelector(ls)
		if err != nil {
			t.Fatalf("LabelSelectorAsSelector(%v): %v", ls, err)
		}
		return sel
	}
	render := func(sel labels.Selector) string {
		var b strings.Builder
		WriteSelector(&b, sel)
		return b.String()
	}

	nothing := asSelector(nil)
	everything := asSelector(&metav1.LabelSelector{})
	matching := asSelector(&metav1.LabelSelector{MatchLabels: map[string]string{"app": "test"}})

	// The fixture has to be the two opposite selectors, or nothing below proves anything.
	if !labels.MatchesNothing(nothing) {
		t.Fatalf("a missing labelSelector is supposed to become labels.Nothing(), got %q", nothing.String())
	}
	if labels.MatchesNothing(everything) {
		t.Fatalf("an empty labelSelector is supposed to become labels.Everything(), got %q", everything.String())
	}
	if nothing.String() != everything.String() {
		t.Fatalf("the premise changed: Nothing() renders %q and Everything() renders %q, so String() alone would tell them apart",
			nothing.String(), everything.String())
	}

	for _, tc := range []struct {
		name string
		sel  labels.Selector
		want string
	}{
		{"a nil selector", nil, "<nil>"},
		{"Nothing", nothing, "<nothing>"},
		{"Everything", everything, "<>"},
		{"a real selector", matching, "<app=test>"},
	} {
		if got := render(tc.sel); got != tc.want {
			t.Errorf("%s rendered %q, want %q", tc.name, got, tc.want)
		}
	}

	seen := map[string]string{}
	for _, tc := range []struct {
		name string
		sel  labels.Selector
	}{
		{"nil", nil}, {"Nothing", nothing}, {"Everything", everything}, {"a real selector", matching},
	} {
		got := render(tc.sel)
		if other, ok := seen[got]; ok {
			t.Errorf("%s and %s both render %q, so two shapes would share one memo entry", other, tc.name, got)
		}
		seen[got] = tc.name
	}
}
