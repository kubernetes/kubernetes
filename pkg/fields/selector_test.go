/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package fields

import (
	"testing"

	"k8s.io/kubernetes/pkg/util/selectors"
	"k8s.io/kubernetes/pkg/util/sets"
)

func TestSelectorParse(t *testing.T) {
	testGoodStrings := []string{
		"x=a,y=b,z=c",
		"",
		"x!=a,y=b",
	}
	testBadStrings := []string{
		"x=a||y=b",
		"x==a==b",
	}
	for _, test := range testGoodStrings {
		lq, err := ParseSelector(test)
		if err != nil {
			t.Errorf("%v: error %v (%#v)\n", test, err, err)
		}
		if test != lq.String() {
			t.Errorf("%v restring gave: %v\n", test, lq.String())
		}
	}
	for _, test := range testBadStrings {
		_, err := ParseSelector(test)
		if err == nil {
			t.Errorf("%v: did not get expected error\n", test)
		}
	}
}

func TestDeterministicParse(t *testing.T) {
	s1, err := ParseSelector("x=a,a=x")
	s2, err2 := ParseSelector("a=x,x=a")
	if err != nil || err2 != nil {
		t.Errorf("Unexpected parse error")
	}
	if s1.String() != s2.String() {
		t.Errorf("Non-deterministic parse")
	}
}

func expectMatch(t *testing.T, selector string, ls Set) {
	lq, err := ParseSelector(selector)
	if err != nil {
		t.Errorf("Unable to parse %v as a selector\n", selector)
		return
	}
	if !lq.Matches(ls) {
		t.Errorf("Wanted %s to match '%s', but it did not.\n", selector, ls)
	}
}

func expectNoMatch(t *testing.T, selector string, ls Set) {
	lq, err := ParseSelector(selector)
	if err != nil {
		t.Errorf("Unable to parse %v as a selector\n", selector)
		return
	}
	if lq.Matches(ls) {
		t.Errorf("Wanted '%s' to not match '%s', but it did.", selector, ls)
	}
}

func TestEverything(t *testing.T) {
	if !Everything().Matches(Set{"x": "y"}) {
		t.Errorf("Nil selector didn't match")
	}
	if !Everything().Empty() {
		t.Errorf("Everything was not empty")
	}
}

func TestSelectorMatches(t *testing.T) {
	expectMatch(t, "", Set{"x": "y"})
	expectMatch(t, "x=y", Set{"x": "y"})
	expectMatch(t, "x=y,z=w", Set{"x": "y", "z": "w"})
	expectMatch(t, "x!=y,z!=w", Set{"x": "z", "z": "a"})
	expectMatch(t, "notin=in", Set{"notin": "in"}) // in and notin in exactMatch
	expectNoMatch(t, "x=y", Set{"x": "z"})
	expectNoMatch(t, "x=y,z=w", Set{"x": "w", "z": "w"})
	expectNoMatch(t, "x!=y,z!=w", Set{"x": "z", "z": "w"})

	labelset := Set{
		"foo": "bar",
		"baz": "blah",
	}
	expectMatch(t, "foo=bar", labelset)
	expectMatch(t, "baz=blah", labelset)
	expectMatch(t, "foo=bar,baz=blah", labelset)
	expectNoMatch(t, "foo=blah", labelset)
	expectNoMatch(t, "baz=bar", labelset)
	expectNoMatch(t, "foo=bar,foobar=bar,baz=blah", labelset)
}

func TestOneTermEqualSelector(t *testing.T) {
	if !OneTermEqualSelector("x", "y").Matches(Set{"x": "y"}) {
		t.Errorf("No match when match expected.")
	}
	if OneTermEqualSelector("x", "y").Matches(Set{"x": "z"}) {
		t.Errorf("Match when none expected.")
	}
}

func expectMatchDirect(t *testing.T, selector, ls Set) {
	if !SelectorFromSet(selector).Matches(ls) {
		t.Errorf("Wanted %s to match '%s', but it did not.\n", selector, ls)
	}
}

func expectNoMatchDirect(t *testing.T, selector, ls Set) {
	if SelectorFromSet(selector).Matches(ls) {
		t.Errorf("Wanted '%s' to not match '%s', but it did.", selector, ls)
	}
}

func TestSetMatches(t *testing.T) {
	labelset := Set{
		"foo": "bar",
		"baz": "blah",
	}
	expectMatchDirect(t, Set{}, labelset)
	expectMatchDirect(t, Set{"foo": "bar"}, labelset)
	expectMatchDirect(t, Set{"baz": "blah"}, labelset)
	expectMatchDirect(t, Set{"foo": "bar", "baz": "blah"}, labelset)
	expectNoMatchDirect(t, Set{"foo": "=blah"}, labelset)
	expectNoMatchDirect(t, Set{"baz": "=bar"}, labelset)
	expectNoMatchDirect(t, Set{"foo": "=bar", "foobar": "bar", "baz": "blah"}, labelset)
}

func TestNilMapIsValid(t *testing.T) {
	selector := Set(nil).AsSelector()
	if selector == nil {
		t.Errorf("Selector for nil set should be Everything")
	}
	if !selector.Empty() {
		t.Errorf("Selector for nil set should be Empty")
	}
}

func TestSetIsEmpty(t *testing.T) {
	if !(Set{}).AsSelector().Empty() {
		t.Errorf("Empty set should be empty")
	}

	selector := NewSelector()
	if !selector.Empty() {
		t.Errorf("Empty set should be empty")
	}

	hasSelector := NewSelector()
	hasRequirement, err := NewRequirement("metadata.name", selectors.ExistsOperator, sets.NewString())
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	hasSelector = hasSelector.Add(*hasRequirement)
	if hasSelector.Empty() {
		t.Errorf("hasSelector should not be empty")
	}

	notHasSelector := NewSelector()
	notHasRequirement, err := NewRequirement("metadata.name", selectors.DoesNotExistOperator, sets.NewString())
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	notHasSelector = notHasSelector.Add(*notHasRequirement)
	if notHasSelector.Empty() {
		t.Errorf("notHasSelector should not be empty")
	}

	equalsSelector := NewSelector()
	equalsRequirement, err := NewRequirement("metadata.name", selectors.EqualsOperator, sets.NewString("test-name"))
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	equalsSelector = equalsSelector.Add(*equalsRequirement)
	if equalsSelector.Empty() {
		t.Errorf("equalsSelector should not be empty")
	}
}

func TestRequiresExactMatch(t *testing.T) {
	testCases := []struct {
		fieldSelector string
		label         string
		value         string
		found         bool
	}{
		{
			fieldSelector: "x=a",
			label:         "x",
			value:         "a",
			found:         true,
		},
		{
			fieldSelector: "y=a",
			label:         "x",
			value:         "",
			found:         false,
		},
		{
			fieldSelector: "x!=a",
			label:         "x",
			found:         false,
		},
		{
			fieldSelector: "x==a,b!=y",
			label:         "x",
			value:         "a",
			found:         true,
		},
		{
			fieldSelector: "x=a,b!=y",
			label:         "b",
			found:         false,
		},
	}
	for _, tc := range testCases {
		selector := ParseSelectorOrDie(tc.fieldSelector)
		value, found := selector.RequiresExactMatch(tc.label)
		if value != tc.value {
			t.Errorf("%s: expected value %s, got %s", tc.fieldSelector, tc.value, value)
		}
		if found != tc.found {
			t.Errorf("%s: expected found %t, got %t", tc.fieldSelector, tc.found, found)
		}
	}
}
