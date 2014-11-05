/*
Copyright 2014 Google Inc. All rights reserved.

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

package labels

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	if !(andTerm(nil)).Empty() {
		t.Errorf("Nil andTerm should be empty")
	}
	if (&hasTerm{}).Empty() {
		t.Errorf("hasTerm should not be empty")
	}
	if (&notHasTerm{}).Empty() {
		t.Errorf("notHasTerm should not be empty")
	}
	if !(andTerm{andTerm{}}).Empty() {
		t.Errorf("Nested andTerm should be empty")
	}
	if (andTerm{&hasTerm{"a", "b"}}).Empty() {
		t.Errorf("Nested andTerm should not be empty")
	}
}

func TestRequiresExactMatch(t *testing.T) {
	testCases := map[string]struct {
		S     Selector
		Label string
		Value string
		Found bool
	}{
		"empty set":                 {Set{}.AsSelector(), "test", "", false},
		"nil andTerm":               {andTerm(nil), "test", "", false},
		"empty hasTerm":             {&hasTerm{}, "test", "", false},
		"skipped hasTerm":           {&hasTerm{"a", "b"}, "test", "", false},
		"valid hasTerm":             {&hasTerm{"test", "b"}, "test", "b", true},
		"valid hasTerm no value":    {&hasTerm{"test", ""}, "test", "", true},
		"valid notHasTerm":          {&notHasTerm{"test", "b"}, "test", "", false},
		"valid notHasTerm no value": {&notHasTerm{"test", ""}, "test", "", false},
		"nested andTerm":            {andTerm{andTerm{}}, "test", "", false},
		"nested andTerm matches":    {andTerm{&hasTerm{"test", "b"}}, "test", "b", true},
		"andTerm with non-match":    {andTerm{&hasTerm{}, &hasTerm{"test", "b"}}, "test", "b", true},
	}
	for k, v := range testCases {
		value, found := v.S.RequiresExactMatch(v.Label)
		if value != v.Value {
			t.Errorf("%s: expected value %v, got %s", k, v.Value, value)
		}
		if found != v.Found {
			t.Errorf("%s: expected found %v, got %s", k, v.Found, found)
		}
	}
}

func TestRequirementConstructor(t *testing.T) {
	requirementConstructorTests := []struct {
		Key     string
		Op      Operator
		Vals    util.StringSet
		Success bool
	}{
		{"x", 8, util.NewStringSet("foo"), false},
		{"x", In, nil, false},
		{"x", NotIn, util.NewStringSet(), false},
		{"x", In, util.NewStringSet("foo"), true},
		{"x", NotIn, util.NewStringSet("foo"), true},
		{"x", Exists, nil, true},
		{"abcdefghijklmnopqrstuvwxy", Exists, nil, false}, //breaks DNS952 rule that len(key) < 25
		{"1foo", In, util.NewStringSet("bar"), false},     //breaks DNS952 rule that keys start with [a-z]
	}
	for _, rc := range requirementConstructorTests {
		if _, err := NewRequirement(rc.Key, rc.Op, rc.Vals); err == nil && !rc.Success {
			t.Errorf("expected error with key:%#v op:%v vals:%v, got no error", rc.Key, rc.Op, rc.Vals)
		} else if err != nil && rc.Success {
			t.Errorf("expected no error with key:%#v op:%v vals:%v, got:%v", rc.Key, rc.Op, rc.Vals, err)
		}
	}
}

func TestToString(t *testing.T) {
	var req Requirement
	toStringTests := []struct {
		In    *LabelSelector
		Out   string
		Valid bool
	}{
		{&LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet("abc", "def"), t),
			getRequirement("y", NotIn, util.NewStringSet("jkl"), t),
			getRequirement("z", Exists, nil, t),
		}}, "x in (abc,def),y not in (jkl),z", true},
		{&LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet("abc", "def"), t),
			req,
		}}, "", false},
		{&LabelSelector{Requirements: []Requirement{
			getRequirement("x", NotIn, util.NewStringSet("abc"), t),
			getRequirement("y", In, util.NewStringSet("jkl", "mno"), t),
			getRequirement("z", NotIn, util.NewStringSet(""), t),
		}}, "x not in (abc),y in (jkl,mno),z not in ()", true},
	}
	for _, ts := range toStringTests {
		if out, err := ts.In.String(); err != nil && ts.Valid {
			t.Errorf("%+v.String() => %v, expected no error", ts.In, err)
		} else if out != ts.Out {
			t.Errorf("%+v.String() => %v, want %v", ts.In, out, ts.Out)
		}
	}
}

func TestRequirementLabelSelectorMatching(t *testing.T) {
	var req Requirement
	labelSelectorMatchingTests := []struct {
		Set   Set
		Sel   *LabelSelector
		Match bool
		Valid bool
	}{
		{Set{"x": "foo", "y": "baz"}, &LabelSelector{Requirements: []Requirement{
			req,
		}}, false, false},
		{Set{"x": "foo", "y": "baz"}, &LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet("foo"), t),
			getRequirement("y", NotIn, util.NewStringSet("alpha"), t),
		}}, true, true},
		{Set{"x": "foo", "y": "baz"}, &LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet("foo"), t),
			getRequirement("y", In, util.NewStringSet("alpha"), t),
		}}, false, true},
		{Set{"y": ""}, &LabelSelector{Requirements: []Requirement{
			getRequirement("x", NotIn, util.NewStringSet(""), t),
			getRequirement("y", Exists, nil, t),
		}}, true, true},
		{Set{"y": "baz"}, &LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet(""), t),
		}}, false, true},
	}
	for _, lsm := range labelSelectorMatchingTests {
		if match, err := lsm.Sel.Matches(lsm.Set); err != nil && lsm.Valid {
			t.Errorf("%+v.Matches(%#v) => %v, expected no error", lsm.Sel, lsm.Set, err)
		} else if match != lsm.Match {
			t.Errorf("%+v.Matches(%#v) => %v, want %v", lsm.Sel, lsm.Set, match, lsm.Match)
		}
	}
}

func TestSetSelectorParser(t *testing.T) {
	setSelectorParserTests := []struct {
		In    string
		Out   SetBasedSelector
		Match bool
		Valid bool
	}{
		{"", &LabelSelector{Requirements: nil}, true, true},
		{"x", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", Exists, nil, t),
		}}, true, true},
		{"foo in (abc)", &LabelSelector{Requirements: []Requirement{
			getRequirement("foo", In, util.NewStringSet("abc"), t),
		}}, true, true},
		{"x not in (abc)", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", NotIn, util.NewStringSet("abc"), t),
		}}, true, true},
		{"x not in (abc,def)", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", NotIn, util.NewStringSet("abc", "def"), t),
		}}, true, true},
		{"x in (abc,def)", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet("abc", "def"), t),
		}}, true, true},
		{"x in (abc,)", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet("abc", ""), t),
		}}, true, true},
		{"x in ()", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", In, util.NewStringSet(""), t),
		}}, true, true},
		{"x not in (abc,,def),bar,z in (),w", &LabelSelector{Requirements: []Requirement{
			getRequirement("x", NotIn, util.NewStringSet("abc", "", "def"), t),
			getRequirement("bar", Exists, nil, t),
			getRequirement("z", In, util.NewStringSet(""), t),
			getRequirement("w", Exists, nil, t),
		}}, true, true},
		{"x,y in (a)", &LabelSelector{Requirements: []Requirement{
			getRequirement("y", In, util.NewStringSet("a"), t),
			getRequirement("x", Exists, nil, t),
		}}, false, true},
		{"x,,y", nil, true, false},
		{",x,y", nil, true, false},
		{"x, y", nil, true, false},
		{"x nott in (y)", nil, true, false},
		{"x not in ( )", nil, true, false},
		{"x not in (, a)", nil, true, false},
		{"a in (xyz),", nil, true, false},
		{"a in (xyz)b not in ()", nil, true, false},
		{"a ", nil, true, false},
		{"a not in(", nil, true, false},
	}
	for _, ssp := range setSelectorParserTests {
		if sel, err := Parse(ssp.In); err != nil && ssp.Valid {
			t.Errorf("Parse(%s) => %v expected no error", ssp.In, err)
		} else if err == nil && !ssp.Valid {
			t.Errorf("Parse(%s) => %+v expected error", ssp.In, sel)
		} else if ssp.Match && !reflect.DeepEqual(sel, ssp.Out) {
			t.Errorf("parse output %+v doesn't match %+v, expected match", sel, ssp.Out)
		}
	}
}

func getRequirement(key string, op Operator, vals util.StringSet, t *testing.T) Requirement {
	req, err := NewRequirement(key, op, vals)
	if err != nil {
		t.Errorf("NewRequirement(%v, %v, %v) resulted in error:%v", key, op, vals, err)
	}
	return *req
}
