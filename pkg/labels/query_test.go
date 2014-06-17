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
	"testing"
)

func TestQueryParse(t *testing.T) {
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
		lq, err := ParseQuery(test)
		if err != nil {
			t.Errorf("%v: error %v (%#v)\n", test, err, err)
		}
		if test != lq.String() {
			t.Errorf("%v restring gave: %v\n", test, lq.String())
		}
	}
	for _, test := range testBadStrings {
		_, err := ParseQuery(test)
		if err == nil {
			t.Errorf("%v: did not get expected error\n", test)
		}
	}
}

func expectMatch(t *testing.T, query string, ls LabelSet) {
	lq, err := ParseQuery(query)
	if err != nil {
		t.Errorf("Unable to parse %v as a query\n", query)
		return
	}
	if !lq.Matches(ls) {
		t.Errorf("Wanted %s to match '%s', but it did not.\n", query, ls)
	}
}

func expectNoMatch(t *testing.T, query string, ls LabelSet) {
	lq, err := ParseQuery(query)
	if err != nil {
		t.Errorf("Unable to parse %v as a query\n", query)
		return
	}
	if lq.Matches(ls) {
		t.Errorf("Wanted '%s' to not match '%s', but it did.", query, ls)
	}
}

func TestEverything(t *testing.T) {
	if !Everything().Matches(LabelSet{"x": "y"}) {
		t.Errorf("Nil query didn't match")
	}
}

func TestLabelQueryMatches(t *testing.T) {
	expectMatch(t, "", LabelSet{"x": "y"})
	expectMatch(t, "x=y", LabelSet{"x": "y"})
	expectMatch(t, "x=y,z=w", LabelSet{"x": "y", "z": "w"})
	expectMatch(t, "x!=y,z!=w", LabelSet{"x": "z", "z": "a"})
	expectNoMatch(t, "x=y", LabelSet{"x": "z"})
	expectNoMatch(t, "x=y,z=w", LabelSet{"x": "w", "z": "w"})
	expectNoMatch(t, "x!=y,z!=w", LabelSet{"x": "z", "z": "w"})

	labelset := LabelSet{
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

func expectMatchDirect(t *testing.T, query, ls LabelSet) {
	if !QueryFromSet(query).Matches(ls) {
		t.Errorf("Wanted %s to match '%s', but it did not.\n", query, ls)
	}
}

func expectNoMatchDirect(t *testing.T, query, ls LabelSet) {
	if QueryFromSet(query).Matches(ls) {
		t.Errorf("Wanted '%s' to not match '%s', but it did.", query, ls)
	}
}

func TestLabelSetMatches(t *testing.T) {
	labelset := LabelSet{
		"foo": "bar",
		"baz": "blah",
	}
	expectMatchDirect(t, LabelSet{}, labelset)
	expectMatchDirect(t, LabelSet{"foo": "bar"}, labelset)
	expectMatchDirect(t, LabelSet{"baz": "blah"}, labelset)
	expectMatchDirect(t, LabelSet{"foo": "bar", "baz": "blah"}, labelset)
	expectNoMatchDirect(t, LabelSet{"foo": "=blah"}, labelset)
	expectNoMatchDirect(t, LabelSet{"baz": "=bar"}, labelset)
	expectNoMatchDirect(t, LabelSet{"foo": "=bar", "foobar": "bar", "baz": "blah"}, labelset)
}
