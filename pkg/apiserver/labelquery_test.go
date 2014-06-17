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

package apiserver

import (
	"testing"
)

func TestLabelQueryParse(t *testing.T) {
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
		lq, err := ParseLabelQuery(test)
		if err != nil {
			t.Errorf("%v: error %v (%#v)\n", test, err, err)
		}
		if test != lq.String() {
			t.Errorf("%v restring gave: %v\n", test, lq.String())
		}
	}
	for _, test := range testBadStrings {
		_, err := ParseLabelQuery(test)
		if err == nil {
			t.Errorf("%v: did not get expected error\n", test)
		}
	}
}

func shouldMatch(t *testing.T, query string, ls LabelSet) {
	lq, err := ParseLabelQuery(query)
	if err != nil {
		t.Errorf("Unable to parse %v as a query\n", query)
		return
	}
	if !lq.Matches(ls) {
		t.Errorf("Wanted %s to match %s, but it did not.\n", query, ls)
	}
}

func shouldNotMatch(t *testing.T, query string, ls LabelSet) {
	lq, err := ParseLabelQuery(query)
	if err != nil {
		t.Errorf("Unable to parse %v as a query\n", query)
		return
	}
	if lq.Matches(ls) {
		t.Errorf("Wanted '%s' to not match %s, but it did.", query, ls)
	}
}

func TestSimpleLabel(t *testing.T) {
	shouldMatch(t, "", LabelSet{"x": "y"})
	shouldMatch(t, "x=y", LabelSet{"x": "y"})
	shouldMatch(t, "x=y,z=w", LabelSet{"x": "y", "z": "w"})
	shouldMatch(t, "x!=y,z!=w", LabelSet{"x": "z", "z": "a"})
	shouldNotMatch(t, "x=y", LabelSet{"x": "z"})
	shouldNotMatch(t, "x=y,z=w", LabelSet{"x": "w", "z": "w"})
	shouldNotMatch(t, "x!=y,z!=w", LabelSet{"x": "z", "z": "w"})
}
