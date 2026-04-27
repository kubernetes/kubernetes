/*
Copyright 2014 The Kubernetes Authors.

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

func matches(t *testing.T, ls Set, want string) {
	if ls.String() != want {
		t.Errorf("Expected '%s', but got '%s'", want, ls.String())
	}
}

func TestSetString(t *testing.T) {
	matches(t, Set{"x": "y"}, "x=y")
	matches(t, Set{"foo": "bar"}, "foo=bar")
	matches(t, Set{"foo": "bar", "baz": "qup"}, "baz=qup,foo=bar")

	// TODO: Make our label representation robust enough to handle labels
	// with ",=!" characters in their names.
}

func TestLabelHas(t *testing.T) {
	labelHasTests := []struct {
		Ls  Labels
		Key string
		Has bool
	}{
		{Set{"x": "y"}, "x", true},
		{Set{"x": ""}, "x", true},
		{Set{"x": "y"}, "foo", false},
	}
	for _, lh := range labelHasTests {
		if has := lh.Ls.Has(lh.Key); has != lh.Has {
			t.Errorf("%#v.Has(%#v) => %v, expected %v", lh.Ls, lh.Key, has, lh.Has)
		}
	}
}

func TestLabelGet(t *testing.T) {
	ls := Set{"x": "y"}
	if ls.Get("x") != "y" {
		t.Errorf("Set.Get is broken")
	}
}

func TestLabelConflict(t *testing.T) {
	tests := []struct {
		labels1  map[string]string
		labels2  map[string]string
		conflict bool
	}{
		{
			labels1:  map[string]string{},
			labels2:  map[string]string{},
			conflict: false,
		},
		{
			labels1:  map[string]string{"env": "test"},
			labels2:  map[string]string{"infra": "true"},
			conflict: false,
		},
		{
			labels1:  map[string]string{"env": "test"},
			labels2:  map[string]string{"infra": "true", "env": "test"},
			conflict: false,
		},
		{
			labels1:  map[string]string{"env": "test"},
			labels2:  map[string]string{"env": "dev"},
			conflict: true,
		},
		{
			labels1:  map[string]string{"env": "test", "infra": "false"},
			labels2:  map[string]string{"infra": "true", "color": "blue"},
			conflict: true,
		},
	}
	for _, test := range tests {
		conflict := Conflicts(Set(test.labels1), Set(test.labels2))
		if conflict != test.conflict {
			t.Errorf("expected: %v but got: %v", test.conflict, conflict)
		}
	}
}

func TestLabelMerge(t *testing.T) {
	tests := []struct {
		labels1      map[string]string
		labels2      map[string]string
		mergedLabels map[string]string
	}{
		{
			labels1:      map[string]string{},
			labels2:      map[string]string{},
			mergedLabels: map[string]string{},
		},
		{
			labels1:      map[string]string{"infra": "true"},
			labels2:      map[string]string{},
			mergedLabels: map[string]string{"infra": "true"},
		},
		{
			labels1:      map[string]string{"infra": "true"},
			labels2:      map[string]string{"env": "test", "color": "blue"},
			mergedLabels: map[string]string{"infra": "true", "env": "test", "color": "blue"},
		},
	}
	for _, test := range tests {
		mergedLabels := Merge(Set(test.labels1), Set(test.labels2))
		if !Equals(mergedLabels, test.mergedLabels) {
			t.Errorf("expected: %v but got: %v", test.mergedLabels, mergedLabels)
		}
	}
}

func TestLabelSelectorParse(t *testing.T) {
	tests := []struct {
		selector string
		labels   map[string]string
		valid    bool
	}{
		{
			selector: "",
			labels:   map[string]string{},
			valid:    true,
		},
		{
			selector: "x=a",
			labels:   map[string]string{"x": "a"},
			valid:    true,
		},
		{
			selector: "x=a,y=b,z=c",
			labels:   map[string]string{"x": "a", "y": "b", "z": "c"},
			valid:    true,
		},
		{
			selector: " x = a , y = b , z = c ",
			labels:   map[string]string{"x": "a", "y": "b", "z": "c"},
			valid:    true,
		},
		{
			selector: "color=green,env=test,service=front",
			labels:   map[string]string{"color": "green", "env": "test", "service": "front"},
			valid:    true,
		},
		{
			selector: "color=green, env=test, service=front",
			labels:   map[string]string{"color": "green", "env": "test", "service": "front"},
			valid:    true,
		},
		{
			selector: ",",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x,y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x=$y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x!=y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x==y",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x=a||y=b",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x in (y)",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x notin (y)",
			labels:   map[string]string{},
			valid:    false,
		},
		{
			selector: "x y",
			labels:   map[string]string{},
			valid:    false,
		},
	}
	for _, test := range tests {
		labels, err := ConvertSelectorToLabelsMap(test.selector)
		if test.valid && err != nil {
			t.Errorf("selector: %s, expected no error but got: %s", test.selector, err)
		} else if !test.valid && err == nil {
			t.Errorf("selector: %s, expected an error", test.selector)
		}

		if !Equals(Set(labels), test.labels) {
			t.Errorf("expected: %s but got: %s", test.labels, labels)
		}
	}
}
