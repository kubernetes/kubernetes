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
