/*
Copyright 2018 The Kubernetes Authors.

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

package fieldpath

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/value"
)

var (
	_V = value.NewValueInterface
)

func TestPathString(t *testing.T) {
	table := []struct {
		name   string
		fp     Path
		expect string
	}{
		{"basic1", MakePathOrDie("foo", 1), ".foo[1]"},
		{"basic2", MakePathOrDie("foo", "bar", 1, "baz"), ".foo.bar[1].baz"},
		{"associative-list-ref", MakePathOrDie("foo", KeyByFields(
			// This makes sure we test all types: string,
			// floats, integers and booleans.
			"a", "b",
			"c", 1,
			"d", 1.5,
			"e", true,
		)), `.foo[a="b",c=1,d=1.5,e=true]`},
		{"sets", MakePathOrDie("foo",
			// This makes sure we test all types: string,
			// floats, integers and booleans.
			_V("b"),
			_V(5),
			_V(false),
			_V(3.14159),
		), `.foo[="b"][=5][=false][=3.14159]`},
	}
	for _, tt := range table {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := tt.fp.String()
			if e, a := tt.expect, got; e != a {
				t.Errorf("Wanted %v, but got %v", e, a)
			}
		})
	}
}
