/*
Copyright 2017 The Kubernetes Authors.

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

package testing

import "testing"

func TestValueFuzz(t *testing.T) {
	type (
		Y struct {
			I int
			B bool
			F float32
			U uint
		}
		X struct {
			Ptr   *X
			Y     Y
			Map   map[string]int
			Slice []int
		}
	)

	x := X{
		Ptr:   &X{},
		Map:   map[string]int{"foo": 42},
		Slice: []int{1, 2, 3},
	}

	p := x.Ptr
	m := x.Map
	s := x.Slice

	ValueFuzz(x)

	if x.Ptr.Y.I == 0 {
		t.Errorf("x.Ptr.Y.I should have changed")
	}

	if x.Map["foo"] == 42 {
		t.Errorf("x.Map[foo] should have changed")
	}

	if x.Slice[0] == 1 {
		t.Errorf("x.Slice[0] should have changed")
	}

	if x.Ptr != p {
		t.Errorf("x.Ptr changed")
	}

	m["foo"] = 7
	if x.Map["foo"] != m["foo"] {
		t.Errorf("x.Map changed")
	}
	s[0] = 7
	if x.Slice[0] != s[0] {
		t.Errorf("x.Slice changed")
	}
}
