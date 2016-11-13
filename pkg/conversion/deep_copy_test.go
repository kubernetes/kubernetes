/*
Copyright 2015 The Kubernetes Authors.

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

package conversion

import (
	"math/rand"
	"testing"

	"github.com/google/gofuzz"
)

func TestDeepCopy(t *testing.T) {
	semantic := EqualitiesOrDie()
	x := 42
	table := []TestObject{
		&TestStruct{Map: map[string]string{}},
		&TestStruct{Int: int(5)},
		&TestStruct{Pointer: nil},
		&TestStruct{Pointer: &x},
		&TestStruct{String: "hello world"},
		&TestStruct{Struct: TestSubStruct{}},
		&TestStruct{StructPointer: &TestSubStruct{X: []int{1}}},
		&TestStruct{StructSlice: []*TestSubStruct{
			{X: []int{1}},
			{X: []int{2}},
		}},
		&TestStruct{StructMap: map[string]*TestSubStruct{
			"A": {X: []int{1}},
			"B": {X: []int{2}},
		}},
	}
	for _, obj := range table {
		obj2 := obj.DeepCopyTestObject()
		if e, a := obj, obj2; !semantic.DeepEqual(e, a) {
			t.Errorf("expected %#v\ngot %#v", e, a)
		}
	}
}

func TestDeepCopyFuzz(t *testing.T) {
	semantic := EqualitiesOrDie()
	f := fuzz.New().NilChance(.5).NumElements(0, 100)
	for x := 0; x < 100; x++ {
		obj := &TestStruct{}
		f.Fuzz(obj)
		obj2 := obj.DeepCopy()
		if e, a := obj, obj2; !semantic.DeepEqual(e, a) {
			t.Errorf("expected %#v\ngot %#v", e, a)
		}
	}
}

func TestDeepCopySliceSeparate(t *testing.T) {
	x := &TestStruct{Struct: TestSubStruct{X: []int{5}}}
	y := x.DeepCopy()
	x.Struct.X[0] = 3
	if y.Struct.X[0] == 3 {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyMapSeparate(t *testing.T) {
	x := &TestStruct{Map: map[string]string{"foo": "bar"}}
	y := x.DeepCopy()
	x.Map["foo"] = "abc"
	if y.Map["foo"] == "abc" {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyPointerSeparate(t *testing.T) {
	z := 5
	x := &TestStruct{Pointer: &z}
	y := x.DeepCopy()
	*x.Pointer = 3
	if *y.Pointer == 3 {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyStruct(t *testing.T) {
	x := &TestStruct{Struct: TestSubStruct{A: TestSubSubStruct{E: 1}}}
	y := x.DeepCopy()
	x.Struct.A.E = 3
	x.Struct.B.E = 4
	if y.Struct.A.E != 1 || y.Struct.B.E != 0 {
		t.Errorf("deep copy wasn't deep: %#v, %#v", x, y)
	}
}

var result TestObject

func BenchmarkDeepCopy(b *testing.B) {
	x := 42
	table := []TestObject{
		&TestStruct{Map: map[string]string{}},
		&TestStruct{Int: int(5)},
		&TestStruct{Pointer: nil},
		&TestStruct{Pointer: &x},
		&TestStruct{String: "hello world"},
		&TestStruct{Struct: TestSubStruct{}},
		&TestStruct{StructPointer: &TestSubStruct{X: []int{1}}},
		&TestStruct{StructSlice: []*TestSubStruct{
			{X: []int{1}},
			{X: []int{2}},
		}},
		&TestStruct{StructMap: map[string]*TestSubStruct{
			"A": {X: []int{1}},
			"B": {X: []int{2}},
		}},
	}

	f := fuzz.New().RandSource(rand.NewSource(1)).NilChance(.5).NumElements(0, 100)
	for i := range table {
		f.Fuzz(table[i])
	}

	b.ResetTimer()
	var r TestObject
	for i := 0; i < b.N; i++ {
		for j := range table {
			r = table[j].DeepCopyTestObject()
		}
	}
	result = r
}
