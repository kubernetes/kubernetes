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
	"reflect"
	"testing"

	"github.com/google/gofuzz"
)

func TestDeepCopy(t *testing.T) {
	semantic := EqualitiesOrDie()
	f := fuzz.New().NilChance(.5).NumElements(0, 100)
	table := []interface{}{
		map[string]string{},
		int(5),
		"hello world",
		struct {
			A, B, C struct {
				D map[string]int
			}
			X []int
			Y []byte
		}{},
	}
	for _, obj := range table {
		obj2, err := NewCloner().DeepCopy(obj)
		if err != nil {
			t.Errorf("Error: couldn't copy %#v", obj)
			continue
		}
		if e, a := obj, obj2; !semantic.DeepEqual(e, a) {
			t.Errorf("expected %#v\ngot %#v", e, a)
		}

		obj3 := reflect.New(reflect.TypeOf(obj)).Interface()
		f.Fuzz(obj3)
		obj4, err := NewCloner().DeepCopy(obj3)
		if err != nil {
			t.Errorf("Error: couldn't copy %#v", obj)
			continue
		}
		if e, a := obj3, obj4; !semantic.DeepEqual(e, a) {
			t.Errorf("expected %#v\ngot %#v", e, a)
		}
		f.Fuzz(obj3)
	}
}

func copyOrDie(t *testing.T, in interface{}) interface{} {
	out, err := NewCloner().DeepCopy(in)
	if err != nil {
		t.Fatalf("DeepCopy failed: %#q: %v", in, err)
	}
	return out
}

func TestDeepCopySliceSeparate(t *testing.T) {
	x := []int{5}
	y := copyOrDie(t, x).([]int)
	x[0] = 3
	if y[0] == 3 {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyArraySeparate(t *testing.T) {
	x := [1]int{5}
	y := copyOrDie(t, x).([1]int)
	x[0] = 3
	if y[0] == 3 {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyMapSeparate(t *testing.T) {
	x := map[string]int{"foo": 5}
	y := copyOrDie(t, x).(map[string]int)
	x["foo"] = 3
	if y["foo"] == 3 {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyPointerSeparate(t *testing.T) {
	z := 5
	x := &z
	y := copyOrDie(t, x).(*int)
	*x = 3
	if *y == 3 {
		t.Errorf("deep copy wasn't deep: %#q %#q", x, y)
	}
}

func TestDeepCopyStruct(t *testing.T) {
	type Foo struct {
		A int
	}
	type Bar struct {
		Foo
		F *Foo
	}
	a := &Bar{Foo{1}, &Foo{2}}
	b := copyOrDie(t, a).(*Bar)
	a.A = 3
	a.F.A = 4

	if b.A != 1 || b.F.A != 2 {
		t.Errorf("deep copy wasn't deep: %#v, %#v", a, b)
	}
}

var result interface{}

func BenchmarkDeepCopy(b *testing.B) {
	table := []interface{}{
		map[string]string{},
		int(5),
		"hello world",
		struct {
			A, B, C struct {
				D map[string]int
			}
			X []int
			Y []byte
		}{},
	}

	f := fuzz.New().RandSource(rand.NewSource(1)).NilChance(.5).NumElements(0, 100)
	for i := range table {
		out := table[i]
		obj := reflect.New(reflect.TypeOf(out)).Interface()
		f.Fuzz(obj)
		table[i] = obj
	}

	b.ResetTimer()
	var r interface{}
	for i := 0; i < b.N; i++ {
		for j := range table {
			r, _ = NewCloner().DeepCopy(table[j])
		}
	}
	result = r
}
