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

package fuzz

import (
	"reflect"
	"testing"
	"time"
)

func TestFuzz_basic(t *testing.T) {
	obj := &struct {
		I    int
		I8   int8
		I16  int16
		I32  int32
		I64  int64
		U    uint
		U8   uint8
		U16  uint16
		U32  uint32
		U64  uint64
		Uptr uintptr
		S    string
		B    bool
		T    time.Time
	}{}

	failed := map[string]int{}
	for i := 0; i < 10; i++ {
		New().Fuzz(obj)

		if n, v := "i", obj.I; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "i8", obj.I8; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "i16", obj.I16; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "i32", obj.I32; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "i64", obj.I64; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "u", obj.U; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "u8", obj.U8; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "u16", obj.U16; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "u32", obj.U32; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "u64", obj.U64; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "uptr", obj.Uptr; v == 0 {
			failed[n] = failed[n] + 1
		}
		if n, v := "s", obj.S; v == "" {
			failed[n] = failed[n] + 1
		}
		if n, v := "b", obj.B; v == false {
			failed[n] = failed[n] + 1
		}
		if n, v := "t", obj.T; v.IsZero() {
			failed[n] = failed[n] + 1
		}
	}
	checkFailed(t, failed)
}

func checkFailed(t *testing.T, failed map[string]int) {
	for k, v := range failed {
		if v > 8 {
			t.Errorf("%v seems to not be getting set, was zero value %v times", k, v)
		}
	}
}

func TestFuzz_structptr(t *testing.T) {
	obj := &struct {
		A *struct {
			S string
		}
	}{}

	f := New().NilChance(.5)
	failed := map[string]int{}
	for i := 0; i < 10; i++ {
		f.Fuzz(obj)

		if n, v := "a not nil", obj.A; v == nil {
			failed[n] = failed[n] + 1
		}
		if n, v := "a nil", obj.A; v != nil {
			failed[n] = failed[n] + 1
		}
		if n, v := "as", obj.A; v == nil || v.S == "" {
			failed[n] = failed[n] + 1
		}
	}
	checkFailed(t, failed)
}

// tryFuzz tries fuzzing up to 20 times. Fail if check() never passes, report the highest
// stage it ever got to.
func tryFuzz(t *testing.T, f *Fuzzer, obj interface{}, check func() (stage int, passed bool)) {
	maxStage := 0
	for i := 0; i < 20; i++ {
		f.Fuzz(obj)
		stage, passed := check()
		if stage > maxStage {
			maxStage = stage
		}
		if passed {
			return
		}
	}
	t.Errorf("Only ever got to stage %v", maxStage)
}

func TestFuzz_structmap(t *testing.T) {
	obj := &struct {
		A map[struct {
			S string
		}]struct {
			S2 string
		}
		B map[string]string
	}{}

	tryFuzz(t, New(), obj, func() (int, bool) {
		if obj.A == nil {
			return 1, false
		}
		if len(obj.A) == 0 {
			return 2, false
		}
		for k, v := range obj.A {
			if k.S == "" {
				return 3, false
			}
			if v.S2 == "" {
				return 4, false
			}
		}

		if obj.B == nil {
			return 5, false
		}
		if len(obj.B) == 0 {
			return 6, false
		}
		for k, v := range obj.B {
			if k == "" {
				return 7, false
			}
			if v == "" {
				return 8, false
			}
		}
		return 9, true
	})
}

func TestFuzz_structslice(t *testing.T) {
	obj := &struct {
		A []struct {
			S string
		}
		B []string
	}{}

	tryFuzz(t, New(), obj, func() (int, bool) {
		if obj.A == nil {
			return 1, false
		}
		if len(obj.A) == 0 {
			return 2, false
		}
		for _, v := range obj.A {
			if v.S == "" {
				return 3, false
			}
		}

		if obj.B == nil {
			return 4, false
		}
		if len(obj.B) == 0 {
			return 5, false
		}
		for _, v := range obj.B {
			if v == "" {
				return 6, false
			}
		}
		return 7, true
	})
}

func TestFuzz_structarray(t *testing.T) {
	obj := &struct {
		A [3]struct {
			S string
		}
		B [2]int
	}{}

	tryFuzz(t, New(), obj, func() (int, bool) {
		for _, v := range obj.A {
			if v.S == "" {
				return 1, false
			}
		}

		for _, v := range obj.B {
			if v == 0 {
				return 2, false
			}
		}
		return 3, true
	})
}

func TestFuzz_custom(t *testing.T) {
	obj := &struct {
		A string
		B *string
		C map[string]string
		D *map[string]string
	}{}

	testPhrase := "gotcalled"
	testMap := map[string]string{"C": "D"}
	f := New().Funcs(
		func(s *string, c Continue) {
			*s = testPhrase
		},
		func(m map[string]string, c Continue) {
			m["C"] = "D"
		},
	)

	tryFuzz(t, f, obj, func() (int, bool) {
		if obj.A != testPhrase {
			return 1, false
		}
		if obj.B == nil {
			return 2, false
		}
		if *obj.B != testPhrase {
			return 3, false
		}
		if e, a := testMap, obj.C; !reflect.DeepEqual(e, a) {
			return 4, false
		}
		if obj.D == nil {
			return 5, false
		}
		if e, a := testMap, *obj.D; !reflect.DeepEqual(e, a) {
			return 6, false
		}
		return 7, true
	})
}

type SelfFuzzer string

// Implement fuzz.Interface.
func (sf *SelfFuzzer) Fuzz(c Continue) {
	*sf = selfFuzzerTestPhrase
}

const selfFuzzerTestPhrase = "was fuzzed"

func TestFuzz_interface(t *testing.T) {
	f := New()

	var obj1 SelfFuzzer
	tryFuzz(t, f, &obj1, func() (int, bool) {
		if obj1 != selfFuzzerTestPhrase {
			return 1, false
		}
		return 1, true
	})

	var obj2 map[int]SelfFuzzer
	tryFuzz(t, f, &obj2, func() (int, bool) {
		for _, v := range obj2 {
			if v != selfFuzzerTestPhrase {
				return 1, false
			}
		}
		return 1, true
	})
}

func TestFuzz_interfaceAndFunc(t *testing.T) {
	const privateTestPhrase = "private phrase"
	f := New().Funcs(
		// This should take precedence over SelfFuzzer.Fuzz().
		func(s *SelfFuzzer, c Continue) {
			*s = privateTestPhrase
		},
	)

	var obj1 SelfFuzzer
	tryFuzz(t, f, &obj1, func() (int, bool) {
		if obj1 != privateTestPhrase {
			return 1, false
		}
		return 1, true
	})

	var obj2 map[int]SelfFuzzer
	tryFuzz(t, f, &obj2, func() (int, bool) {
		for _, v := range obj2 {
			if v != privateTestPhrase {
				return 1, false
			}
		}
		return 1, true
	})
}

func TestFuzz_noCustom(t *testing.T) {
	type Inner struct {
		Str string
	}
	type Outer struct {
		Str string
		In  Inner
	}

	testPhrase := "gotcalled"
	f := New().Funcs(
		func(outer *Outer, c Continue) {
			outer.Str = testPhrase
			c.Fuzz(&outer.In)
		},
		func(inner *Inner, c Continue) {
			inner.Str = testPhrase
		},
	)
	c := Continue{f: f, Rand: f.r}

	// Fuzzer.Fuzz()
	obj1 := Outer{}
	f.Fuzz(&obj1)
	if obj1.Str != testPhrase {
		t.Errorf("expected Outer custom function to have been called")
	}
	if obj1.In.Str != testPhrase {
		t.Errorf("expected Inner custom function to have been called")
	}

	// Continue.Fuzz()
	obj2 := Outer{}
	c.Fuzz(&obj2)
	if obj2.Str != testPhrase {
		t.Errorf("expected Outer custom function to have been called")
	}
	if obj2.In.Str != testPhrase {
		t.Errorf("expected Inner custom function to have been called")
	}

	// Fuzzer.FuzzNoCustom()
	obj3 := Outer{}
	f.FuzzNoCustom(&obj3)
	if obj3.Str == testPhrase {
		t.Errorf("expected Outer custom function to not have been called")
	}
	if obj3.In.Str != testPhrase {
		t.Errorf("expected Inner custom function to have been called")
	}

	// Continue.FuzzNoCustom()
	obj4 := Outer{}
	c.FuzzNoCustom(&obj4)
	if obj4.Str == testPhrase {
		t.Errorf("expected Outer custom function to not have been called")
	}
	if obj4.In.Str != testPhrase {
		t.Errorf("expected Inner custom function to have been called")
	}
}

func TestFuzz_NumElements(t *testing.T) {
	f := New().NilChance(0).NumElements(0, 1)
	obj := &struct {
		A []int
	}{}

	tryFuzz(t, f, obj, func() (int, bool) {
		if obj.A == nil {
			return 1, false
		}
		return 2, len(obj.A) == 0
	})
	tryFuzz(t, f, obj, func() (int, bool) {
		if obj.A == nil {
			return 3, false
		}
		return 4, len(obj.A) == 1
	})
}
