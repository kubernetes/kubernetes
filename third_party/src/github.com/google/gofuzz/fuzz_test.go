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
