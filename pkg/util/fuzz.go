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

package util

import (
	"fmt"
	"math/rand"
	"reflect"
)

// Fuzzer knows how to fill any object with random fields.
type Fuzzer struct {
	customFuzz map[reflect.Type]func(reflect.Value)
}

// NewFuzzer returns a new Fuzzer with the given custom fuzzing functions.
//
// Each entry in fuzzFuncs must be a function with one parameter, which will
// be the variable we wish that function to fill with random data. For this
// to be effective, the variable type should be either a pointer or a map.
//
// These functions are called sensibly, e.g., if you wanted custom string
// fuzzing, the function `func(s *string)` would get called and passed the
// address of strings. Maps and pointers will be made/new'd for you. For
// slices, it doesn't make much sense to pre-create them--Fuzzer doesn't
// know how long you want your slice--so take a pointer to a slice, and
// make it yourself. (If you don't want your map/pointer type pre-made,
// take a pointer to it, and make it yourself.)
//
// TODO: Take a source of randomness for deterministic, repeatable fuzzing.
// TODO: Make probability of getting a nil customizable.
func NewFuzzer(fuzzFuncs ...interface{}) *Fuzzer {
	f := &Fuzzer{
		map[reflect.Type]func(reflect.Value){},
	}
	for i := range fuzzFuncs {
		v := reflect.ValueOf(fuzzFuncs[i])
		if v.Kind() != reflect.Func {
			panic("Need only funcs!")
		}
		t := v.Type()
		if t.NumIn() != 1 || t.NumOut() != 0 {
			panic("Need 1 in and 0 out params!")
		}
		argT := t.In(0)
		switch argT.Kind() {
		case reflect.Ptr, reflect.Map:
		default:
			panic("fuzzFunc must take pointer or map type")
		}
		f.customFuzz[argT] = func(toFuzz reflect.Value) {
			if toFuzz.Type().AssignableTo(argT) {
				v.Call([]reflect.Value{toFuzz})
			} else if toFuzz.Type().ConvertibleTo(argT) {
				v.Call([]reflect.Value{toFuzz.Convert(argT)})
			} else {
				panic(fmt.Errorf("%#v neither ConvertibleTo nor AssignableTo %v",
					toFuzz.Interface(),
					argT))
			}
		}
	}
	return f
}

// Fuzz recursively fills all of obj's fields with something random.
// Not safe for cyclic or tree-like structs!
// obj must be a pointer. Only exported (public) fields can be set (thanks, golang :/ )
// Intended for tests, so will panic on bad input or unimplemented fields.
func (f *Fuzzer) Fuzz(obj interface{}) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		panic("needed ptr!")
	}
	v = v.Elem()
	f.doFuzz(v)
}

func (f *Fuzzer) doFuzz(v reflect.Value) {
	if !v.CanSet() {
		return
	}
	// Check for both pointer and non-pointer custom functions.
	if v.CanAddr() && f.tryCustom(v.Addr()) {
		return
	}
	if f.tryCustom(v) {
		return
	}
	if fn, ok := fillFuncMap[v.Kind()]; ok {
		fn(v)
		return
	}
	switch v.Kind() {
	case reflect.Map:
		if rand.Intn(5) > 0 {
			v.Set(reflect.MakeMap(v.Type()))
			n := 1 + rand.Intn(10)
			for i := 0; i < n; i++ {
				key := reflect.New(v.Type().Key()).Elem()
				f.doFuzz(key)
				val := reflect.New(v.Type().Elem()).Elem()
				f.doFuzz(val)
				v.SetMapIndex(key, val)
			}
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Ptr:
		if rand.Intn(5) > 0 {
			v.Set(reflect.New(v.Type().Elem()))
			f.doFuzz(v.Elem())
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Slice:
		if rand.Intn(5) > 0 {
			n := 1 + rand.Intn(10)
			v.Set(reflect.MakeSlice(v.Type(), n, n))
			for i := 0; i < n; i++ {
				f.doFuzz(v.Index(i))
			}
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			f.doFuzz(v.Field(i))
		}
	case reflect.Array:
		fallthrough
	case reflect.Chan:
		fallthrough
	case reflect.Func:
		fallthrough
	case reflect.Interface:
		fallthrough
	default:
		panic(fmt.Sprintf("Can't handle %#v", v.Interface()))
	}
}

// tryCustom searches for custom handlers, and returns true iff it finds a match
// and successfully randomizes v.
func (f Fuzzer) tryCustom(v reflect.Value) bool {
	doCustom, ok := f.customFuzz[v.Type()]
	if !ok {
		return false
	}

	switch v.Kind() {
	case reflect.Ptr:
		if v.IsNil() {
			if !v.CanSet() {
				return false
			}
			v.Set(reflect.New(v.Type().Elem()))
		}
	case reflect.Map:
		if v.IsNil() {
			if !v.CanSet() {
				return false
			}
			v.Set(reflect.MakeMap(v.Type()))
		}
	default:
		return false
	}

	doCustom(v)
	return true
}

func fuzzInt(v reflect.Value) {
	v.SetInt(int64(RandUint64()))
}

func fuzzUint(v reflect.Value) {
	v.SetUint(RandUint64())
}

var fillFuncMap = map[reflect.Kind]func(reflect.Value){
	reflect.Bool: func(v reflect.Value) {
		v.SetBool(RandBool())
	},
	reflect.Int:     fuzzInt,
	reflect.Int8:    fuzzInt,
	reflect.Int16:   fuzzInt,
	reflect.Int32:   fuzzInt,
	reflect.Int64:   fuzzInt,
	reflect.Uint:    fuzzUint,
	reflect.Uint8:   fuzzUint,
	reflect.Uint16:  fuzzUint,
	reflect.Uint32:  fuzzUint,
	reflect.Uint64:  fuzzUint,
	reflect.Uintptr: fuzzUint,
	reflect.Float32: func(v reflect.Value) {
		v.SetFloat(float64(rand.Float32()))
	},
	reflect.Float64: func(v reflect.Value) {
		v.SetFloat(rand.Float64())
	},
	reflect.Complex64: func(v reflect.Value) {
		panic("unimplemented")
	},
	reflect.Complex128: func(v reflect.Value) {
		panic("unimplemented")
	},
	reflect.String: func(v reflect.Value) {
		v.SetString(RandString())
	},
	reflect.UnsafePointer: func(v reflect.Value) {
		panic("unimplemented")
	},
}

// RandBool returns true or false randomly.
func RandBool() bool {
	if rand.Int()&1 == 1 {
		return true
	}
	return false
}

type charRange struct {
	first, last rune
}

// choose returns a random unicode character from the given range.
func (r *charRange) choose() rune {
	count := int64(r.last - r.first)
	return r.first + rune(rand.Int63n(count))
}

var unicodeRanges = []charRange{
	{' ', '~'},           // ASCII characters
	{'\u00a0', '\u02af'}, // Multi-byte encoded characters
	{'\u4e00', '\u9fff'}, // Common CJK (even longer encodings)
}

// RandString makes a random string up to 20 characters long. The returned string
// may include a variety of (valid) UTF-8 encodings. For testing.
func RandString() string {
	n := rand.Intn(20)
	runes := make([]rune, n)
	for i := range runes {
		runes[i] = unicodeRanges[rand.Intn(len(unicodeRanges))].choose()
	}
	return string(runes)
}

// RandUint64 makes random 64 bit numbers.
// Weirdly, rand doesn't have a function that gives you 64 random bits.
func RandUint64() uint64 {
	return uint64(rand.Uint32())<<32 | uint64(rand.Uint32())
}
