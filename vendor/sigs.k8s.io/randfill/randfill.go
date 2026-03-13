/*
Copyright 2014 Google Inc. All rights reserved.
Copyright 2014 The gofuzz Authors.
Copyright 2025 The Kubernetes Authors.

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

// Package randfill is a library for populating go objects with random values.
package randfill

import (
	"fmt"
	"math/rand"
	"reflect"
	"regexp"
	"sync"
	"time"
	"unsafe"

	"strings"

	"sigs.k8s.io/randfill/bytesource"
)

// funcMap is a map from a type to a function that randfills that type.  The
// function is a reflect.Value because the type being filled is different for
// each func.
type funcMap map[reflect.Type]reflect.Value

// Filler knows how to fill any object with random fields.
type Filler struct {
	customFuncs           funcMap
	defaultFuncs          funcMap
	r                     *rand.Rand
	nilChance             float64
	minElements           int
	maxElements           int
	maxDepth              int
	allowUnexportedFields bool
	skipFieldPatterns     []*regexp.Regexp

	lock sync.Mutex
}

// New returns a new Filler. Customize your Filler further by calling Funcs,
// RandSource, NilChance, or NumElements in any order.
func New() *Filler {
	return NewWithSeed(time.Now().UnixNano())
}

func NewWithSeed(seed int64) *Filler {
	f := &Filler{
		defaultFuncs: funcMap{
			reflect.TypeOf(&time.Time{}): reflect.ValueOf(randfillTime),
		},

		customFuncs:           funcMap{},
		r:                     rand.New(rand.NewSource(seed)),
		nilChance:             .2,
		minElements:           1,
		maxElements:           10,
		maxDepth:              100,
		allowUnexportedFields: false,
	}
	return f
}

// NewFromGoFuzz is a helper function that enables using randfill (this
// project) with go-fuzz (https://github.com/dvyukov/go-fuzz) for continuous
// fuzzing. Essentially, it enables translating the fuzzing bytes from
// go-fuzz to any Go object using this library.
//
// This implementation promises a constant translation from a given slice of
// bytes to the fuzzed objects. This promise will remain over future
// versions of Go and of this library.
//
// Note: the returned Filler should not be shared between multiple goroutines,
// as its deterministic output will no longer be available.
//
// Example: use go-fuzz to test the function `MyFunc(int)` in the package
// `mypackage`. Add the file: "mypackage_fuzz.go" with the content:
//
// // +build gofuzz
// package mypackage
// import "sigs.k8s.io/randfill"
//
//	func Fuzz(data []byte) int {
//		var i int
//		randfill.NewFromGoFuzz(data).Fill(&i)
//		MyFunc(i)
//		return 0
//	}
func NewFromGoFuzz(data []byte) *Filler {
	return New().RandSource(bytesource.New(data))
}

// Funcs registers custom fill functions for this Filler.
//
// Each entry in customFuncs must be a function taking two parameters.
// The first parameter must be a pointer or map. It is the variable that
// function will fill with random data. The second parameter must be a
// randfill.Continue, which will provide a source of randomness and a way
// to automatically continue filling smaller pieces of the first parameter.
//
// These functions are called sensibly, e.g., if you wanted custom string
// filling, the function `func(s *string, c randfill.Continue)` would get
// called and passed the address of strings. Maps and pointers will always
// be made/new'd for you, ignoring the NilChance option. For slices, it
// doesn't make much sense to pre-create them--Filler doesn't know how
// long you want your slice--so take a pointer to a slice, and make it
// yourself. (If you don't want your map/pointer type pre-made, take a
// pointer to it, and make it yourself.) See the examples for a range of
// custom functions.
//
// If a function is already registered for a type, and a new function is
// provided, the previous function will be replaced with the new one.
func (f *Filler) Funcs(customFuncs ...interface{}) *Filler {
	for i := range customFuncs {
		v := reflect.ValueOf(customFuncs[i])
		if v.Kind() != reflect.Func {
			panic("Filler.Funcs: all arguments must be functions")
		}
		t := v.Type()
		if t.NumIn() != 2 || t.NumOut() != 0 {
			panic("Filler.Funcs: all customFuncs must have 2 arguments and 0 returns")
		}
		argT := t.In(0)
		switch argT.Kind() {
		case reflect.Ptr, reflect.Map:
		default:
			panic("Filler.Funcs: customFuncs' first argument must be a pointer or map type")
		}
		if t.In(1) != reflect.TypeOf(Continue{}) {
			panic("Filler.Funcs: customFuncs' second argument must be a randfill.Continue")
		}
		f.customFuncs[argT] = v
	}
	return f
}

// RandSource causes this Filler to get values from the given source of
// randomness. Use this if you want deterministic filling.
func (f *Filler) RandSource(s rand.Source) *Filler {
	f.r = rand.New(s)
	return f
}

// NilChance sets the probability of creating a nil pointer, map, or slice to
// 'p'. 'p' should be between 0 (no nils) and 1 (all nils), inclusive.
func (f *Filler) NilChance(p float64) *Filler {
	if p < 0 || p > 1 {
		panic("Filler.NilChance: p must be between 0 and 1, inclusive")
	}
	f.nilChance = p
	return f
}

// NumElements sets the minimum and maximum number of elements that will be
// added to a non-nil map or slice.
func (f *Filler) NumElements(min, max int) *Filler {
	if min < 0 {
		panic("Filler.NumElements: min must be >= 0")
	}
	if min > max {
		panic("Filler.NumElements: min must be <= max")
	}
	f.minElements = min
	f.maxElements = max
	return f
}

func (f *Filler) genElementCount() int {
	if f.minElements == f.maxElements {
		return f.minElements
	}
	return f.minElements + f.r.Intn(f.maxElements-f.minElements+1)
}

func (f *Filler) genShouldFill() bool {
	return f.r.Float64() >= f.nilChance
}

// MaxDepth sets the maximum number of recursive fill calls that will be made
// before stopping.  This includes struct members, pointers, and map and slice
// elements.
func (f *Filler) MaxDepth(d int) *Filler {
	f.maxDepth = d
	return f
}

// AllowUnexportedFields defines whether to fill unexported fields.
func (f *Filler) AllowUnexportedFields(flag bool) *Filler {
	f.allowUnexportedFields = flag
	return f
}

// SkipFieldsWithPattern tells this Filler to skip any field whose name matches
// the supplied pattern. Call this multiple times if needed. This is useful to
// skip XXX_ fields generated by protobuf.
func (f *Filler) SkipFieldsWithPattern(pattern *regexp.Regexp) *Filler {
	f.skipFieldPatterns = append(f.skipFieldPatterns, pattern)
	return f
}

// SimpleSelfFiller represents an object that knows how to randfill itself.
//
// Unlike NativeSelfFiller, this interface does not cause the type in question
// to depend on the randfill package.  This is most useful for simple types.  For
// more complex types, consider using NativeSelfFiller.
type SimpleSelfFiller interface {
	// RandFill fills the current object with random data.
	RandFill(r *rand.Rand)
}

// NativeSelfFiller represents an object that knows how to randfill itself.
//
// Unlike SimpleSelfFiller, this interface allows for recursive filling of
// child objects with the same rules as the parent Filler.
type NativeSelfFiller interface {
	// RandFill fills the current object with random data.
	RandFill(c Continue)
}

// Fill recursively fills all of obj's fields with something random.  First
// this tries to find a custom fill function (see Funcs).  If there is no
// custom function, this tests whether the object implements SimpleSelfFiller
// or NativeSelfFiller and if so, calls RandFill on it to fill itself.  If that
// fails, this will see if there is a default fill function provided by this
// package. If all of that fails, this will generate random values for all
// primitive fields and then recurse for all non-primitives.
//
// This is safe for cyclic or tree-like structs, up to a limit.  Use the
// MaxDepth method to adjust how deep you need it to recurse.
//
// obj must be a pointer. Exported (public) fields can always be set, and if
// the AllowUnexportedFields() modifier was called it can try to set unexported
// (private) fields, too.
//
// This is intended for tests, so will panic on bad input or unimplemented
// types.  This method takes a lock for the whole Filler, so it is not
// reentrant.  See Continue.
func (f *Filler) Fill(obj interface{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		panic("Filler.Fill: obj must be a pointer")
	}
	v = v.Elem()
	f.fillWithContext(v, 0)
}

// FillNoCustom is just like Fill, except that any custom fill function for
// obj's type will not be called and obj will not be tested for
// SimpleSelfFiller or NativeSelfFiller. This applies only to obj and not other
// instances of obj's type or to obj's child fields.
//
// obj must be a pointer. Exported (public) fields can always be set, and if
// the AllowUnexportedFields() modifier was called it can try to set unexported
// (private) fields, too.
//
// This is intended for tests, so will panic on bad input or unimplemented
// types.  This method takes a lock for the whole Filler, so it is not
// reentrant.  See Continue.
func (f *Filler) FillNoCustom(obj interface{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		panic("Filler.FillNoCustom: obj must be a pointer")
	}
	v = v.Elem()
	f.fillWithContext(v, flagNoCustomFill)
}

const (
	// Do not try to find a custom fill function.  Does not apply recursively.
	flagNoCustomFill uint64 = 1 << iota
)

func (f *Filler) fillWithContext(v reflect.Value, flags uint64) {
	fc := &fillerContext{filler: f}
	fc.doFill(v, flags)
}

// fillerContext carries context about a single filling run, which lets Filler
// be thread-safe.
type fillerContext struct {
	filler   *Filler
	curDepth int
}

func (fc *fillerContext) doFill(v reflect.Value, flags uint64) {
	if fc.curDepth >= fc.filler.maxDepth {
		return
	}
	fc.curDepth++
	defer func() { fc.curDepth-- }()

	if !v.CanSet() {
		if !fc.filler.allowUnexportedFields || !v.CanAddr() {
			return
		}
		v = reflect.NewAt(v.Type(), unsafe.Pointer(v.UnsafeAddr())).Elem()
	}

	if flags&flagNoCustomFill == 0 {
		// Check for both pointer and non-pointer custom functions.
		if v.CanAddr() && fc.tryCustom(v.Addr()) {
			return
		}
		if fc.tryCustom(v) {
			return
		}
	}

	if fn, ok := fillFuncMap[v.Kind()]; ok {
		fn(v, fc.filler.r)
		return
	}

	switch v.Kind() {
	case reflect.Map:
		if fc.filler.genShouldFill() {
			v.Set(reflect.MakeMap(v.Type()))
			n := fc.filler.genElementCount()
			for i := 0; i < n; i++ {
				key := reflect.New(v.Type().Key()).Elem()
				fc.doFill(key, 0)
				val := reflect.New(v.Type().Elem()).Elem()
				fc.doFill(val, 0)
				v.SetMapIndex(key, val)
			}
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Ptr:
		if fc.filler.genShouldFill() {
			v.Set(reflect.New(v.Type().Elem()))
			fc.doFill(v.Elem(), 0)
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Slice:
		if fc.filler.genShouldFill() {
			n := fc.filler.genElementCount()
			v.Set(reflect.MakeSlice(v.Type(), n, n))
			for i := 0; i < n; i++ {
				fc.doFill(v.Index(i), 0)
			}
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Array:
		if fc.filler.genShouldFill() {
			n := v.Len()
			for i := 0; i < n; i++ {
				fc.doFill(v.Index(i), 0)
			}
			return
		}
		v.Set(reflect.Zero(v.Type()))
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			skipField := false
			fieldName := v.Type().Field(i).Name
			for _, pattern := range fc.filler.skipFieldPatterns {
				if pattern.MatchString(fieldName) {
					skipField = true
					break
				}
			}
			if !skipField {
				fc.doFill(v.Field(i), 0)
			}
		}
	case reflect.Chan:
		fallthrough
	case reflect.Func:
		fallthrough
	case reflect.Interface:
		fallthrough
	default:
		panic(fmt.Sprintf("can't fill type %v, kind %v", v.Type(), v.Kind()))
	}
}

// tryCustom searches for custom handlers, and returns true iff it finds a match
// and successfully randomizes v.
func (fc *fillerContext) tryCustom(v reflect.Value) bool {
	// First: see if we have a fill function for it.
	doCustom, ok := fc.filler.customFuncs[v.Type()]
	if !ok {
		// Second: see if it can fill itself.
		if v.CanInterface() {
			intf := v.Interface()
			if fillable, ok := intf.(SimpleSelfFiller); ok {
				fillable.RandFill(fc.filler.r)
				return true
			}
			if fillable, ok := intf.(NativeSelfFiller); ok {
				fillable.RandFill(Continue{fc: fc, Rand: fc.filler.r})
				return true
			}
		}
		// Finally: see if there is a default fill function.
		doCustom, ok = fc.filler.defaultFuncs[v.Type()]
		if !ok {
			return false
		}
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

	doCustom.Call([]reflect.Value{
		v,
		reflect.ValueOf(Continue{
			fc:   fc,
			Rand: fc.filler.r,
		}),
	})
	return true
}

// Continue can be passed to custom fill functions to allow them to use
// the correct source of randomness and to continue filling their members.
type Continue struct {
	fc *fillerContext

	// For convenience, Continue implements rand.Rand via embedding.
	// Use this for generating any randomness if you want your filling
	// to be repeatable for a given seed.
	*rand.Rand
}

// Fill continues filling obj. obj must be a pointer or a reflect.Value of a
// pointer.  See Filler.Fill.
func (c Continue) Fill(obj interface{}) {
	v, ok := obj.(reflect.Value)
	if !ok {
		v = reflect.ValueOf(obj)
	}
	if v.Kind() != reflect.Ptr {
		panic("Continue.Fill: obj must be a pointer")
	}
	v = v.Elem()
	c.fc.doFill(v, 0)
}

// FillNoCustom continues filling obj, except that any custom fill function for
// obj's type will not be called and obj will not be tested for
// SimpleSelfFiller or NativeSelfFiller.  See Filler.FillNoCustom.
func (c Continue) FillNoCustom(obj interface{}) {
	v, ok := obj.(reflect.Value)
	if !ok {
		v = reflect.ValueOf(obj)
	}
	if v.Kind() != reflect.Ptr {
		panic("Continue.FillNoCustom: obj must be a pointer")
	}
	v = v.Elem()
	c.fc.doFill(v, flagNoCustomFill)
}

const defaultStringMaxLen = 20

// String makes a random string up to n characters long. If n is 0, the default
// size range is [0-20). The returned string may include a variety of (valid)
// UTF-8 encodings.
func (c Continue) String(n int) string {
	return randString(c.Rand, n)
}

// Uint64 makes random 64 bit numbers.
// Weirdly, rand doesn't have a function that gives you 64 random bits.
func (c Continue) Uint64() uint64 {
	return randUint64(c.Rand)
}

// Bool returns true or false randomly.
func (c Continue) Bool() bool {
	return randBool(c.Rand)
}

func fillInt(v reflect.Value, r *rand.Rand) {
	v.SetInt(int64(randUint64(r)))
}

func fillUint(v reflect.Value, r *rand.Rand) {
	v.SetUint(randUint64(r))
}

func randfillTime(t *time.Time, c Continue) {
	var sec, nsec int64
	// Allow for about 1000 years of random time values, which keeps things
	// like JSON parsing reasonably happy.
	sec = c.Rand.Int63n(1000 * 365 * 24 * 60 * 60)
	// Nanosecond values greater than 1Bn are technically allowed but result in
	// time.Time values with invalid timezone offsets.
	nsec = c.Rand.Int63n(999999999)
	*t = time.Unix(sec, nsec)
}

var fillFuncMap = map[reflect.Kind]func(reflect.Value, *rand.Rand){
	reflect.Bool: func(v reflect.Value, r *rand.Rand) {
		v.SetBool(randBool(r))
	},
	reflect.Int:     fillInt,
	reflect.Int8:    fillInt,
	reflect.Int16:   fillInt,
	reflect.Int32:   fillInt,
	reflect.Int64:   fillInt,
	reflect.Uint:    fillUint,
	reflect.Uint8:   fillUint,
	reflect.Uint16:  fillUint,
	reflect.Uint32:  fillUint,
	reflect.Uint64:  fillUint,
	reflect.Uintptr: fillUint,
	reflect.Float32: func(v reflect.Value, r *rand.Rand) {
		v.SetFloat(float64(r.Float32()))
	},
	reflect.Float64: func(v reflect.Value, r *rand.Rand) {
		v.SetFloat(r.Float64())
	},
	reflect.Complex64: func(v reflect.Value, r *rand.Rand) {
		v.SetComplex(complex128(complex(r.Float32(), r.Float32())))
	},
	reflect.Complex128: func(v reflect.Value, r *rand.Rand) {
		v.SetComplex(complex(r.Float64(), r.Float64()))
	},
	reflect.String: func(v reflect.Value, r *rand.Rand) {
		v.SetString(randString(r, 0))
	},
	reflect.UnsafePointer: func(v reflect.Value, r *rand.Rand) {
		panic("filling of UnsafePointers is not implemented")
	},
}

// randBool returns true or false randomly.
func randBool(r *rand.Rand) bool {
	return r.Int31()&(1<<30) == 0
}

type int63nPicker interface {
	Int63n(int64) int64
}

// UnicodeRange describes a sequential range of unicode characters.
// Last must be numerically greater than First.
type UnicodeRange struct {
	First, Last rune
}

// UnicodeRanges describes an arbitrary number of sequential ranges of unicode characters.
// To be useful, each range must have at least one character (First <= Last) and
// there must be at least one range.
type UnicodeRanges []UnicodeRange

// choose returns a random unicode character from the given range, using the
// given randomness source.
func (ur UnicodeRange) choose(r int63nPicker) rune {
	count := int64(ur.Last - ur.First + 1)
	return ur.First + rune(r.Int63n(count))
}

// CustomStringFillFunc constructs a FillFunc which produces random strings.
// Each character is selected from the range ur. If there are no characters
// in the range (cr.Last < cr.First), this will panic.
func (ur UnicodeRange) CustomStringFillFunc(n int) func(s *string, c Continue) {
	ur.check()
	return func(s *string, c Continue) {
		*s = ur.randString(c.Rand, n)
	}
}

// check is a function that used to check whether the first of ur(UnicodeRange)
// is greater than the last one.
func (ur UnicodeRange) check() {
	if ur.Last < ur.First {
		panic("UnicodeRange.check: the last encoding must be greater than the first")
	}
}

// randString of UnicodeRange makes a random string up to 20 characters long.
// Each character is selected form ur(UnicodeRange).
func (ur UnicodeRange) randString(r *rand.Rand, max int) string {
	if max == 0 {
		max = defaultStringMaxLen
	}
	n := r.Intn(max)
	sb := strings.Builder{}
	sb.Grow(n)
	for i := 0; i < n; i++ {
		sb.WriteRune(ur.choose(r))
	}
	return sb.String()
}

// defaultUnicodeRanges sets a default unicode range when users do not set
// CustomStringFillFunc() but want to fill strings.
var defaultUnicodeRanges = UnicodeRanges{
	{' ', '~'},           // ASCII characters
	{'\u00a0', '\u02af'}, // Multi-byte encoded characters
	{'\u4e00', '\u9fff'}, // Common CJK (even longer encodings)
}

// CustomStringFillFunc constructs a FillFunc which produces random strings.
// Each character is selected from one of the ranges of ur(UnicodeRanges).
// Each range has an equal probability of being chosen. If there are no ranges,
// or a selected range has no characters (.Last < .First), this will panic.
// Do not modify any of the ranges in ur after calling this function.
func (ur UnicodeRanges) CustomStringFillFunc(n int) func(s *string, c Continue) {
	// Check unicode ranges slice is empty.
	if len(ur) == 0 {
		panic("UnicodeRanges is empty")
	}
	// if not empty, each range should be checked.
	for i := range ur {
		ur[i].check()
	}
	return func(s *string, c Continue) {
		*s = ur.randString(c.Rand, n)
	}
}

// randString of UnicodeRanges makes a random string up to 20 characters long.
// Each character is selected form one of the ranges of ur(UnicodeRanges),
// and each range has an equal probability of being chosen.
func (ur UnicodeRanges) randString(r *rand.Rand, max int) string {
	if max == 0 {
		max = defaultStringMaxLen
	}
	n := r.Intn(max)
	sb := strings.Builder{}
	sb.Grow(n)
	for i := 0; i < n; i++ {
		sb.WriteRune(ur[r.Intn(len(ur))].choose(r))
	}
	return sb.String()
}

// randString makes a random string up to 20 characters long. The returned string
// may include a variety of (valid) UTF-8 encodings.
func randString(r *rand.Rand, max int) string {
	return defaultUnicodeRanges.randString(r, max)
}

// randUint64 makes random 64 bit numbers.
// Weirdly, rand doesn't have a function that gives you 64 random bits.
func randUint64(r *rand.Rand) uint64 {
	return uint64(r.Uint32())<<32 | uint64(r.Uint32())
}
