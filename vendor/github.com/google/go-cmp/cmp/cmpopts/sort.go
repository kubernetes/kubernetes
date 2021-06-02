// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmpopts

import (
	"fmt"
	"reflect"
	"sort"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/internal/function"
)

// SortSlices returns a Transformer option that sorts all []V.
// The less function must be of the form "func(T, T) bool" which is used to
// sort any slice with element type V that is assignable to T.
//
// The less function must be:
//	• Deterministic: less(x, y) == less(x, y)
//	• Irreflexive: !less(x, x)
//	• Transitive: if !less(x, y) and !less(y, z), then !less(x, z)
//
// The less function does not have to be "total". That is, if !less(x, y) and
// !less(y, x) for two elements x and y, their relative order is maintained.
//
// SortSlices can be used in conjunction with EquateEmpty.
func SortSlices(lessFunc interface{}) cmp.Option {
	vf := reflect.ValueOf(lessFunc)
	if !function.IsType(vf.Type(), function.Less) || vf.IsNil() {
		panic(fmt.Sprintf("invalid less function: %T", lessFunc))
	}
	ss := sliceSorter{vf.Type().In(0), vf}
	return cmp.FilterValues(ss.filter, cmp.Transformer("cmpopts.SortSlices", ss.sort))
}

type sliceSorter struct {
	in  reflect.Type  // T
	fnc reflect.Value // func(T, T) bool
}

func (ss sliceSorter) filter(x, y interface{}) bool {
	vx, vy := reflect.ValueOf(x), reflect.ValueOf(y)
	if !(x != nil && y != nil && vx.Type() == vy.Type()) ||
		!(vx.Kind() == reflect.Slice && vx.Type().Elem().AssignableTo(ss.in)) ||
		(vx.Len() <= 1 && vy.Len() <= 1) {
		return false
	}
	// Check whether the slices are already sorted to avoid an infinite
	// recursion cycle applying the same transform to itself.
	ok1 := sort.SliceIsSorted(x, func(i, j int) bool { return ss.less(vx, i, j) })
	ok2 := sort.SliceIsSorted(y, func(i, j int) bool { return ss.less(vy, i, j) })
	return !ok1 || !ok2
}
func (ss sliceSorter) sort(x interface{}) interface{} {
	src := reflect.ValueOf(x)
	dst := reflect.MakeSlice(src.Type(), src.Len(), src.Len())
	for i := 0; i < src.Len(); i++ {
		dst.Index(i).Set(src.Index(i))
	}
	sort.SliceStable(dst.Interface(), func(i, j int) bool { return ss.less(dst, i, j) })
	ss.checkSort(dst)
	return dst.Interface()
}
func (ss sliceSorter) checkSort(v reflect.Value) {
	start := -1 // Start of a sequence of equal elements.
	for i := 1; i < v.Len(); i++ {
		if ss.less(v, i-1, i) {
			// Check that first and last elements in v[start:i] are equal.
			if start >= 0 && (ss.less(v, start, i-1) || ss.less(v, i-1, start)) {
				panic(fmt.Sprintf("incomparable values detected: want equal elements: %v", v.Slice(start, i)))
			}
			start = -1
		} else if start == -1 {
			start = i
		}
	}
}
func (ss sliceSorter) less(v reflect.Value, i, j int) bool {
	vx, vy := v.Index(i), v.Index(j)
	return ss.fnc.Call([]reflect.Value{vx, vy})[0].Bool()
}

// SortMaps returns a Transformer option that flattens map[K]V types to be a
// sorted []struct{K, V}. The less function must be of the form
// "func(T, T) bool" which is used to sort any map with key K that is
// assignable to T.
//
// Flattening the map into a slice has the property that cmp.Equal is able to
// use Comparers on K or the K.Equal method if it exists.
//
// The less function must be:
//	• Deterministic: less(x, y) == less(x, y)
//	• Irreflexive: !less(x, x)
//	• Transitive: if !less(x, y) and !less(y, z), then !less(x, z)
//	• Total: if x != y, then either less(x, y) or less(y, x)
//
// SortMaps can be used in conjunction with EquateEmpty.
func SortMaps(lessFunc interface{}) cmp.Option {
	vf := reflect.ValueOf(lessFunc)
	if !function.IsType(vf.Type(), function.Less) || vf.IsNil() {
		panic(fmt.Sprintf("invalid less function: %T", lessFunc))
	}
	ms := mapSorter{vf.Type().In(0), vf}
	return cmp.FilterValues(ms.filter, cmp.Transformer("cmpopts.SortMaps", ms.sort))
}

type mapSorter struct {
	in  reflect.Type  // T
	fnc reflect.Value // func(T, T) bool
}

func (ms mapSorter) filter(x, y interface{}) bool {
	vx, vy := reflect.ValueOf(x), reflect.ValueOf(y)
	return (x != nil && y != nil && vx.Type() == vy.Type()) &&
		(vx.Kind() == reflect.Map && vx.Type().Key().AssignableTo(ms.in)) &&
		(vx.Len() != 0 || vy.Len() != 0)
}
func (ms mapSorter) sort(x interface{}) interface{} {
	src := reflect.ValueOf(x)
	outType := reflect.StructOf([]reflect.StructField{
		{Name: "K", Type: src.Type().Key()},
		{Name: "V", Type: src.Type().Elem()},
	})
	dst := reflect.MakeSlice(reflect.SliceOf(outType), src.Len(), src.Len())
	for i, k := range src.MapKeys() {
		v := reflect.New(outType).Elem()
		v.Field(0).Set(k)
		v.Field(1).Set(src.MapIndex(k))
		dst.Index(i).Set(v)
	}
	sort.Slice(dst.Interface(), func(i, j int) bool { return ms.less(dst, i, j) })
	ms.checkSort(dst)
	return dst.Interface()
}
func (ms mapSorter) checkSort(v reflect.Value) {
	for i := 1; i < v.Len(); i++ {
		if !ms.less(v, i-1, i) {
			panic(fmt.Sprintf("partial order detected: want %v < %v", v.Index(i-1), v.Index(i)))
		}
	}
}
func (ms mapSorter) less(v reflect.Value, i, j int) bool {
	vx, vy := v.Index(i).Field(0), v.Index(j).Field(0)
	return ms.fnc.Call([]reflect.Value{vx, vy})[0].Bool()
}
