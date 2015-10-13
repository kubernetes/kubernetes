// //+build ignore

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// ************************************************************
// DO NOT EDIT.
// THIS FILE IS AUTO-GENERATED from fast-path.go.tmpl
// ************************************************************

package codec

// Fast path functions try to create a fast path encode or decode implementation
// for common maps and slices.
//
// We define the functions and register then in this single file
// so as not to pollute the encode.go and decode.go, and create a dependency in there.
// This file can be omitted without causing a build failure.
//
// The advantage of fast paths is:
//    - Many calls bypass reflection altogether
//
// Currently support
//    - slice of all builtin types,
//    - map of all builtin types to string or interface value
//    - symetrical maps of all builtin types (e.g. str-str, uint8-uint8)
// This should provide adequate "typical" implementations.
//
// Note that fast track decode functions must handle values for which an address cannot be obtained.
// For example:
//   m2 := map[string]int{}
//   p2 := []interface{}{m2}
//   // decoding into p2 will bomb if fast track functions do not treat like unaddressable.
//

import (
	"reflect"
	"sort"
)

const fastpathCheckNilFalse = false // for reflect
const fastpathCheckNilTrue = true   // for type switch

type fastpathT struct{}

var fastpathTV fastpathT

type fastpathE struct {
	rtid  uintptr
	rt    reflect.Type
	encfn func(*encFnInfo, reflect.Value)
	decfn func(*decFnInfo, reflect.Value)
}

type fastpathA [239]fastpathE

func (x *fastpathA) index(rtid uintptr) int {
	// use binary search to grab the index (adapted from sort/search.go)
	h, i, j := 0, 0, 239 // len(x)
	for i < j {
		h = i + (j-i)/2
		if x[h].rtid < rtid {
			i = h + 1
		} else {
			j = h
		}
	}
	if i < 239 && x[i].rtid == rtid {
		return i
	}
	return -1
}

type fastpathAslice []fastpathE

func (x fastpathAslice) Len() int           { return len(x) }
func (x fastpathAslice) Less(i, j int) bool { return x[i].rtid < x[j].rtid }
func (x fastpathAslice) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }

var fastpathAV fastpathA

// due to possible initialization loop error, make fastpath in an init()
func init() {
	if !fastpathEnabled {
		return
	}
	i := 0
	fn := func(v interface{}, fe func(*encFnInfo, reflect.Value), fd func(*decFnInfo, reflect.Value)) (f fastpathE) {
		xrt := reflect.TypeOf(v)
		xptr := reflect.ValueOf(xrt).Pointer()
		fastpathAV[i] = fastpathE{xptr, xrt, fe, fd}
		i++
		return
	}

	fn([]interface{}(nil), (*encFnInfo).fastpathEncSliceIntfR, (*decFnInfo).fastpathDecSliceIntfR)
	fn([]string(nil), (*encFnInfo).fastpathEncSliceStringR, (*decFnInfo).fastpathDecSliceStringR)
	fn([]float32(nil), (*encFnInfo).fastpathEncSliceFloat32R, (*decFnInfo).fastpathDecSliceFloat32R)
	fn([]float64(nil), (*encFnInfo).fastpathEncSliceFloat64R, (*decFnInfo).fastpathDecSliceFloat64R)
	fn([]uint(nil), (*encFnInfo).fastpathEncSliceUintR, (*decFnInfo).fastpathDecSliceUintR)
	fn([]uint16(nil), (*encFnInfo).fastpathEncSliceUint16R, (*decFnInfo).fastpathDecSliceUint16R)
	fn([]uint32(nil), (*encFnInfo).fastpathEncSliceUint32R, (*decFnInfo).fastpathDecSliceUint32R)
	fn([]uint64(nil), (*encFnInfo).fastpathEncSliceUint64R, (*decFnInfo).fastpathDecSliceUint64R)
	fn([]int(nil), (*encFnInfo).fastpathEncSliceIntR, (*decFnInfo).fastpathDecSliceIntR)
	fn([]int8(nil), (*encFnInfo).fastpathEncSliceInt8R, (*decFnInfo).fastpathDecSliceInt8R)
	fn([]int16(nil), (*encFnInfo).fastpathEncSliceInt16R, (*decFnInfo).fastpathDecSliceInt16R)
	fn([]int32(nil), (*encFnInfo).fastpathEncSliceInt32R, (*decFnInfo).fastpathDecSliceInt32R)
	fn([]int64(nil), (*encFnInfo).fastpathEncSliceInt64R, (*decFnInfo).fastpathDecSliceInt64R)
	fn([]bool(nil), (*encFnInfo).fastpathEncSliceBoolR, (*decFnInfo).fastpathDecSliceBoolR)

	fn(map[interface{}]interface{}(nil), (*encFnInfo).fastpathEncMapIntfIntfR, (*decFnInfo).fastpathDecMapIntfIntfR)
	fn(map[interface{}]string(nil), (*encFnInfo).fastpathEncMapIntfStringR, (*decFnInfo).fastpathDecMapIntfStringR)
	fn(map[interface{}]uint(nil), (*encFnInfo).fastpathEncMapIntfUintR, (*decFnInfo).fastpathDecMapIntfUintR)
	fn(map[interface{}]uint8(nil), (*encFnInfo).fastpathEncMapIntfUint8R, (*decFnInfo).fastpathDecMapIntfUint8R)
	fn(map[interface{}]uint16(nil), (*encFnInfo).fastpathEncMapIntfUint16R, (*decFnInfo).fastpathDecMapIntfUint16R)
	fn(map[interface{}]uint32(nil), (*encFnInfo).fastpathEncMapIntfUint32R, (*decFnInfo).fastpathDecMapIntfUint32R)
	fn(map[interface{}]uint64(nil), (*encFnInfo).fastpathEncMapIntfUint64R, (*decFnInfo).fastpathDecMapIntfUint64R)
	fn(map[interface{}]int(nil), (*encFnInfo).fastpathEncMapIntfIntR, (*decFnInfo).fastpathDecMapIntfIntR)
	fn(map[interface{}]int8(nil), (*encFnInfo).fastpathEncMapIntfInt8R, (*decFnInfo).fastpathDecMapIntfInt8R)
	fn(map[interface{}]int16(nil), (*encFnInfo).fastpathEncMapIntfInt16R, (*decFnInfo).fastpathDecMapIntfInt16R)
	fn(map[interface{}]int32(nil), (*encFnInfo).fastpathEncMapIntfInt32R, (*decFnInfo).fastpathDecMapIntfInt32R)
	fn(map[interface{}]int64(nil), (*encFnInfo).fastpathEncMapIntfInt64R, (*decFnInfo).fastpathDecMapIntfInt64R)
	fn(map[interface{}]float32(nil), (*encFnInfo).fastpathEncMapIntfFloat32R, (*decFnInfo).fastpathDecMapIntfFloat32R)
	fn(map[interface{}]float64(nil), (*encFnInfo).fastpathEncMapIntfFloat64R, (*decFnInfo).fastpathDecMapIntfFloat64R)
	fn(map[interface{}]bool(nil), (*encFnInfo).fastpathEncMapIntfBoolR, (*decFnInfo).fastpathDecMapIntfBoolR)
	fn(map[string]interface{}(nil), (*encFnInfo).fastpathEncMapStringIntfR, (*decFnInfo).fastpathDecMapStringIntfR)
	fn(map[string]string(nil), (*encFnInfo).fastpathEncMapStringStringR, (*decFnInfo).fastpathDecMapStringStringR)
	fn(map[string]uint(nil), (*encFnInfo).fastpathEncMapStringUintR, (*decFnInfo).fastpathDecMapStringUintR)
	fn(map[string]uint8(nil), (*encFnInfo).fastpathEncMapStringUint8R, (*decFnInfo).fastpathDecMapStringUint8R)
	fn(map[string]uint16(nil), (*encFnInfo).fastpathEncMapStringUint16R, (*decFnInfo).fastpathDecMapStringUint16R)
	fn(map[string]uint32(nil), (*encFnInfo).fastpathEncMapStringUint32R, (*decFnInfo).fastpathDecMapStringUint32R)
	fn(map[string]uint64(nil), (*encFnInfo).fastpathEncMapStringUint64R, (*decFnInfo).fastpathDecMapStringUint64R)
	fn(map[string]int(nil), (*encFnInfo).fastpathEncMapStringIntR, (*decFnInfo).fastpathDecMapStringIntR)
	fn(map[string]int8(nil), (*encFnInfo).fastpathEncMapStringInt8R, (*decFnInfo).fastpathDecMapStringInt8R)
	fn(map[string]int16(nil), (*encFnInfo).fastpathEncMapStringInt16R, (*decFnInfo).fastpathDecMapStringInt16R)
	fn(map[string]int32(nil), (*encFnInfo).fastpathEncMapStringInt32R, (*decFnInfo).fastpathDecMapStringInt32R)
	fn(map[string]int64(nil), (*encFnInfo).fastpathEncMapStringInt64R, (*decFnInfo).fastpathDecMapStringInt64R)
	fn(map[string]float32(nil), (*encFnInfo).fastpathEncMapStringFloat32R, (*decFnInfo).fastpathDecMapStringFloat32R)
	fn(map[string]float64(nil), (*encFnInfo).fastpathEncMapStringFloat64R, (*decFnInfo).fastpathDecMapStringFloat64R)
	fn(map[string]bool(nil), (*encFnInfo).fastpathEncMapStringBoolR, (*decFnInfo).fastpathDecMapStringBoolR)
	fn(map[float32]interface{}(nil), (*encFnInfo).fastpathEncMapFloat32IntfR, (*decFnInfo).fastpathDecMapFloat32IntfR)
	fn(map[float32]string(nil), (*encFnInfo).fastpathEncMapFloat32StringR, (*decFnInfo).fastpathDecMapFloat32StringR)
	fn(map[float32]uint(nil), (*encFnInfo).fastpathEncMapFloat32UintR, (*decFnInfo).fastpathDecMapFloat32UintR)
	fn(map[float32]uint8(nil), (*encFnInfo).fastpathEncMapFloat32Uint8R, (*decFnInfo).fastpathDecMapFloat32Uint8R)
	fn(map[float32]uint16(nil), (*encFnInfo).fastpathEncMapFloat32Uint16R, (*decFnInfo).fastpathDecMapFloat32Uint16R)
	fn(map[float32]uint32(nil), (*encFnInfo).fastpathEncMapFloat32Uint32R, (*decFnInfo).fastpathDecMapFloat32Uint32R)
	fn(map[float32]uint64(nil), (*encFnInfo).fastpathEncMapFloat32Uint64R, (*decFnInfo).fastpathDecMapFloat32Uint64R)
	fn(map[float32]int(nil), (*encFnInfo).fastpathEncMapFloat32IntR, (*decFnInfo).fastpathDecMapFloat32IntR)
	fn(map[float32]int8(nil), (*encFnInfo).fastpathEncMapFloat32Int8R, (*decFnInfo).fastpathDecMapFloat32Int8R)
	fn(map[float32]int16(nil), (*encFnInfo).fastpathEncMapFloat32Int16R, (*decFnInfo).fastpathDecMapFloat32Int16R)
	fn(map[float32]int32(nil), (*encFnInfo).fastpathEncMapFloat32Int32R, (*decFnInfo).fastpathDecMapFloat32Int32R)
	fn(map[float32]int64(nil), (*encFnInfo).fastpathEncMapFloat32Int64R, (*decFnInfo).fastpathDecMapFloat32Int64R)
	fn(map[float32]float32(nil), (*encFnInfo).fastpathEncMapFloat32Float32R, (*decFnInfo).fastpathDecMapFloat32Float32R)
	fn(map[float32]float64(nil), (*encFnInfo).fastpathEncMapFloat32Float64R, (*decFnInfo).fastpathDecMapFloat32Float64R)
	fn(map[float32]bool(nil), (*encFnInfo).fastpathEncMapFloat32BoolR, (*decFnInfo).fastpathDecMapFloat32BoolR)
	fn(map[float64]interface{}(nil), (*encFnInfo).fastpathEncMapFloat64IntfR, (*decFnInfo).fastpathDecMapFloat64IntfR)
	fn(map[float64]string(nil), (*encFnInfo).fastpathEncMapFloat64StringR, (*decFnInfo).fastpathDecMapFloat64StringR)
	fn(map[float64]uint(nil), (*encFnInfo).fastpathEncMapFloat64UintR, (*decFnInfo).fastpathDecMapFloat64UintR)
	fn(map[float64]uint8(nil), (*encFnInfo).fastpathEncMapFloat64Uint8R, (*decFnInfo).fastpathDecMapFloat64Uint8R)
	fn(map[float64]uint16(nil), (*encFnInfo).fastpathEncMapFloat64Uint16R, (*decFnInfo).fastpathDecMapFloat64Uint16R)
	fn(map[float64]uint32(nil), (*encFnInfo).fastpathEncMapFloat64Uint32R, (*decFnInfo).fastpathDecMapFloat64Uint32R)
	fn(map[float64]uint64(nil), (*encFnInfo).fastpathEncMapFloat64Uint64R, (*decFnInfo).fastpathDecMapFloat64Uint64R)
	fn(map[float64]int(nil), (*encFnInfo).fastpathEncMapFloat64IntR, (*decFnInfo).fastpathDecMapFloat64IntR)
	fn(map[float64]int8(nil), (*encFnInfo).fastpathEncMapFloat64Int8R, (*decFnInfo).fastpathDecMapFloat64Int8R)
	fn(map[float64]int16(nil), (*encFnInfo).fastpathEncMapFloat64Int16R, (*decFnInfo).fastpathDecMapFloat64Int16R)
	fn(map[float64]int32(nil), (*encFnInfo).fastpathEncMapFloat64Int32R, (*decFnInfo).fastpathDecMapFloat64Int32R)
	fn(map[float64]int64(nil), (*encFnInfo).fastpathEncMapFloat64Int64R, (*decFnInfo).fastpathDecMapFloat64Int64R)
	fn(map[float64]float32(nil), (*encFnInfo).fastpathEncMapFloat64Float32R, (*decFnInfo).fastpathDecMapFloat64Float32R)
	fn(map[float64]float64(nil), (*encFnInfo).fastpathEncMapFloat64Float64R, (*decFnInfo).fastpathDecMapFloat64Float64R)
	fn(map[float64]bool(nil), (*encFnInfo).fastpathEncMapFloat64BoolR, (*decFnInfo).fastpathDecMapFloat64BoolR)
	fn(map[uint]interface{}(nil), (*encFnInfo).fastpathEncMapUintIntfR, (*decFnInfo).fastpathDecMapUintIntfR)
	fn(map[uint]string(nil), (*encFnInfo).fastpathEncMapUintStringR, (*decFnInfo).fastpathDecMapUintStringR)
	fn(map[uint]uint(nil), (*encFnInfo).fastpathEncMapUintUintR, (*decFnInfo).fastpathDecMapUintUintR)
	fn(map[uint]uint8(nil), (*encFnInfo).fastpathEncMapUintUint8R, (*decFnInfo).fastpathDecMapUintUint8R)
	fn(map[uint]uint16(nil), (*encFnInfo).fastpathEncMapUintUint16R, (*decFnInfo).fastpathDecMapUintUint16R)
	fn(map[uint]uint32(nil), (*encFnInfo).fastpathEncMapUintUint32R, (*decFnInfo).fastpathDecMapUintUint32R)
	fn(map[uint]uint64(nil), (*encFnInfo).fastpathEncMapUintUint64R, (*decFnInfo).fastpathDecMapUintUint64R)
	fn(map[uint]int(nil), (*encFnInfo).fastpathEncMapUintIntR, (*decFnInfo).fastpathDecMapUintIntR)
	fn(map[uint]int8(nil), (*encFnInfo).fastpathEncMapUintInt8R, (*decFnInfo).fastpathDecMapUintInt8R)
	fn(map[uint]int16(nil), (*encFnInfo).fastpathEncMapUintInt16R, (*decFnInfo).fastpathDecMapUintInt16R)
	fn(map[uint]int32(nil), (*encFnInfo).fastpathEncMapUintInt32R, (*decFnInfo).fastpathDecMapUintInt32R)
	fn(map[uint]int64(nil), (*encFnInfo).fastpathEncMapUintInt64R, (*decFnInfo).fastpathDecMapUintInt64R)
	fn(map[uint]float32(nil), (*encFnInfo).fastpathEncMapUintFloat32R, (*decFnInfo).fastpathDecMapUintFloat32R)
	fn(map[uint]float64(nil), (*encFnInfo).fastpathEncMapUintFloat64R, (*decFnInfo).fastpathDecMapUintFloat64R)
	fn(map[uint]bool(nil), (*encFnInfo).fastpathEncMapUintBoolR, (*decFnInfo).fastpathDecMapUintBoolR)
	fn(map[uint8]interface{}(nil), (*encFnInfo).fastpathEncMapUint8IntfR, (*decFnInfo).fastpathDecMapUint8IntfR)
	fn(map[uint8]string(nil), (*encFnInfo).fastpathEncMapUint8StringR, (*decFnInfo).fastpathDecMapUint8StringR)
	fn(map[uint8]uint(nil), (*encFnInfo).fastpathEncMapUint8UintR, (*decFnInfo).fastpathDecMapUint8UintR)
	fn(map[uint8]uint8(nil), (*encFnInfo).fastpathEncMapUint8Uint8R, (*decFnInfo).fastpathDecMapUint8Uint8R)
	fn(map[uint8]uint16(nil), (*encFnInfo).fastpathEncMapUint8Uint16R, (*decFnInfo).fastpathDecMapUint8Uint16R)
	fn(map[uint8]uint32(nil), (*encFnInfo).fastpathEncMapUint8Uint32R, (*decFnInfo).fastpathDecMapUint8Uint32R)
	fn(map[uint8]uint64(nil), (*encFnInfo).fastpathEncMapUint8Uint64R, (*decFnInfo).fastpathDecMapUint8Uint64R)
	fn(map[uint8]int(nil), (*encFnInfo).fastpathEncMapUint8IntR, (*decFnInfo).fastpathDecMapUint8IntR)
	fn(map[uint8]int8(nil), (*encFnInfo).fastpathEncMapUint8Int8R, (*decFnInfo).fastpathDecMapUint8Int8R)
	fn(map[uint8]int16(nil), (*encFnInfo).fastpathEncMapUint8Int16R, (*decFnInfo).fastpathDecMapUint8Int16R)
	fn(map[uint8]int32(nil), (*encFnInfo).fastpathEncMapUint8Int32R, (*decFnInfo).fastpathDecMapUint8Int32R)
	fn(map[uint8]int64(nil), (*encFnInfo).fastpathEncMapUint8Int64R, (*decFnInfo).fastpathDecMapUint8Int64R)
	fn(map[uint8]float32(nil), (*encFnInfo).fastpathEncMapUint8Float32R, (*decFnInfo).fastpathDecMapUint8Float32R)
	fn(map[uint8]float64(nil), (*encFnInfo).fastpathEncMapUint8Float64R, (*decFnInfo).fastpathDecMapUint8Float64R)
	fn(map[uint8]bool(nil), (*encFnInfo).fastpathEncMapUint8BoolR, (*decFnInfo).fastpathDecMapUint8BoolR)
	fn(map[uint16]interface{}(nil), (*encFnInfo).fastpathEncMapUint16IntfR, (*decFnInfo).fastpathDecMapUint16IntfR)
	fn(map[uint16]string(nil), (*encFnInfo).fastpathEncMapUint16StringR, (*decFnInfo).fastpathDecMapUint16StringR)
	fn(map[uint16]uint(nil), (*encFnInfo).fastpathEncMapUint16UintR, (*decFnInfo).fastpathDecMapUint16UintR)
	fn(map[uint16]uint8(nil), (*encFnInfo).fastpathEncMapUint16Uint8R, (*decFnInfo).fastpathDecMapUint16Uint8R)
	fn(map[uint16]uint16(nil), (*encFnInfo).fastpathEncMapUint16Uint16R, (*decFnInfo).fastpathDecMapUint16Uint16R)
	fn(map[uint16]uint32(nil), (*encFnInfo).fastpathEncMapUint16Uint32R, (*decFnInfo).fastpathDecMapUint16Uint32R)
	fn(map[uint16]uint64(nil), (*encFnInfo).fastpathEncMapUint16Uint64R, (*decFnInfo).fastpathDecMapUint16Uint64R)
	fn(map[uint16]int(nil), (*encFnInfo).fastpathEncMapUint16IntR, (*decFnInfo).fastpathDecMapUint16IntR)
	fn(map[uint16]int8(nil), (*encFnInfo).fastpathEncMapUint16Int8R, (*decFnInfo).fastpathDecMapUint16Int8R)
	fn(map[uint16]int16(nil), (*encFnInfo).fastpathEncMapUint16Int16R, (*decFnInfo).fastpathDecMapUint16Int16R)
	fn(map[uint16]int32(nil), (*encFnInfo).fastpathEncMapUint16Int32R, (*decFnInfo).fastpathDecMapUint16Int32R)
	fn(map[uint16]int64(nil), (*encFnInfo).fastpathEncMapUint16Int64R, (*decFnInfo).fastpathDecMapUint16Int64R)
	fn(map[uint16]float32(nil), (*encFnInfo).fastpathEncMapUint16Float32R, (*decFnInfo).fastpathDecMapUint16Float32R)
	fn(map[uint16]float64(nil), (*encFnInfo).fastpathEncMapUint16Float64R, (*decFnInfo).fastpathDecMapUint16Float64R)
	fn(map[uint16]bool(nil), (*encFnInfo).fastpathEncMapUint16BoolR, (*decFnInfo).fastpathDecMapUint16BoolR)
	fn(map[uint32]interface{}(nil), (*encFnInfo).fastpathEncMapUint32IntfR, (*decFnInfo).fastpathDecMapUint32IntfR)
	fn(map[uint32]string(nil), (*encFnInfo).fastpathEncMapUint32StringR, (*decFnInfo).fastpathDecMapUint32StringR)
	fn(map[uint32]uint(nil), (*encFnInfo).fastpathEncMapUint32UintR, (*decFnInfo).fastpathDecMapUint32UintR)
	fn(map[uint32]uint8(nil), (*encFnInfo).fastpathEncMapUint32Uint8R, (*decFnInfo).fastpathDecMapUint32Uint8R)
	fn(map[uint32]uint16(nil), (*encFnInfo).fastpathEncMapUint32Uint16R, (*decFnInfo).fastpathDecMapUint32Uint16R)
	fn(map[uint32]uint32(nil), (*encFnInfo).fastpathEncMapUint32Uint32R, (*decFnInfo).fastpathDecMapUint32Uint32R)
	fn(map[uint32]uint64(nil), (*encFnInfo).fastpathEncMapUint32Uint64R, (*decFnInfo).fastpathDecMapUint32Uint64R)
	fn(map[uint32]int(nil), (*encFnInfo).fastpathEncMapUint32IntR, (*decFnInfo).fastpathDecMapUint32IntR)
	fn(map[uint32]int8(nil), (*encFnInfo).fastpathEncMapUint32Int8R, (*decFnInfo).fastpathDecMapUint32Int8R)
	fn(map[uint32]int16(nil), (*encFnInfo).fastpathEncMapUint32Int16R, (*decFnInfo).fastpathDecMapUint32Int16R)
	fn(map[uint32]int32(nil), (*encFnInfo).fastpathEncMapUint32Int32R, (*decFnInfo).fastpathDecMapUint32Int32R)
	fn(map[uint32]int64(nil), (*encFnInfo).fastpathEncMapUint32Int64R, (*decFnInfo).fastpathDecMapUint32Int64R)
	fn(map[uint32]float32(nil), (*encFnInfo).fastpathEncMapUint32Float32R, (*decFnInfo).fastpathDecMapUint32Float32R)
	fn(map[uint32]float64(nil), (*encFnInfo).fastpathEncMapUint32Float64R, (*decFnInfo).fastpathDecMapUint32Float64R)
	fn(map[uint32]bool(nil), (*encFnInfo).fastpathEncMapUint32BoolR, (*decFnInfo).fastpathDecMapUint32BoolR)
	fn(map[uint64]interface{}(nil), (*encFnInfo).fastpathEncMapUint64IntfR, (*decFnInfo).fastpathDecMapUint64IntfR)
	fn(map[uint64]string(nil), (*encFnInfo).fastpathEncMapUint64StringR, (*decFnInfo).fastpathDecMapUint64StringR)
	fn(map[uint64]uint(nil), (*encFnInfo).fastpathEncMapUint64UintR, (*decFnInfo).fastpathDecMapUint64UintR)
	fn(map[uint64]uint8(nil), (*encFnInfo).fastpathEncMapUint64Uint8R, (*decFnInfo).fastpathDecMapUint64Uint8R)
	fn(map[uint64]uint16(nil), (*encFnInfo).fastpathEncMapUint64Uint16R, (*decFnInfo).fastpathDecMapUint64Uint16R)
	fn(map[uint64]uint32(nil), (*encFnInfo).fastpathEncMapUint64Uint32R, (*decFnInfo).fastpathDecMapUint64Uint32R)
	fn(map[uint64]uint64(nil), (*encFnInfo).fastpathEncMapUint64Uint64R, (*decFnInfo).fastpathDecMapUint64Uint64R)
	fn(map[uint64]int(nil), (*encFnInfo).fastpathEncMapUint64IntR, (*decFnInfo).fastpathDecMapUint64IntR)
	fn(map[uint64]int8(nil), (*encFnInfo).fastpathEncMapUint64Int8R, (*decFnInfo).fastpathDecMapUint64Int8R)
	fn(map[uint64]int16(nil), (*encFnInfo).fastpathEncMapUint64Int16R, (*decFnInfo).fastpathDecMapUint64Int16R)
	fn(map[uint64]int32(nil), (*encFnInfo).fastpathEncMapUint64Int32R, (*decFnInfo).fastpathDecMapUint64Int32R)
	fn(map[uint64]int64(nil), (*encFnInfo).fastpathEncMapUint64Int64R, (*decFnInfo).fastpathDecMapUint64Int64R)
	fn(map[uint64]float32(nil), (*encFnInfo).fastpathEncMapUint64Float32R, (*decFnInfo).fastpathDecMapUint64Float32R)
	fn(map[uint64]float64(nil), (*encFnInfo).fastpathEncMapUint64Float64R, (*decFnInfo).fastpathDecMapUint64Float64R)
	fn(map[uint64]bool(nil), (*encFnInfo).fastpathEncMapUint64BoolR, (*decFnInfo).fastpathDecMapUint64BoolR)
	fn(map[int]interface{}(nil), (*encFnInfo).fastpathEncMapIntIntfR, (*decFnInfo).fastpathDecMapIntIntfR)
	fn(map[int]string(nil), (*encFnInfo).fastpathEncMapIntStringR, (*decFnInfo).fastpathDecMapIntStringR)
	fn(map[int]uint(nil), (*encFnInfo).fastpathEncMapIntUintR, (*decFnInfo).fastpathDecMapIntUintR)
	fn(map[int]uint8(nil), (*encFnInfo).fastpathEncMapIntUint8R, (*decFnInfo).fastpathDecMapIntUint8R)
	fn(map[int]uint16(nil), (*encFnInfo).fastpathEncMapIntUint16R, (*decFnInfo).fastpathDecMapIntUint16R)
	fn(map[int]uint32(nil), (*encFnInfo).fastpathEncMapIntUint32R, (*decFnInfo).fastpathDecMapIntUint32R)
	fn(map[int]uint64(nil), (*encFnInfo).fastpathEncMapIntUint64R, (*decFnInfo).fastpathDecMapIntUint64R)
	fn(map[int]int(nil), (*encFnInfo).fastpathEncMapIntIntR, (*decFnInfo).fastpathDecMapIntIntR)
	fn(map[int]int8(nil), (*encFnInfo).fastpathEncMapIntInt8R, (*decFnInfo).fastpathDecMapIntInt8R)
	fn(map[int]int16(nil), (*encFnInfo).fastpathEncMapIntInt16R, (*decFnInfo).fastpathDecMapIntInt16R)
	fn(map[int]int32(nil), (*encFnInfo).fastpathEncMapIntInt32R, (*decFnInfo).fastpathDecMapIntInt32R)
	fn(map[int]int64(nil), (*encFnInfo).fastpathEncMapIntInt64R, (*decFnInfo).fastpathDecMapIntInt64R)
	fn(map[int]float32(nil), (*encFnInfo).fastpathEncMapIntFloat32R, (*decFnInfo).fastpathDecMapIntFloat32R)
	fn(map[int]float64(nil), (*encFnInfo).fastpathEncMapIntFloat64R, (*decFnInfo).fastpathDecMapIntFloat64R)
	fn(map[int]bool(nil), (*encFnInfo).fastpathEncMapIntBoolR, (*decFnInfo).fastpathDecMapIntBoolR)
	fn(map[int8]interface{}(nil), (*encFnInfo).fastpathEncMapInt8IntfR, (*decFnInfo).fastpathDecMapInt8IntfR)
	fn(map[int8]string(nil), (*encFnInfo).fastpathEncMapInt8StringR, (*decFnInfo).fastpathDecMapInt8StringR)
	fn(map[int8]uint(nil), (*encFnInfo).fastpathEncMapInt8UintR, (*decFnInfo).fastpathDecMapInt8UintR)
	fn(map[int8]uint8(nil), (*encFnInfo).fastpathEncMapInt8Uint8R, (*decFnInfo).fastpathDecMapInt8Uint8R)
	fn(map[int8]uint16(nil), (*encFnInfo).fastpathEncMapInt8Uint16R, (*decFnInfo).fastpathDecMapInt8Uint16R)
	fn(map[int8]uint32(nil), (*encFnInfo).fastpathEncMapInt8Uint32R, (*decFnInfo).fastpathDecMapInt8Uint32R)
	fn(map[int8]uint64(nil), (*encFnInfo).fastpathEncMapInt8Uint64R, (*decFnInfo).fastpathDecMapInt8Uint64R)
	fn(map[int8]int(nil), (*encFnInfo).fastpathEncMapInt8IntR, (*decFnInfo).fastpathDecMapInt8IntR)
	fn(map[int8]int8(nil), (*encFnInfo).fastpathEncMapInt8Int8R, (*decFnInfo).fastpathDecMapInt8Int8R)
	fn(map[int8]int16(nil), (*encFnInfo).fastpathEncMapInt8Int16R, (*decFnInfo).fastpathDecMapInt8Int16R)
	fn(map[int8]int32(nil), (*encFnInfo).fastpathEncMapInt8Int32R, (*decFnInfo).fastpathDecMapInt8Int32R)
	fn(map[int8]int64(nil), (*encFnInfo).fastpathEncMapInt8Int64R, (*decFnInfo).fastpathDecMapInt8Int64R)
	fn(map[int8]float32(nil), (*encFnInfo).fastpathEncMapInt8Float32R, (*decFnInfo).fastpathDecMapInt8Float32R)
	fn(map[int8]float64(nil), (*encFnInfo).fastpathEncMapInt8Float64R, (*decFnInfo).fastpathDecMapInt8Float64R)
	fn(map[int8]bool(nil), (*encFnInfo).fastpathEncMapInt8BoolR, (*decFnInfo).fastpathDecMapInt8BoolR)
	fn(map[int16]interface{}(nil), (*encFnInfo).fastpathEncMapInt16IntfR, (*decFnInfo).fastpathDecMapInt16IntfR)
	fn(map[int16]string(nil), (*encFnInfo).fastpathEncMapInt16StringR, (*decFnInfo).fastpathDecMapInt16StringR)
	fn(map[int16]uint(nil), (*encFnInfo).fastpathEncMapInt16UintR, (*decFnInfo).fastpathDecMapInt16UintR)
	fn(map[int16]uint8(nil), (*encFnInfo).fastpathEncMapInt16Uint8R, (*decFnInfo).fastpathDecMapInt16Uint8R)
	fn(map[int16]uint16(nil), (*encFnInfo).fastpathEncMapInt16Uint16R, (*decFnInfo).fastpathDecMapInt16Uint16R)
	fn(map[int16]uint32(nil), (*encFnInfo).fastpathEncMapInt16Uint32R, (*decFnInfo).fastpathDecMapInt16Uint32R)
	fn(map[int16]uint64(nil), (*encFnInfo).fastpathEncMapInt16Uint64R, (*decFnInfo).fastpathDecMapInt16Uint64R)
	fn(map[int16]int(nil), (*encFnInfo).fastpathEncMapInt16IntR, (*decFnInfo).fastpathDecMapInt16IntR)
	fn(map[int16]int8(nil), (*encFnInfo).fastpathEncMapInt16Int8R, (*decFnInfo).fastpathDecMapInt16Int8R)
	fn(map[int16]int16(nil), (*encFnInfo).fastpathEncMapInt16Int16R, (*decFnInfo).fastpathDecMapInt16Int16R)
	fn(map[int16]int32(nil), (*encFnInfo).fastpathEncMapInt16Int32R, (*decFnInfo).fastpathDecMapInt16Int32R)
	fn(map[int16]int64(nil), (*encFnInfo).fastpathEncMapInt16Int64R, (*decFnInfo).fastpathDecMapInt16Int64R)
	fn(map[int16]float32(nil), (*encFnInfo).fastpathEncMapInt16Float32R, (*decFnInfo).fastpathDecMapInt16Float32R)
	fn(map[int16]float64(nil), (*encFnInfo).fastpathEncMapInt16Float64R, (*decFnInfo).fastpathDecMapInt16Float64R)
	fn(map[int16]bool(nil), (*encFnInfo).fastpathEncMapInt16BoolR, (*decFnInfo).fastpathDecMapInt16BoolR)
	fn(map[int32]interface{}(nil), (*encFnInfo).fastpathEncMapInt32IntfR, (*decFnInfo).fastpathDecMapInt32IntfR)
	fn(map[int32]string(nil), (*encFnInfo).fastpathEncMapInt32StringR, (*decFnInfo).fastpathDecMapInt32StringR)
	fn(map[int32]uint(nil), (*encFnInfo).fastpathEncMapInt32UintR, (*decFnInfo).fastpathDecMapInt32UintR)
	fn(map[int32]uint8(nil), (*encFnInfo).fastpathEncMapInt32Uint8R, (*decFnInfo).fastpathDecMapInt32Uint8R)
	fn(map[int32]uint16(nil), (*encFnInfo).fastpathEncMapInt32Uint16R, (*decFnInfo).fastpathDecMapInt32Uint16R)
	fn(map[int32]uint32(nil), (*encFnInfo).fastpathEncMapInt32Uint32R, (*decFnInfo).fastpathDecMapInt32Uint32R)
	fn(map[int32]uint64(nil), (*encFnInfo).fastpathEncMapInt32Uint64R, (*decFnInfo).fastpathDecMapInt32Uint64R)
	fn(map[int32]int(nil), (*encFnInfo).fastpathEncMapInt32IntR, (*decFnInfo).fastpathDecMapInt32IntR)
	fn(map[int32]int8(nil), (*encFnInfo).fastpathEncMapInt32Int8R, (*decFnInfo).fastpathDecMapInt32Int8R)
	fn(map[int32]int16(nil), (*encFnInfo).fastpathEncMapInt32Int16R, (*decFnInfo).fastpathDecMapInt32Int16R)
	fn(map[int32]int32(nil), (*encFnInfo).fastpathEncMapInt32Int32R, (*decFnInfo).fastpathDecMapInt32Int32R)
	fn(map[int32]int64(nil), (*encFnInfo).fastpathEncMapInt32Int64R, (*decFnInfo).fastpathDecMapInt32Int64R)
	fn(map[int32]float32(nil), (*encFnInfo).fastpathEncMapInt32Float32R, (*decFnInfo).fastpathDecMapInt32Float32R)
	fn(map[int32]float64(nil), (*encFnInfo).fastpathEncMapInt32Float64R, (*decFnInfo).fastpathDecMapInt32Float64R)
	fn(map[int32]bool(nil), (*encFnInfo).fastpathEncMapInt32BoolR, (*decFnInfo).fastpathDecMapInt32BoolR)
	fn(map[int64]interface{}(nil), (*encFnInfo).fastpathEncMapInt64IntfR, (*decFnInfo).fastpathDecMapInt64IntfR)
	fn(map[int64]string(nil), (*encFnInfo).fastpathEncMapInt64StringR, (*decFnInfo).fastpathDecMapInt64StringR)
	fn(map[int64]uint(nil), (*encFnInfo).fastpathEncMapInt64UintR, (*decFnInfo).fastpathDecMapInt64UintR)
	fn(map[int64]uint8(nil), (*encFnInfo).fastpathEncMapInt64Uint8R, (*decFnInfo).fastpathDecMapInt64Uint8R)
	fn(map[int64]uint16(nil), (*encFnInfo).fastpathEncMapInt64Uint16R, (*decFnInfo).fastpathDecMapInt64Uint16R)
	fn(map[int64]uint32(nil), (*encFnInfo).fastpathEncMapInt64Uint32R, (*decFnInfo).fastpathDecMapInt64Uint32R)
	fn(map[int64]uint64(nil), (*encFnInfo).fastpathEncMapInt64Uint64R, (*decFnInfo).fastpathDecMapInt64Uint64R)
	fn(map[int64]int(nil), (*encFnInfo).fastpathEncMapInt64IntR, (*decFnInfo).fastpathDecMapInt64IntR)
	fn(map[int64]int8(nil), (*encFnInfo).fastpathEncMapInt64Int8R, (*decFnInfo).fastpathDecMapInt64Int8R)
	fn(map[int64]int16(nil), (*encFnInfo).fastpathEncMapInt64Int16R, (*decFnInfo).fastpathDecMapInt64Int16R)
	fn(map[int64]int32(nil), (*encFnInfo).fastpathEncMapInt64Int32R, (*decFnInfo).fastpathDecMapInt64Int32R)
	fn(map[int64]int64(nil), (*encFnInfo).fastpathEncMapInt64Int64R, (*decFnInfo).fastpathDecMapInt64Int64R)
	fn(map[int64]float32(nil), (*encFnInfo).fastpathEncMapInt64Float32R, (*decFnInfo).fastpathDecMapInt64Float32R)
	fn(map[int64]float64(nil), (*encFnInfo).fastpathEncMapInt64Float64R, (*decFnInfo).fastpathDecMapInt64Float64R)
	fn(map[int64]bool(nil), (*encFnInfo).fastpathEncMapInt64BoolR, (*decFnInfo).fastpathDecMapInt64BoolR)
	fn(map[bool]interface{}(nil), (*encFnInfo).fastpathEncMapBoolIntfR, (*decFnInfo).fastpathDecMapBoolIntfR)
	fn(map[bool]string(nil), (*encFnInfo).fastpathEncMapBoolStringR, (*decFnInfo).fastpathDecMapBoolStringR)
	fn(map[bool]uint(nil), (*encFnInfo).fastpathEncMapBoolUintR, (*decFnInfo).fastpathDecMapBoolUintR)
	fn(map[bool]uint8(nil), (*encFnInfo).fastpathEncMapBoolUint8R, (*decFnInfo).fastpathDecMapBoolUint8R)
	fn(map[bool]uint16(nil), (*encFnInfo).fastpathEncMapBoolUint16R, (*decFnInfo).fastpathDecMapBoolUint16R)
	fn(map[bool]uint32(nil), (*encFnInfo).fastpathEncMapBoolUint32R, (*decFnInfo).fastpathDecMapBoolUint32R)
	fn(map[bool]uint64(nil), (*encFnInfo).fastpathEncMapBoolUint64R, (*decFnInfo).fastpathDecMapBoolUint64R)
	fn(map[bool]int(nil), (*encFnInfo).fastpathEncMapBoolIntR, (*decFnInfo).fastpathDecMapBoolIntR)
	fn(map[bool]int8(nil), (*encFnInfo).fastpathEncMapBoolInt8R, (*decFnInfo).fastpathDecMapBoolInt8R)
	fn(map[bool]int16(nil), (*encFnInfo).fastpathEncMapBoolInt16R, (*decFnInfo).fastpathDecMapBoolInt16R)
	fn(map[bool]int32(nil), (*encFnInfo).fastpathEncMapBoolInt32R, (*decFnInfo).fastpathDecMapBoolInt32R)
	fn(map[bool]int64(nil), (*encFnInfo).fastpathEncMapBoolInt64R, (*decFnInfo).fastpathDecMapBoolInt64R)
	fn(map[bool]float32(nil), (*encFnInfo).fastpathEncMapBoolFloat32R, (*decFnInfo).fastpathDecMapBoolFloat32R)
	fn(map[bool]float64(nil), (*encFnInfo).fastpathEncMapBoolFloat64R, (*decFnInfo).fastpathDecMapBoolFloat64R)
	fn(map[bool]bool(nil), (*encFnInfo).fastpathEncMapBoolBoolR, (*decFnInfo).fastpathDecMapBoolBoolR)

	sort.Sort(fastpathAslice(fastpathAV[:]))
}

// -- encode

// -- -- fast path type switch
func fastpathEncodeTypeSwitch(iv interface{}, e *Encoder) bool {
	switch v := iv.(type) {

	case []interface{}:
		fastpathTV.EncSliceIntfV(v, fastpathCheckNilTrue, e)
	case *[]interface{}:
		fastpathTV.EncSliceIntfV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]string:
		fastpathTV.EncMapIntfStringV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]string:
		fastpathTV.EncMapIntfStringV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int:
		fastpathTV.EncMapIntfIntV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int:
		fastpathTV.EncMapIntfIntV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(*v, fastpathCheckNilTrue, e)

	case []string:
		fastpathTV.EncSliceStringV(v, fastpathCheckNilTrue, e)
	case *[]string:
		fastpathTV.EncSliceStringV(*v, fastpathCheckNilTrue, e)

	case map[string]interface{}:
		fastpathTV.EncMapStringIntfV(v, fastpathCheckNilTrue, e)
	case *map[string]interface{}:
		fastpathTV.EncMapStringIntfV(*v, fastpathCheckNilTrue, e)

	case map[string]string:
		fastpathTV.EncMapStringStringV(v, fastpathCheckNilTrue, e)
	case *map[string]string:
		fastpathTV.EncMapStringStringV(*v, fastpathCheckNilTrue, e)

	case map[string]uint:
		fastpathTV.EncMapStringUintV(v, fastpathCheckNilTrue, e)
	case *map[string]uint:
		fastpathTV.EncMapStringUintV(*v, fastpathCheckNilTrue, e)

	case map[string]uint8:
		fastpathTV.EncMapStringUint8V(v, fastpathCheckNilTrue, e)
	case *map[string]uint8:
		fastpathTV.EncMapStringUint8V(*v, fastpathCheckNilTrue, e)

	case map[string]uint16:
		fastpathTV.EncMapStringUint16V(v, fastpathCheckNilTrue, e)
	case *map[string]uint16:
		fastpathTV.EncMapStringUint16V(*v, fastpathCheckNilTrue, e)

	case map[string]uint32:
		fastpathTV.EncMapStringUint32V(v, fastpathCheckNilTrue, e)
	case *map[string]uint32:
		fastpathTV.EncMapStringUint32V(*v, fastpathCheckNilTrue, e)

	case map[string]uint64:
		fastpathTV.EncMapStringUint64V(v, fastpathCheckNilTrue, e)
	case *map[string]uint64:
		fastpathTV.EncMapStringUint64V(*v, fastpathCheckNilTrue, e)

	case map[string]int:
		fastpathTV.EncMapStringIntV(v, fastpathCheckNilTrue, e)
	case *map[string]int:
		fastpathTV.EncMapStringIntV(*v, fastpathCheckNilTrue, e)

	case map[string]int8:
		fastpathTV.EncMapStringInt8V(v, fastpathCheckNilTrue, e)
	case *map[string]int8:
		fastpathTV.EncMapStringInt8V(*v, fastpathCheckNilTrue, e)

	case map[string]int16:
		fastpathTV.EncMapStringInt16V(v, fastpathCheckNilTrue, e)
	case *map[string]int16:
		fastpathTV.EncMapStringInt16V(*v, fastpathCheckNilTrue, e)

	case map[string]int32:
		fastpathTV.EncMapStringInt32V(v, fastpathCheckNilTrue, e)
	case *map[string]int32:
		fastpathTV.EncMapStringInt32V(*v, fastpathCheckNilTrue, e)

	case map[string]int64:
		fastpathTV.EncMapStringInt64V(v, fastpathCheckNilTrue, e)
	case *map[string]int64:
		fastpathTV.EncMapStringInt64V(*v, fastpathCheckNilTrue, e)

	case map[string]float32:
		fastpathTV.EncMapStringFloat32V(v, fastpathCheckNilTrue, e)
	case *map[string]float32:
		fastpathTV.EncMapStringFloat32V(*v, fastpathCheckNilTrue, e)

	case map[string]float64:
		fastpathTV.EncMapStringFloat64V(v, fastpathCheckNilTrue, e)
	case *map[string]float64:
		fastpathTV.EncMapStringFloat64V(*v, fastpathCheckNilTrue, e)

	case map[string]bool:
		fastpathTV.EncMapStringBoolV(v, fastpathCheckNilTrue, e)
	case *map[string]bool:
		fastpathTV.EncMapStringBoolV(*v, fastpathCheckNilTrue, e)

	case []float32:
		fastpathTV.EncSliceFloat32V(v, fastpathCheckNilTrue, e)
	case *[]float32:
		fastpathTV.EncSliceFloat32V(*v, fastpathCheckNilTrue, e)

	case map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(v, fastpathCheckNilTrue, e)
	case *map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(*v, fastpathCheckNilTrue, e)

	case map[float32]string:
		fastpathTV.EncMapFloat32StringV(v, fastpathCheckNilTrue, e)
	case *map[float32]string:
		fastpathTV.EncMapFloat32StringV(*v, fastpathCheckNilTrue, e)

	case map[float32]uint:
		fastpathTV.EncMapFloat32UintV(v, fastpathCheckNilTrue, e)
	case *map[float32]uint:
		fastpathTV.EncMapFloat32UintV(*v, fastpathCheckNilTrue, e)

	case map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(*v, fastpathCheckNilTrue, e)

	case map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(*v, fastpathCheckNilTrue, e)

	case map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(*v, fastpathCheckNilTrue, e)

	case map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(*v, fastpathCheckNilTrue, e)

	case map[float32]int:
		fastpathTV.EncMapFloat32IntV(v, fastpathCheckNilTrue, e)
	case *map[float32]int:
		fastpathTV.EncMapFloat32IntV(*v, fastpathCheckNilTrue, e)

	case map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(v, fastpathCheckNilTrue, e)
	case *map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(*v, fastpathCheckNilTrue, e)

	case map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(v, fastpathCheckNilTrue, e)
	case *map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(*v, fastpathCheckNilTrue, e)

	case map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(v, fastpathCheckNilTrue, e)
	case *map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(*v, fastpathCheckNilTrue, e)

	case map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(v, fastpathCheckNilTrue, e)
	case *map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(*v, fastpathCheckNilTrue, e)

	case map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(v, fastpathCheckNilTrue, e)
	case *map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(*v, fastpathCheckNilTrue, e)

	case map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(v, fastpathCheckNilTrue, e)
	case *map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(*v, fastpathCheckNilTrue, e)

	case map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(v, fastpathCheckNilTrue, e)
	case *map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(*v, fastpathCheckNilTrue, e)

	case []float64:
		fastpathTV.EncSliceFloat64V(v, fastpathCheckNilTrue, e)
	case *[]float64:
		fastpathTV.EncSliceFloat64V(*v, fastpathCheckNilTrue, e)

	case map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(v, fastpathCheckNilTrue, e)
	case *map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(*v, fastpathCheckNilTrue, e)

	case map[float64]string:
		fastpathTV.EncMapFloat64StringV(v, fastpathCheckNilTrue, e)
	case *map[float64]string:
		fastpathTV.EncMapFloat64StringV(*v, fastpathCheckNilTrue, e)

	case map[float64]uint:
		fastpathTV.EncMapFloat64UintV(v, fastpathCheckNilTrue, e)
	case *map[float64]uint:
		fastpathTV.EncMapFloat64UintV(*v, fastpathCheckNilTrue, e)

	case map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(*v, fastpathCheckNilTrue, e)

	case map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(*v, fastpathCheckNilTrue, e)

	case map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(*v, fastpathCheckNilTrue, e)

	case map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(*v, fastpathCheckNilTrue, e)

	case map[float64]int:
		fastpathTV.EncMapFloat64IntV(v, fastpathCheckNilTrue, e)
	case *map[float64]int:
		fastpathTV.EncMapFloat64IntV(*v, fastpathCheckNilTrue, e)

	case map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(v, fastpathCheckNilTrue, e)
	case *map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(*v, fastpathCheckNilTrue, e)

	case map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(v, fastpathCheckNilTrue, e)
	case *map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(*v, fastpathCheckNilTrue, e)

	case map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(v, fastpathCheckNilTrue, e)
	case *map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(*v, fastpathCheckNilTrue, e)

	case map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(v, fastpathCheckNilTrue, e)
	case *map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(*v, fastpathCheckNilTrue, e)

	case map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(v, fastpathCheckNilTrue, e)
	case *map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(*v, fastpathCheckNilTrue, e)

	case map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(v, fastpathCheckNilTrue, e)
	case *map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(*v, fastpathCheckNilTrue, e)

	case map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(v, fastpathCheckNilTrue, e)
	case *map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(*v, fastpathCheckNilTrue, e)

	case []uint:
		fastpathTV.EncSliceUintV(v, fastpathCheckNilTrue, e)
	case *[]uint:
		fastpathTV.EncSliceUintV(*v, fastpathCheckNilTrue, e)

	case map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(v, fastpathCheckNilTrue, e)
	case *map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(*v, fastpathCheckNilTrue, e)

	case map[uint]string:
		fastpathTV.EncMapUintStringV(v, fastpathCheckNilTrue, e)
	case *map[uint]string:
		fastpathTV.EncMapUintStringV(*v, fastpathCheckNilTrue, e)

	case map[uint]uint:
		fastpathTV.EncMapUintUintV(v, fastpathCheckNilTrue, e)
	case *map[uint]uint:
		fastpathTV.EncMapUintUintV(*v, fastpathCheckNilTrue, e)

	case map[uint]uint8:
		fastpathTV.EncMapUintUint8V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint8:
		fastpathTV.EncMapUintUint8V(*v, fastpathCheckNilTrue, e)

	case map[uint]uint16:
		fastpathTV.EncMapUintUint16V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint16:
		fastpathTV.EncMapUintUint16V(*v, fastpathCheckNilTrue, e)

	case map[uint]uint32:
		fastpathTV.EncMapUintUint32V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint32:
		fastpathTV.EncMapUintUint32V(*v, fastpathCheckNilTrue, e)

	case map[uint]uint64:
		fastpathTV.EncMapUintUint64V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint64:
		fastpathTV.EncMapUintUint64V(*v, fastpathCheckNilTrue, e)

	case map[uint]int:
		fastpathTV.EncMapUintIntV(v, fastpathCheckNilTrue, e)
	case *map[uint]int:
		fastpathTV.EncMapUintIntV(*v, fastpathCheckNilTrue, e)

	case map[uint]int8:
		fastpathTV.EncMapUintInt8V(v, fastpathCheckNilTrue, e)
	case *map[uint]int8:
		fastpathTV.EncMapUintInt8V(*v, fastpathCheckNilTrue, e)

	case map[uint]int16:
		fastpathTV.EncMapUintInt16V(v, fastpathCheckNilTrue, e)
	case *map[uint]int16:
		fastpathTV.EncMapUintInt16V(*v, fastpathCheckNilTrue, e)

	case map[uint]int32:
		fastpathTV.EncMapUintInt32V(v, fastpathCheckNilTrue, e)
	case *map[uint]int32:
		fastpathTV.EncMapUintInt32V(*v, fastpathCheckNilTrue, e)

	case map[uint]int64:
		fastpathTV.EncMapUintInt64V(v, fastpathCheckNilTrue, e)
	case *map[uint]int64:
		fastpathTV.EncMapUintInt64V(*v, fastpathCheckNilTrue, e)

	case map[uint]float32:
		fastpathTV.EncMapUintFloat32V(v, fastpathCheckNilTrue, e)
	case *map[uint]float32:
		fastpathTV.EncMapUintFloat32V(*v, fastpathCheckNilTrue, e)

	case map[uint]float64:
		fastpathTV.EncMapUintFloat64V(v, fastpathCheckNilTrue, e)
	case *map[uint]float64:
		fastpathTV.EncMapUintFloat64V(*v, fastpathCheckNilTrue, e)

	case map[uint]bool:
		fastpathTV.EncMapUintBoolV(v, fastpathCheckNilTrue, e)
	case *map[uint]bool:
		fastpathTV.EncMapUintBoolV(*v, fastpathCheckNilTrue, e)

	case map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint8]string:
		fastpathTV.EncMapUint8StringV(v, fastpathCheckNilTrue, e)
	case *map[uint8]string:
		fastpathTV.EncMapUint8StringV(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint:
		fastpathTV.EncMapUint8UintV(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint:
		fastpathTV.EncMapUint8UintV(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int:
		fastpathTV.EncMapUint8IntV(v, fastpathCheckNilTrue, e)
	case *map[uint8]int:
		fastpathTV.EncMapUint8IntV(*v, fastpathCheckNilTrue, e)

	case map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(*v, fastpathCheckNilTrue, e)

	case []uint16:
		fastpathTV.EncSliceUint16V(v, fastpathCheckNilTrue, e)
	case *[]uint16:
		fastpathTV.EncSliceUint16V(*v, fastpathCheckNilTrue, e)

	case map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint16]string:
		fastpathTV.EncMapUint16StringV(v, fastpathCheckNilTrue, e)
	case *map[uint16]string:
		fastpathTV.EncMapUint16StringV(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint:
		fastpathTV.EncMapUint16UintV(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint:
		fastpathTV.EncMapUint16UintV(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int:
		fastpathTV.EncMapUint16IntV(v, fastpathCheckNilTrue, e)
	case *map[uint16]int:
		fastpathTV.EncMapUint16IntV(*v, fastpathCheckNilTrue, e)

	case map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(*v, fastpathCheckNilTrue, e)

	case []uint32:
		fastpathTV.EncSliceUint32V(v, fastpathCheckNilTrue, e)
	case *[]uint32:
		fastpathTV.EncSliceUint32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint32]string:
		fastpathTV.EncMapUint32StringV(v, fastpathCheckNilTrue, e)
	case *map[uint32]string:
		fastpathTV.EncMapUint32StringV(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint:
		fastpathTV.EncMapUint32UintV(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint:
		fastpathTV.EncMapUint32UintV(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int:
		fastpathTV.EncMapUint32IntV(v, fastpathCheckNilTrue, e)
	case *map[uint32]int:
		fastpathTV.EncMapUint32IntV(*v, fastpathCheckNilTrue, e)

	case map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(*v, fastpathCheckNilTrue, e)

	case []uint64:
		fastpathTV.EncSliceUint64V(v, fastpathCheckNilTrue, e)
	case *[]uint64:
		fastpathTV.EncSliceUint64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint64]string:
		fastpathTV.EncMapUint64StringV(v, fastpathCheckNilTrue, e)
	case *map[uint64]string:
		fastpathTV.EncMapUint64StringV(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint:
		fastpathTV.EncMapUint64UintV(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint:
		fastpathTV.EncMapUint64UintV(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int:
		fastpathTV.EncMapUint64IntV(v, fastpathCheckNilTrue, e)
	case *map[uint64]int:
		fastpathTV.EncMapUint64IntV(*v, fastpathCheckNilTrue, e)

	case map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(*v, fastpathCheckNilTrue, e)

	case []int:
		fastpathTV.EncSliceIntV(v, fastpathCheckNilTrue, e)
	case *[]int:
		fastpathTV.EncSliceIntV(*v, fastpathCheckNilTrue, e)

	case map[int]interface{}:
		fastpathTV.EncMapIntIntfV(v, fastpathCheckNilTrue, e)
	case *map[int]interface{}:
		fastpathTV.EncMapIntIntfV(*v, fastpathCheckNilTrue, e)

	case map[int]string:
		fastpathTV.EncMapIntStringV(v, fastpathCheckNilTrue, e)
	case *map[int]string:
		fastpathTV.EncMapIntStringV(*v, fastpathCheckNilTrue, e)

	case map[int]uint:
		fastpathTV.EncMapIntUintV(v, fastpathCheckNilTrue, e)
	case *map[int]uint:
		fastpathTV.EncMapIntUintV(*v, fastpathCheckNilTrue, e)

	case map[int]uint8:
		fastpathTV.EncMapIntUint8V(v, fastpathCheckNilTrue, e)
	case *map[int]uint8:
		fastpathTV.EncMapIntUint8V(*v, fastpathCheckNilTrue, e)

	case map[int]uint16:
		fastpathTV.EncMapIntUint16V(v, fastpathCheckNilTrue, e)
	case *map[int]uint16:
		fastpathTV.EncMapIntUint16V(*v, fastpathCheckNilTrue, e)

	case map[int]uint32:
		fastpathTV.EncMapIntUint32V(v, fastpathCheckNilTrue, e)
	case *map[int]uint32:
		fastpathTV.EncMapIntUint32V(*v, fastpathCheckNilTrue, e)

	case map[int]uint64:
		fastpathTV.EncMapIntUint64V(v, fastpathCheckNilTrue, e)
	case *map[int]uint64:
		fastpathTV.EncMapIntUint64V(*v, fastpathCheckNilTrue, e)

	case map[int]int:
		fastpathTV.EncMapIntIntV(v, fastpathCheckNilTrue, e)
	case *map[int]int:
		fastpathTV.EncMapIntIntV(*v, fastpathCheckNilTrue, e)

	case map[int]int8:
		fastpathTV.EncMapIntInt8V(v, fastpathCheckNilTrue, e)
	case *map[int]int8:
		fastpathTV.EncMapIntInt8V(*v, fastpathCheckNilTrue, e)

	case map[int]int16:
		fastpathTV.EncMapIntInt16V(v, fastpathCheckNilTrue, e)
	case *map[int]int16:
		fastpathTV.EncMapIntInt16V(*v, fastpathCheckNilTrue, e)

	case map[int]int32:
		fastpathTV.EncMapIntInt32V(v, fastpathCheckNilTrue, e)
	case *map[int]int32:
		fastpathTV.EncMapIntInt32V(*v, fastpathCheckNilTrue, e)

	case map[int]int64:
		fastpathTV.EncMapIntInt64V(v, fastpathCheckNilTrue, e)
	case *map[int]int64:
		fastpathTV.EncMapIntInt64V(*v, fastpathCheckNilTrue, e)

	case map[int]float32:
		fastpathTV.EncMapIntFloat32V(v, fastpathCheckNilTrue, e)
	case *map[int]float32:
		fastpathTV.EncMapIntFloat32V(*v, fastpathCheckNilTrue, e)

	case map[int]float64:
		fastpathTV.EncMapIntFloat64V(v, fastpathCheckNilTrue, e)
	case *map[int]float64:
		fastpathTV.EncMapIntFloat64V(*v, fastpathCheckNilTrue, e)

	case map[int]bool:
		fastpathTV.EncMapIntBoolV(v, fastpathCheckNilTrue, e)
	case *map[int]bool:
		fastpathTV.EncMapIntBoolV(*v, fastpathCheckNilTrue, e)

	case []int8:
		fastpathTV.EncSliceInt8V(v, fastpathCheckNilTrue, e)
	case *[]int8:
		fastpathTV.EncSliceInt8V(*v, fastpathCheckNilTrue, e)

	case map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(v, fastpathCheckNilTrue, e)
	case *map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(*v, fastpathCheckNilTrue, e)

	case map[int8]string:
		fastpathTV.EncMapInt8StringV(v, fastpathCheckNilTrue, e)
	case *map[int8]string:
		fastpathTV.EncMapInt8StringV(*v, fastpathCheckNilTrue, e)

	case map[int8]uint:
		fastpathTV.EncMapInt8UintV(v, fastpathCheckNilTrue, e)
	case *map[int8]uint:
		fastpathTV.EncMapInt8UintV(*v, fastpathCheckNilTrue, e)

	case map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int8]int:
		fastpathTV.EncMapInt8IntV(v, fastpathCheckNilTrue, e)
	case *map[int8]int:
		fastpathTV.EncMapInt8IntV(*v, fastpathCheckNilTrue, e)

	case map[int8]int8:
		fastpathTV.EncMapInt8Int8V(v, fastpathCheckNilTrue, e)
	case *map[int8]int8:
		fastpathTV.EncMapInt8Int8V(*v, fastpathCheckNilTrue, e)

	case map[int8]int16:
		fastpathTV.EncMapInt8Int16V(v, fastpathCheckNilTrue, e)
	case *map[int8]int16:
		fastpathTV.EncMapInt8Int16V(*v, fastpathCheckNilTrue, e)

	case map[int8]int32:
		fastpathTV.EncMapInt8Int32V(v, fastpathCheckNilTrue, e)
	case *map[int8]int32:
		fastpathTV.EncMapInt8Int32V(*v, fastpathCheckNilTrue, e)

	case map[int8]int64:
		fastpathTV.EncMapInt8Int64V(v, fastpathCheckNilTrue, e)
	case *map[int8]int64:
		fastpathTV.EncMapInt8Int64V(*v, fastpathCheckNilTrue, e)

	case map[int8]float32:
		fastpathTV.EncMapInt8Float32V(v, fastpathCheckNilTrue, e)
	case *map[int8]float32:
		fastpathTV.EncMapInt8Float32V(*v, fastpathCheckNilTrue, e)

	case map[int8]float64:
		fastpathTV.EncMapInt8Float64V(v, fastpathCheckNilTrue, e)
	case *map[int8]float64:
		fastpathTV.EncMapInt8Float64V(*v, fastpathCheckNilTrue, e)

	case map[int8]bool:
		fastpathTV.EncMapInt8BoolV(v, fastpathCheckNilTrue, e)
	case *map[int8]bool:
		fastpathTV.EncMapInt8BoolV(*v, fastpathCheckNilTrue, e)

	case []int16:
		fastpathTV.EncSliceInt16V(v, fastpathCheckNilTrue, e)
	case *[]int16:
		fastpathTV.EncSliceInt16V(*v, fastpathCheckNilTrue, e)

	case map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(v, fastpathCheckNilTrue, e)
	case *map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(*v, fastpathCheckNilTrue, e)

	case map[int16]string:
		fastpathTV.EncMapInt16StringV(v, fastpathCheckNilTrue, e)
	case *map[int16]string:
		fastpathTV.EncMapInt16StringV(*v, fastpathCheckNilTrue, e)

	case map[int16]uint:
		fastpathTV.EncMapInt16UintV(v, fastpathCheckNilTrue, e)
	case *map[int16]uint:
		fastpathTV.EncMapInt16UintV(*v, fastpathCheckNilTrue, e)

	case map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int16]int:
		fastpathTV.EncMapInt16IntV(v, fastpathCheckNilTrue, e)
	case *map[int16]int:
		fastpathTV.EncMapInt16IntV(*v, fastpathCheckNilTrue, e)

	case map[int16]int8:
		fastpathTV.EncMapInt16Int8V(v, fastpathCheckNilTrue, e)
	case *map[int16]int8:
		fastpathTV.EncMapInt16Int8V(*v, fastpathCheckNilTrue, e)

	case map[int16]int16:
		fastpathTV.EncMapInt16Int16V(v, fastpathCheckNilTrue, e)
	case *map[int16]int16:
		fastpathTV.EncMapInt16Int16V(*v, fastpathCheckNilTrue, e)

	case map[int16]int32:
		fastpathTV.EncMapInt16Int32V(v, fastpathCheckNilTrue, e)
	case *map[int16]int32:
		fastpathTV.EncMapInt16Int32V(*v, fastpathCheckNilTrue, e)

	case map[int16]int64:
		fastpathTV.EncMapInt16Int64V(v, fastpathCheckNilTrue, e)
	case *map[int16]int64:
		fastpathTV.EncMapInt16Int64V(*v, fastpathCheckNilTrue, e)

	case map[int16]float32:
		fastpathTV.EncMapInt16Float32V(v, fastpathCheckNilTrue, e)
	case *map[int16]float32:
		fastpathTV.EncMapInt16Float32V(*v, fastpathCheckNilTrue, e)

	case map[int16]float64:
		fastpathTV.EncMapInt16Float64V(v, fastpathCheckNilTrue, e)
	case *map[int16]float64:
		fastpathTV.EncMapInt16Float64V(*v, fastpathCheckNilTrue, e)

	case map[int16]bool:
		fastpathTV.EncMapInt16BoolV(v, fastpathCheckNilTrue, e)
	case *map[int16]bool:
		fastpathTV.EncMapInt16BoolV(*v, fastpathCheckNilTrue, e)

	case []int32:
		fastpathTV.EncSliceInt32V(v, fastpathCheckNilTrue, e)
	case *[]int32:
		fastpathTV.EncSliceInt32V(*v, fastpathCheckNilTrue, e)

	case map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(v, fastpathCheckNilTrue, e)
	case *map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(*v, fastpathCheckNilTrue, e)

	case map[int32]string:
		fastpathTV.EncMapInt32StringV(v, fastpathCheckNilTrue, e)
	case *map[int32]string:
		fastpathTV.EncMapInt32StringV(*v, fastpathCheckNilTrue, e)

	case map[int32]uint:
		fastpathTV.EncMapInt32UintV(v, fastpathCheckNilTrue, e)
	case *map[int32]uint:
		fastpathTV.EncMapInt32UintV(*v, fastpathCheckNilTrue, e)

	case map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int32]int:
		fastpathTV.EncMapInt32IntV(v, fastpathCheckNilTrue, e)
	case *map[int32]int:
		fastpathTV.EncMapInt32IntV(*v, fastpathCheckNilTrue, e)

	case map[int32]int8:
		fastpathTV.EncMapInt32Int8V(v, fastpathCheckNilTrue, e)
	case *map[int32]int8:
		fastpathTV.EncMapInt32Int8V(*v, fastpathCheckNilTrue, e)

	case map[int32]int16:
		fastpathTV.EncMapInt32Int16V(v, fastpathCheckNilTrue, e)
	case *map[int32]int16:
		fastpathTV.EncMapInt32Int16V(*v, fastpathCheckNilTrue, e)

	case map[int32]int32:
		fastpathTV.EncMapInt32Int32V(v, fastpathCheckNilTrue, e)
	case *map[int32]int32:
		fastpathTV.EncMapInt32Int32V(*v, fastpathCheckNilTrue, e)

	case map[int32]int64:
		fastpathTV.EncMapInt32Int64V(v, fastpathCheckNilTrue, e)
	case *map[int32]int64:
		fastpathTV.EncMapInt32Int64V(*v, fastpathCheckNilTrue, e)

	case map[int32]float32:
		fastpathTV.EncMapInt32Float32V(v, fastpathCheckNilTrue, e)
	case *map[int32]float32:
		fastpathTV.EncMapInt32Float32V(*v, fastpathCheckNilTrue, e)

	case map[int32]float64:
		fastpathTV.EncMapInt32Float64V(v, fastpathCheckNilTrue, e)
	case *map[int32]float64:
		fastpathTV.EncMapInt32Float64V(*v, fastpathCheckNilTrue, e)

	case map[int32]bool:
		fastpathTV.EncMapInt32BoolV(v, fastpathCheckNilTrue, e)
	case *map[int32]bool:
		fastpathTV.EncMapInt32BoolV(*v, fastpathCheckNilTrue, e)

	case []int64:
		fastpathTV.EncSliceInt64V(v, fastpathCheckNilTrue, e)
	case *[]int64:
		fastpathTV.EncSliceInt64V(*v, fastpathCheckNilTrue, e)

	case map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(v, fastpathCheckNilTrue, e)
	case *map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(*v, fastpathCheckNilTrue, e)

	case map[int64]string:
		fastpathTV.EncMapInt64StringV(v, fastpathCheckNilTrue, e)
	case *map[int64]string:
		fastpathTV.EncMapInt64StringV(*v, fastpathCheckNilTrue, e)

	case map[int64]uint:
		fastpathTV.EncMapInt64UintV(v, fastpathCheckNilTrue, e)
	case *map[int64]uint:
		fastpathTV.EncMapInt64UintV(*v, fastpathCheckNilTrue, e)

	case map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int64]int:
		fastpathTV.EncMapInt64IntV(v, fastpathCheckNilTrue, e)
	case *map[int64]int:
		fastpathTV.EncMapInt64IntV(*v, fastpathCheckNilTrue, e)

	case map[int64]int8:
		fastpathTV.EncMapInt64Int8V(v, fastpathCheckNilTrue, e)
	case *map[int64]int8:
		fastpathTV.EncMapInt64Int8V(*v, fastpathCheckNilTrue, e)

	case map[int64]int16:
		fastpathTV.EncMapInt64Int16V(v, fastpathCheckNilTrue, e)
	case *map[int64]int16:
		fastpathTV.EncMapInt64Int16V(*v, fastpathCheckNilTrue, e)

	case map[int64]int32:
		fastpathTV.EncMapInt64Int32V(v, fastpathCheckNilTrue, e)
	case *map[int64]int32:
		fastpathTV.EncMapInt64Int32V(*v, fastpathCheckNilTrue, e)

	case map[int64]int64:
		fastpathTV.EncMapInt64Int64V(v, fastpathCheckNilTrue, e)
	case *map[int64]int64:
		fastpathTV.EncMapInt64Int64V(*v, fastpathCheckNilTrue, e)

	case map[int64]float32:
		fastpathTV.EncMapInt64Float32V(v, fastpathCheckNilTrue, e)
	case *map[int64]float32:
		fastpathTV.EncMapInt64Float32V(*v, fastpathCheckNilTrue, e)

	case map[int64]float64:
		fastpathTV.EncMapInt64Float64V(v, fastpathCheckNilTrue, e)
	case *map[int64]float64:
		fastpathTV.EncMapInt64Float64V(*v, fastpathCheckNilTrue, e)

	case map[int64]bool:
		fastpathTV.EncMapInt64BoolV(v, fastpathCheckNilTrue, e)
	case *map[int64]bool:
		fastpathTV.EncMapInt64BoolV(*v, fastpathCheckNilTrue, e)

	case []bool:
		fastpathTV.EncSliceBoolV(v, fastpathCheckNilTrue, e)
	case *[]bool:
		fastpathTV.EncSliceBoolV(*v, fastpathCheckNilTrue, e)

	case map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(v, fastpathCheckNilTrue, e)
	case *map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(*v, fastpathCheckNilTrue, e)

	case map[bool]string:
		fastpathTV.EncMapBoolStringV(v, fastpathCheckNilTrue, e)
	case *map[bool]string:
		fastpathTV.EncMapBoolStringV(*v, fastpathCheckNilTrue, e)

	case map[bool]uint:
		fastpathTV.EncMapBoolUintV(v, fastpathCheckNilTrue, e)
	case *map[bool]uint:
		fastpathTV.EncMapBoolUintV(*v, fastpathCheckNilTrue, e)

	case map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(*v, fastpathCheckNilTrue, e)

	case map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(*v, fastpathCheckNilTrue, e)

	case map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(*v, fastpathCheckNilTrue, e)

	case map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(*v, fastpathCheckNilTrue, e)

	case map[bool]int:
		fastpathTV.EncMapBoolIntV(v, fastpathCheckNilTrue, e)
	case *map[bool]int:
		fastpathTV.EncMapBoolIntV(*v, fastpathCheckNilTrue, e)

	case map[bool]int8:
		fastpathTV.EncMapBoolInt8V(v, fastpathCheckNilTrue, e)
	case *map[bool]int8:
		fastpathTV.EncMapBoolInt8V(*v, fastpathCheckNilTrue, e)

	case map[bool]int16:
		fastpathTV.EncMapBoolInt16V(v, fastpathCheckNilTrue, e)
	case *map[bool]int16:
		fastpathTV.EncMapBoolInt16V(*v, fastpathCheckNilTrue, e)

	case map[bool]int32:
		fastpathTV.EncMapBoolInt32V(v, fastpathCheckNilTrue, e)
	case *map[bool]int32:
		fastpathTV.EncMapBoolInt32V(*v, fastpathCheckNilTrue, e)

	case map[bool]int64:
		fastpathTV.EncMapBoolInt64V(v, fastpathCheckNilTrue, e)
	case *map[bool]int64:
		fastpathTV.EncMapBoolInt64V(*v, fastpathCheckNilTrue, e)

	case map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(v, fastpathCheckNilTrue, e)
	case *map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(*v, fastpathCheckNilTrue, e)

	case map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(v, fastpathCheckNilTrue, e)
	case *map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(*v, fastpathCheckNilTrue, e)

	case map[bool]bool:
		fastpathTV.EncMapBoolBoolV(v, fastpathCheckNilTrue, e)
	case *map[bool]bool:
		fastpathTV.EncMapBoolBoolV(*v, fastpathCheckNilTrue, e)

	default:
		return false
	}
	return true
}

func fastpathEncodeTypeSwitchSlice(iv interface{}, e *Encoder) bool {
	switch v := iv.(type) {

	case []interface{}:
		fastpathTV.EncSliceIntfV(v, fastpathCheckNilTrue, e)
	case *[]interface{}:
		fastpathTV.EncSliceIntfV(*v, fastpathCheckNilTrue, e)

	case []string:
		fastpathTV.EncSliceStringV(v, fastpathCheckNilTrue, e)
	case *[]string:
		fastpathTV.EncSliceStringV(*v, fastpathCheckNilTrue, e)

	case []float32:
		fastpathTV.EncSliceFloat32V(v, fastpathCheckNilTrue, e)
	case *[]float32:
		fastpathTV.EncSliceFloat32V(*v, fastpathCheckNilTrue, e)

	case []float64:
		fastpathTV.EncSliceFloat64V(v, fastpathCheckNilTrue, e)
	case *[]float64:
		fastpathTV.EncSliceFloat64V(*v, fastpathCheckNilTrue, e)

	case []uint:
		fastpathTV.EncSliceUintV(v, fastpathCheckNilTrue, e)
	case *[]uint:
		fastpathTV.EncSliceUintV(*v, fastpathCheckNilTrue, e)

	case []uint16:
		fastpathTV.EncSliceUint16V(v, fastpathCheckNilTrue, e)
	case *[]uint16:
		fastpathTV.EncSliceUint16V(*v, fastpathCheckNilTrue, e)

	case []uint32:
		fastpathTV.EncSliceUint32V(v, fastpathCheckNilTrue, e)
	case *[]uint32:
		fastpathTV.EncSliceUint32V(*v, fastpathCheckNilTrue, e)

	case []uint64:
		fastpathTV.EncSliceUint64V(v, fastpathCheckNilTrue, e)
	case *[]uint64:
		fastpathTV.EncSliceUint64V(*v, fastpathCheckNilTrue, e)

	case []int:
		fastpathTV.EncSliceIntV(v, fastpathCheckNilTrue, e)
	case *[]int:
		fastpathTV.EncSliceIntV(*v, fastpathCheckNilTrue, e)

	case []int8:
		fastpathTV.EncSliceInt8V(v, fastpathCheckNilTrue, e)
	case *[]int8:
		fastpathTV.EncSliceInt8V(*v, fastpathCheckNilTrue, e)

	case []int16:
		fastpathTV.EncSliceInt16V(v, fastpathCheckNilTrue, e)
	case *[]int16:
		fastpathTV.EncSliceInt16V(*v, fastpathCheckNilTrue, e)

	case []int32:
		fastpathTV.EncSliceInt32V(v, fastpathCheckNilTrue, e)
	case *[]int32:
		fastpathTV.EncSliceInt32V(*v, fastpathCheckNilTrue, e)

	case []int64:
		fastpathTV.EncSliceInt64V(v, fastpathCheckNilTrue, e)
	case *[]int64:
		fastpathTV.EncSliceInt64V(*v, fastpathCheckNilTrue, e)

	case []bool:
		fastpathTV.EncSliceBoolV(v, fastpathCheckNilTrue, e)
	case *[]bool:
		fastpathTV.EncSliceBoolV(*v, fastpathCheckNilTrue, e)

	default:
		return false
	}
	return true
}

func fastpathEncodeTypeSwitchMap(iv interface{}, e *Encoder) bool {
	switch v := iv.(type) {

	case map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]string:
		fastpathTV.EncMapIntfStringV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]string:
		fastpathTV.EncMapIntfStringV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int:
		fastpathTV.EncMapIntfIntV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int:
		fastpathTV.EncMapIntfIntV(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(v, fastpathCheckNilTrue, e)
	case *map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(*v, fastpathCheckNilTrue, e)

	case map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(v, fastpathCheckNilTrue, e)
	case *map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(*v, fastpathCheckNilTrue, e)

	case map[string]interface{}:
		fastpathTV.EncMapStringIntfV(v, fastpathCheckNilTrue, e)
	case *map[string]interface{}:
		fastpathTV.EncMapStringIntfV(*v, fastpathCheckNilTrue, e)

	case map[string]string:
		fastpathTV.EncMapStringStringV(v, fastpathCheckNilTrue, e)
	case *map[string]string:
		fastpathTV.EncMapStringStringV(*v, fastpathCheckNilTrue, e)

	case map[string]uint:
		fastpathTV.EncMapStringUintV(v, fastpathCheckNilTrue, e)
	case *map[string]uint:
		fastpathTV.EncMapStringUintV(*v, fastpathCheckNilTrue, e)

	case map[string]uint8:
		fastpathTV.EncMapStringUint8V(v, fastpathCheckNilTrue, e)
	case *map[string]uint8:
		fastpathTV.EncMapStringUint8V(*v, fastpathCheckNilTrue, e)

	case map[string]uint16:
		fastpathTV.EncMapStringUint16V(v, fastpathCheckNilTrue, e)
	case *map[string]uint16:
		fastpathTV.EncMapStringUint16V(*v, fastpathCheckNilTrue, e)

	case map[string]uint32:
		fastpathTV.EncMapStringUint32V(v, fastpathCheckNilTrue, e)
	case *map[string]uint32:
		fastpathTV.EncMapStringUint32V(*v, fastpathCheckNilTrue, e)

	case map[string]uint64:
		fastpathTV.EncMapStringUint64V(v, fastpathCheckNilTrue, e)
	case *map[string]uint64:
		fastpathTV.EncMapStringUint64V(*v, fastpathCheckNilTrue, e)

	case map[string]int:
		fastpathTV.EncMapStringIntV(v, fastpathCheckNilTrue, e)
	case *map[string]int:
		fastpathTV.EncMapStringIntV(*v, fastpathCheckNilTrue, e)

	case map[string]int8:
		fastpathTV.EncMapStringInt8V(v, fastpathCheckNilTrue, e)
	case *map[string]int8:
		fastpathTV.EncMapStringInt8V(*v, fastpathCheckNilTrue, e)

	case map[string]int16:
		fastpathTV.EncMapStringInt16V(v, fastpathCheckNilTrue, e)
	case *map[string]int16:
		fastpathTV.EncMapStringInt16V(*v, fastpathCheckNilTrue, e)

	case map[string]int32:
		fastpathTV.EncMapStringInt32V(v, fastpathCheckNilTrue, e)
	case *map[string]int32:
		fastpathTV.EncMapStringInt32V(*v, fastpathCheckNilTrue, e)

	case map[string]int64:
		fastpathTV.EncMapStringInt64V(v, fastpathCheckNilTrue, e)
	case *map[string]int64:
		fastpathTV.EncMapStringInt64V(*v, fastpathCheckNilTrue, e)

	case map[string]float32:
		fastpathTV.EncMapStringFloat32V(v, fastpathCheckNilTrue, e)
	case *map[string]float32:
		fastpathTV.EncMapStringFloat32V(*v, fastpathCheckNilTrue, e)

	case map[string]float64:
		fastpathTV.EncMapStringFloat64V(v, fastpathCheckNilTrue, e)
	case *map[string]float64:
		fastpathTV.EncMapStringFloat64V(*v, fastpathCheckNilTrue, e)

	case map[string]bool:
		fastpathTV.EncMapStringBoolV(v, fastpathCheckNilTrue, e)
	case *map[string]bool:
		fastpathTV.EncMapStringBoolV(*v, fastpathCheckNilTrue, e)

	case map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(v, fastpathCheckNilTrue, e)
	case *map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(*v, fastpathCheckNilTrue, e)

	case map[float32]string:
		fastpathTV.EncMapFloat32StringV(v, fastpathCheckNilTrue, e)
	case *map[float32]string:
		fastpathTV.EncMapFloat32StringV(*v, fastpathCheckNilTrue, e)

	case map[float32]uint:
		fastpathTV.EncMapFloat32UintV(v, fastpathCheckNilTrue, e)
	case *map[float32]uint:
		fastpathTV.EncMapFloat32UintV(*v, fastpathCheckNilTrue, e)

	case map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(*v, fastpathCheckNilTrue, e)

	case map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(*v, fastpathCheckNilTrue, e)

	case map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(*v, fastpathCheckNilTrue, e)

	case map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(v, fastpathCheckNilTrue, e)
	case *map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(*v, fastpathCheckNilTrue, e)

	case map[float32]int:
		fastpathTV.EncMapFloat32IntV(v, fastpathCheckNilTrue, e)
	case *map[float32]int:
		fastpathTV.EncMapFloat32IntV(*v, fastpathCheckNilTrue, e)

	case map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(v, fastpathCheckNilTrue, e)
	case *map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(*v, fastpathCheckNilTrue, e)

	case map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(v, fastpathCheckNilTrue, e)
	case *map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(*v, fastpathCheckNilTrue, e)

	case map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(v, fastpathCheckNilTrue, e)
	case *map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(*v, fastpathCheckNilTrue, e)

	case map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(v, fastpathCheckNilTrue, e)
	case *map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(*v, fastpathCheckNilTrue, e)

	case map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(v, fastpathCheckNilTrue, e)
	case *map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(*v, fastpathCheckNilTrue, e)

	case map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(v, fastpathCheckNilTrue, e)
	case *map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(*v, fastpathCheckNilTrue, e)

	case map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(v, fastpathCheckNilTrue, e)
	case *map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(*v, fastpathCheckNilTrue, e)

	case map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(v, fastpathCheckNilTrue, e)
	case *map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(*v, fastpathCheckNilTrue, e)

	case map[float64]string:
		fastpathTV.EncMapFloat64StringV(v, fastpathCheckNilTrue, e)
	case *map[float64]string:
		fastpathTV.EncMapFloat64StringV(*v, fastpathCheckNilTrue, e)

	case map[float64]uint:
		fastpathTV.EncMapFloat64UintV(v, fastpathCheckNilTrue, e)
	case *map[float64]uint:
		fastpathTV.EncMapFloat64UintV(*v, fastpathCheckNilTrue, e)

	case map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(*v, fastpathCheckNilTrue, e)

	case map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(*v, fastpathCheckNilTrue, e)

	case map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(*v, fastpathCheckNilTrue, e)

	case map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(v, fastpathCheckNilTrue, e)
	case *map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(*v, fastpathCheckNilTrue, e)

	case map[float64]int:
		fastpathTV.EncMapFloat64IntV(v, fastpathCheckNilTrue, e)
	case *map[float64]int:
		fastpathTV.EncMapFloat64IntV(*v, fastpathCheckNilTrue, e)

	case map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(v, fastpathCheckNilTrue, e)
	case *map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(*v, fastpathCheckNilTrue, e)

	case map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(v, fastpathCheckNilTrue, e)
	case *map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(*v, fastpathCheckNilTrue, e)

	case map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(v, fastpathCheckNilTrue, e)
	case *map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(*v, fastpathCheckNilTrue, e)

	case map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(v, fastpathCheckNilTrue, e)
	case *map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(*v, fastpathCheckNilTrue, e)

	case map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(v, fastpathCheckNilTrue, e)
	case *map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(*v, fastpathCheckNilTrue, e)

	case map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(v, fastpathCheckNilTrue, e)
	case *map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(*v, fastpathCheckNilTrue, e)

	case map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(v, fastpathCheckNilTrue, e)
	case *map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(*v, fastpathCheckNilTrue, e)

	case map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(v, fastpathCheckNilTrue, e)
	case *map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(*v, fastpathCheckNilTrue, e)

	case map[uint]string:
		fastpathTV.EncMapUintStringV(v, fastpathCheckNilTrue, e)
	case *map[uint]string:
		fastpathTV.EncMapUintStringV(*v, fastpathCheckNilTrue, e)

	case map[uint]uint:
		fastpathTV.EncMapUintUintV(v, fastpathCheckNilTrue, e)
	case *map[uint]uint:
		fastpathTV.EncMapUintUintV(*v, fastpathCheckNilTrue, e)

	case map[uint]uint8:
		fastpathTV.EncMapUintUint8V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint8:
		fastpathTV.EncMapUintUint8V(*v, fastpathCheckNilTrue, e)

	case map[uint]uint16:
		fastpathTV.EncMapUintUint16V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint16:
		fastpathTV.EncMapUintUint16V(*v, fastpathCheckNilTrue, e)

	case map[uint]uint32:
		fastpathTV.EncMapUintUint32V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint32:
		fastpathTV.EncMapUintUint32V(*v, fastpathCheckNilTrue, e)

	case map[uint]uint64:
		fastpathTV.EncMapUintUint64V(v, fastpathCheckNilTrue, e)
	case *map[uint]uint64:
		fastpathTV.EncMapUintUint64V(*v, fastpathCheckNilTrue, e)

	case map[uint]int:
		fastpathTV.EncMapUintIntV(v, fastpathCheckNilTrue, e)
	case *map[uint]int:
		fastpathTV.EncMapUintIntV(*v, fastpathCheckNilTrue, e)

	case map[uint]int8:
		fastpathTV.EncMapUintInt8V(v, fastpathCheckNilTrue, e)
	case *map[uint]int8:
		fastpathTV.EncMapUintInt8V(*v, fastpathCheckNilTrue, e)

	case map[uint]int16:
		fastpathTV.EncMapUintInt16V(v, fastpathCheckNilTrue, e)
	case *map[uint]int16:
		fastpathTV.EncMapUintInt16V(*v, fastpathCheckNilTrue, e)

	case map[uint]int32:
		fastpathTV.EncMapUintInt32V(v, fastpathCheckNilTrue, e)
	case *map[uint]int32:
		fastpathTV.EncMapUintInt32V(*v, fastpathCheckNilTrue, e)

	case map[uint]int64:
		fastpathTV.EncMapUintInt64V(v, fastpathCheckNilTrue, e)
	case *map[uint]int64:
		fastpathTV.EncMapUintInt64V(*v, fastpathCheckNilTrue, e)

	case map[uint]float32:
		fastpathTV.EncMapUintFloat32V(v, fastpathCheckNilTrue, e)
	case *map[uint]float32:
		fastpathTV.EncMapUintFloat32V(*v, fastpathCheckNilTrue, e)

	case map[uint]float64:
		fastpathTV.EncMapUintFloat64V(v, fastpathCheckNilTrue, e)
	case *map[uint]float64:
		fastpathTV.EncMapUintFloat64V(*v, fastpathCheckNilTrue, e)

	case map[uint]bool:
		fastpathTV.EncMapUintBoolV(v, fastpathCheckNilTrue, e)
	case *map[uint]bool:
		fastpathTV.EncMapUintBoolV(*v, fastpathCheckNilTrue, e)

	case map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint8]string:
		fastpathTV.EncMapUint8StringV(v, fastpathCheckNilTrue, e)
	case *map[uint8]string:
		fastpathTV.EncMapUint8StringV(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint:
		fastpathTV.EncMapUint8UintV(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint:
		fastpathTV.EncMapUint8UintV(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int:
		fastpathTV.EncMapUint8IntV(v, fastpathCheckNilTrue, e)
	case *map[uint8]int:
		fastpathTV.EncMapUint8IntV(*v, fastpathCheckNilTrue, e)

	case map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(*v, fastpathCheckNilTrue, e)

	case map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint16]string:
		fastpathTV.EncMapUint16StringV(v, fastpathCheckNilTrue, e)
	case *map[uint16]string:
		fastpathTV.EncMapUint16StringV(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint:
		fastpathTV.EncMapUint16UintV(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint:
		fastpathTV.EncMapUint16UintV(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int:
		fastpathTV.EncMapUint16IntV(v, fastpathCheckNilTrue, e)
	case *map[uint16]int:
		fastpathTV.EncMapUint16IntV(*v, fastpathCheckNilTrue, e)

	case map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(*v, fastpathCheckNilTrue, e)

	case map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint32]string:
		fastpathTV.EncMapUint32StringV(v, fastpathCheckNilTrue, e)
	case *map[uint32]string:
		fastpathTV.EncMapUint32StringV(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint:
		fastpathTV.EncMapUint32UintV(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint:
		fastpathTV.EncMapUint32UintV(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int:
		fastpathTV.EncMapUint32IntV(v, fastpathCheckNilTrue, e)
	case *map[uint32]int:
		fastpathTV.EncMapUint32IntV(*v, fastpathCheckNilTrue, e)

	case map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(*v, fastpathCheckNilTrue, e)

	case map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(v, fastpathCheckNilTrue, e)
	case *map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(*v, fastpathCheckNilTrue, e)

	case map[uint64]string:
		fastpathTV.EncMapUint64StringV(v, fastpathCheckNilTrue, e)
	case *map[uint64]string:
		fastpathTV.EncMapUint64StringV(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint:
		fastpathTV.EncMapUint64UintV(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint:
		fastpathTV.EncMapUint64UintV(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(*v, fastpathCheckNilTrue, e)

	case map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(v, fastpathCheckNilTrue, e)
	case *map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int:
		fastpathTV.EncMapUint64IntV(v, fastpathCheckNilTrue, e)
	case *map[uint64]int:
		fastpathTV.EncMapUint64IntV(*v, fastpathCheckNilTrue, e)

	case map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(*v, fastpathCheckNilTrue, e)

	case map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(v, fastpathCheckNilTrue, e)
	case *map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(v, fastpathCheckNilTrue, e)
	case *map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(*v, fastpathCheckNilTrue, e)

	case map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(v, fastpathCheckNilTrue, e)
	case *map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(*v, fastpathCheckNilTrue, e)

	case map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(v, fastpathCheckNilTrue, e)
	case *map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(*v, fastpathCheckNilTrue, e)

	case map[int]interface{}:
		fastpathTV.EncMapIntIntfV(v, fastpathCheckNilTrue, e)
	case *map[int]interface{}:
		fastpathTV.EncMapIntIntfV(*v, fastpathCheckNilTrue, e)

	case map[int]string:
		fastpathTV.EncMapIntStringV(v, fastpathCheckNilTrue, e)
	case *map[int]string:
		fastpathTV.EncMapIntStringV(*v, fastpathCheckNilTrue, e)

	case map[int]uint:
		fastpathTV.EncMapIntUintV(v, fastpathCheckNilTrue, e)
	case *map[int]uint:
		fastpathTV.EncMapIntUintV(*v, fastpathCheckNilTrue, e)

	case map[int]uint8:
		fastpathTV.EncMapIntUint8V(v, fastpathCheckNilTrue, e)
	case *map[int]uint8:
		fastpathTV.EncMapIntUint8V(*v, fastpathCheckNilTrue, e)

	case map[int]uint16:
		fastpathTV.EncMapIntUint16V(v, fastpathCheckNilTrue, e)
	case *map[int]uint16:
		fastpathTV.EncMapIntUint16V(*v, fastpathCheckNilTrue, e)

	case map[int]uint32:
		fastpathTV.EncMapIntUint32V(v, fastpathCheckNilTrue, e)
	case *map[int]uint32:
		fastpathTV.EncMapIntUint32V(*v, fastpathCheckNilTrue, e)

	case map[int]uint64:
		fastpathTV.EncMapIntUint64V(v, fastpathCheckNilTrue, e)
	case *map[int]uint64:
		fastpathTV.EncMapIntUint64V(*v, fastpathCheckNilTrue, e)

	case map[int]int:
		fastpathTV.EncMapIntIntV(v, fastpathCheckNilTrue, e)
	case *map[int]int:
		fastpathTV.EncMapIntIntV(*v, fastpathCheckNilTrue, e)

	case map[int]int8:
		fastpathTV.EncMapIntInt8V(v, fastpathCheckNilTrue, e)
	case *map[int]int8:
		fastpathTV.EncMapIntInt8V(*v, fastpathCheckNilTrue, e)

	case map[int]int16:
		fastpathTV.EncMapIntInt16V(v, fastpathCheckNilTrue, e)
	case *map[int]int16:
		fastpathTV.EncMapIntInt16V(*v, fastpathCheckNilTrue, e)

	case map[int]int32:
		fastpathTV.EncMapIntInt32V(v, fastpathCheckNilTrue, e)
	case *map[int]int32:
		fastpathTV.EncMapIntInt32V(*v, fastpathCheckNilTrue, e)

	case map[int]int64:
		fastpathTV.EncMapIntInt64V(v, fastpathCheckNilTrue, e)
	case *map[int]int64:
		fastpathTV.EncMapIntInt64V(*v, fastpathCheckNilTrue, e)

	case map[int]float32:
		fastpathTV.EncMapIntFloat32V(v, fastpathCheckNilTrue, e)
	case *map[int]float32:
		fastpathTV.EncMapIntFloat32V(*v, fastpathCheckNilTrue, e)

	case map[int]float64:
		fastpathTV.EncMapIntFloat64V(v, fastpathCheckNilTrue, e)
	case *map[int]float64:
		fastpathTV.EncMapIntFloat64V(*v, fastpathCheckNilTrue, e)

	case map[int]bool:
		fastpathTV.EncMapIntBoolV(v, fastpathCheckNilTrue, e)
	case *map[int]bool:
		fastpathTV.EncMapIntBoolV(*v, fastpathCheckNilTrue, e)

	case map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(v, fastpathCheckNilTrue, e)
	case *map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(*v, fastpathCheckNilTrue, e)

	case map[int8]string:
		fastpathTV.EncMapInt8StringV(v, fastpathCheckNilTrue, e)
	case *map[int8]string:
		fastpathTV.EncMapInt8StringV(*v, fastpathCheckNilTrue, e)

	case map[int8]uint:
		fastpathTV.EncMapInt8UintV(v, fastpathCheckNilTrue, e)
	case *map[int8]uint:
		fastpathTV.EncMapInt8UintV(*v, fastpathCheckNilTrue, e)

	case map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int8]int:
		fastpathTV.EncMapInt8IntV(v, fastpathCheckNilTrue, e)
	case *map[int8]int:
		fastpathTV.EncMapInt8IntV(*v, fastpathCheckNilTrue, e)

	case map[int8]int8:
		fastpathTV.EncMapInt8Int8V(v, fastpathCheckNilTrue, e)
	case *map[int8]int8:
		fastpathTV.EncMapInt8Int8V(*v, fastpathCheckNilTrue, e)

	case map[int8]int16:
		fastpathTV.EncMapInt8Int16V(v, fastpathCheckNilTrue, e)
	case *map[int8]int16:
		fastpathTV.EncMapInt8Int16V(*v, fastpathCheckNilTrue, e)

	case map[int8]int32:
		fastpathTV.EncMapInt8Int32V(v, fastpathCheckNilTrue, e)
	case *map[int8]int32:
		fastpathTV.EncMapInt8Int32V(*v, fastpathCheckNilTrue, e)

	case map[int8]int64:
		fastpathTV.EncMapInt8Int64V(v, fastpathCheckNilTrue, e)
	case *map[int8]int64:
		fastpathTV.EncMapInt8Int64V(*v, fastpathCheckNilTrue, e)

	case map[int8]float32:
		fastpathTV.EncMapInt8Float32V(v, fastpathCheckNilTrue, e)
	case *map[int8]float32:
		fastpathTV.EncMapInt8Float32V(*v, fastpathCheckNilTrue, e)

	case map[int8]float64:
		fastpathTV.EncMapInt8Float64V(v, fastpathCheckNilTrue, e)
	case *map[int8]float64:
		fastpathTV.EncMapInt8Float64V(*v, fastpathCheckNilTrue, e)

	case map[int8]bool:
		fastpathTV.EncMapInt8BoolV(v, fastpathCheckNilTrue, e)
	case *map[int8]bool:
		fastpathTV.EncMapInt8BoolV(*v, fastpathCheckNilTrue, e)

	case map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(v, fastpathCheckNilTrue, e)
	case *map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(*v, fastpathCheckNilTrue, e)

	case map[int16]string:
		fastpathTV.EncMapInt16StringV(v, fastpathCheckNilTrue, e)
	case *map[int16]string:
		fastpathTV.EncMapInt16StringV(*v, fastpathCheckNilTrue, e)

	case map[int16]uint:
		fastpathTV.EncMapInt16UintV(v, fastpathCheckNilTrue, e)
	case *map[int16]uint:
		fastpathTV.EncMapInt16UintV(*v, fastpathCheckNilTrue, e)

	case map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int16]int:
		fastpathTV.EncMapInt16IntV(v, fastpathCheckNilTrue, e)
	case *map[int16]int:
		fastpathTV.EncMapInt16IntV(*v, fastpathCheckNilTrue, e)

	case map[int16]int8:
		fastpathTV.EncMapInt16Int8V(v, fastpathCheckNilTrue, e)
	case *map[int16]int8:
		fastpathTV.EncMapInt16Int8V(*v, fastpathCheckNilTrue, e)

	case map[int16]int16:
		fastpathTV.EncMapInt16Int16V(v, fastpathCheckNilTrue, e)
	case *map[int16]int16:
		fastpathTV.EncMapInt16Int16V(*v, fastpathCheckNilTrue, e)

	case map[int16]int32:
		fastpathTV.EncMapInt16Int32V(v, fastpathCheckNilTrue, e)
	case *map[int16]int32:
		fastpathTV.EncMapInt16Int32V(*v, fastpathCheckNilTrue, e)

	case map[int16]int64:
		fastpathTV.EncMapInt16Int64V(v, fastpathCheckNilTrue, e)
	case *map[int16]int64:
		fastpathTV.EncMapInt16Int64V(*v, fastpathCheckNilTrue, e)

	case map[int16]float32:
		fastpathTV.EncMapInt16Float32V(v, fastpathCheckNilTrue, e)
	case *map[int16]float32:
		fastpathTV.EncMapInt16Float32V(*v, fastpathCheckNilTrue, e)

	case map[int16]float64:
		fastpathTV.EncMapInt16Float64V(v, fastpathCheckNilTrue, e)
	case *map[int16]float64:
		fastpathTV.EncMapInt16Float64V(*v, fastpathCheckNilTrue, e)

	case map[int16]bool:
		fastpathTV.EncMapInt16BoolV(v, fastpathCheckNilTrue, e)
	case *map[int16]bool:
		fastpathTV.EncMapInt16BoolV(*v, fastpathCheckNilTrue, e)

	case map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(v, fastpathCheckNilTrue, e)
	case *map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(*v, fastpathCheckNilTrue, e)

	case map[int32]string:
		fastpathTV.EncMapInt32StringV(v, fastpathCheckNilTrue, e)
	case *map[int32]string:
		fastpathTV.EncMapInt32StringV(*v, fastpathCheckNilTrue, e)

	case map[int32]uint:
		fastpathTV.EncMapInt32UintV(v, fastpathCheckNilTrue, e)
	case *map[int32]uint:
		fastpathTV.EncMapInt32UintV(*v, fastpathCheckNilTrue, e)

	case map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int32]int:
		fastpathTV.EncMapInt32IntV(v, fastpathCheckNilTrue, e)
	case *map[int32]int:
		fastpathTV.EncMapInt32IntV(*v, fastpathCheckNilTrue, e)

	case map[int32]int8:
		fastpathTV.EncMapInt32Int8V(v, fastpathCheckNilTrue, e)
	case *map[int32]int8:
		fastpathTV.EncMapInt32Int8V(*v, fastpathCheckNilTrue, e)

	case map[int32]int16:
		fastpathTV.EncMapInt32Int16V(v, fastpathCheckNilTrue, e)
	case *map[int32]int16:
		fastpathTV.EncMapInt32Int16V(*v, fastpathCheckNilTrue, e)

	case map[int32]int32:
		fastpathTV.EncMapInt32Int32V(v, fastpathCheckNilTrue, e)
	case *map[int32]int32:
		fastpathTV.EncMapInt32Int32V(*v, fastpathCheckNilTrue, e)

	case map[int32]int64:
		fastpathTV.EncMapInt32Int64V(v, fastpathCheckNilTrue, e)
	case *map[int32]int64:
		fastpathTV.EncMapInt32Int64V(*v, fastpathCheckNilTrue, e)

	case map[int32]float32:
		fastpathTV.EncMapInt32Float32V(v, fastpathCheckNilTrue, e)
	case *map[int32]float32:
		fastpathTV.EncMapInt32Float32V(*v, fastpathCheckNilTrue, e)

	case map[int32]float64:
		fastpathTV.EncMapInt32Float64V(v, fastpathCheckNilTrue, e)
	case *map[int32]float64:
		fastpathTV.EncMapInt32Float64V(*v, fastpathCheckNilTrue, e)

	case map[int32]bool:
		fastpathTV.EncMapInt32BoolV(v, fastpathCheckNilTrue, e)
	case *map[int32]bool:
		fastpathTV.EncMapInt32BoolV(*v, fastpathCheckNilTrue, e)

	case map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(v, fastpathCheckNilTrue, e)
	case *map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(*v, fastpathCheckNilTrue, e)

	case map[int64]string:
		fastpathTV.EncMapInt64StringV(v, fastpathCheckNilTrue, e)
	case *map[int64]string:
		fastpathTV.EncMapInt64StringV(*v, fastpathCheckNilTrue, e)

	case map[int64]uint:
		fastpathTV.EncMapInt64UintV(v, fastpathCheckNilTrue, e)
	case *map[int64]uint:
		fastpathTV.EncMapInt64UintV(*v, fastpathCheckNilTrue, e)

	case map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(*v, fastpathCheckNilTrue, e)

	case map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(*v, fastpathCheckNilTrue, e)

	case map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(*v, fastpathCheckNilTrue, e)

	case map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(v, fastpathCheckNilTrue, e)
	case *map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(*v, fastpathCheckNilTrue, e)

	case map[int64]int:
		fastpathTV.EncMapInt64IntV(v, fastpathCheckNilTrue, e)
	case *map[int64]int:
		fastpathTV.EncMapInt64IntV(*v, fastpathCheckNilTrue, e)

	case map[int64]int8:
		fastpathTV.EncMapInt64Int8V(v, fastpathCheckNilTrue, e)
	case *map[int64]int8:
		fastpathTV.EncMapInt64Int8V(*v, fastpathCheckNilTrue, e)

	case map[int64]int16:
		fastpathTV.EncMapInt64Int16V(v, fastpathCheckNilTrue, e)
	case *map[int64]int16:
		fastpathTV.EncMapInt64Int16V(*v, fastpathCheckNilTrue, e)

	case map[int64]int32:
		fastpathTV.EncMapInt64Int32V(v, fastpathCheckNilTrue, e)
	case *map[int64]int32:
		fastpathTV.EncMapInt64Int32V(*v, fastpathCheckNilTrue, e)

	case map[int64]int64:
		fastpathTV.EncMapInt64Int64V(v, fastpathCheckNilTrue, e)
	case *map[int64]int64:
		fastpathTV.EncMapInt64Int64V(*v, fastpathCheckNilTrue, e)

	case map[int64]float32:
		fastpathTV.EncMapInt64Float32V(v, fastpathCheckNilTrue, e)
	case *map[int64]float32:
		fastpathTV.EncMapInt64Float32V(*v, fastpathCheckNilTrue, e)

	case map[int64]float64:
		fastpathTV.EncMapInt64Float64V(v, fastpathCheckNilTrue, e)
	case *map[int64]float64:
		fastpathTV.EncMapInt64Float64V(*v, fastpathCheckNilTrue, e)

	case map[int64]bool:
		fastpathTV.EncMapInt64BoolV(v, fastpathCheckNilTrue, e)
	case *map[int64]bool:
		fastpathTV.EncMapInt64BoolV(*v, fastpathCheckNilTrue, e)

	case map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(v, fastpathCheckNilTrue, e)
	case *map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(*v, fastpathCheckNilTrue, e)

	case map[bool]string:
		fastpathTV.EncMapBoolStringV(v, fastpathCheckNilTrue, e)
	case *map[bool]string:
		fastpathTV.EncMapBoolStringV(*v, fastpathCheckNilTrue, e)

	case map[bool]uint:
		fastpathTV.EncMapBoolUintV(v, fastpathCheckNilTrue, e)
	case *map[bool]uint:
		fastpathTV.EncMapBoolUintV(*v, fastpathCheckNilTrue, e)

	case map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(*v, fastpathCheckNilTrue, e)

	case map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(*v, fastpathCheckNilTrue, e)

	case map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(*v, fastpathCheckNilTrue, e)

	case map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(v, fastpathCheckNilTrue, e)
	case *map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(*v, fastpathCheckNilTrue, e)

	case map[bool]int:
		fastpathTV.EncMapBoolIntV(v, fastpathCheckNilTrue, e)
	case *map[bool]int:
		fastpathTV.EncMapBoolIntV(*v, fastpathCheckNilTrue, e)

	case map[bool]int8:
		fastpathTV.EncMapBoolInt8V(v, fastpathCheckNilTrue, e)
	case *map[bool]int8:
		fastpathTV.EncMapBoolInt8V(*v, fastpathCheckNilTrue, e)

	case map[bool]int16:
		fastpathTV.EncMapBoolInt16V(v, fastpathCheckNilTrue, e)
	case *map[bool]int16:
		fastpathTV.EncMapBoolInt16V(*v, fastpathCheckNilTrue, e)

	case map[bool]int32:
		fastpathTV.EncMapBoolInt32V(v, fastpathCheckNilTrue, e)
	case *map[bool]int32:
		fastpathTV.EncMapBoolInt32V(*v, fastpathCheckNilTrue, e)

	case map[bool]int64:
		fastpathTV.EncMapBoolInt64V(v, fastpathCheckNilTrue, e)
	case *map[bool]int64:
		fastpathTV.EncMapBoolInt64V(*v, fastpathCheckNilTrue, e)

	case map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(v, fastpathCheckNilTrue, e)
	case *map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(*v, fastpathCheckNilTrue, e)

	case map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(v, fastpathCheckNilTrue, e)
	case *map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(*v, fastpathCheckNilTrue, e)

	case map[bool]bool:
		fastpathTV.EncMapBoolBoolV(v, fastpathCheckNilTrue, e)
	case *map[bool]bool:
		fastpathTV.EncMapBoolBoolV(*v, fastpathCheckNilTrue, e)

	default:
		return false
	}
	return true
}

// -- -- fast path functions

func (f *encFnInfo) fastpathEncSliceIntfR(rv reflect.Value) {
	fastpathTV.EncSliceIntfV(rv.Interface().([]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceIntfV(v []interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceStringR(rv reflect.Value) {
	fastpathTV.EncSliceStringV(rv.Interface().([]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceStringV(v []string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceFloat32R(rv reflect.Value) {
	fastpathTV.EncSliceFloat32V(rv.Interface().([]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceFloat32V(v []float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceFloat64R(rv reflect.Value) {
	fastpathTV.EncSliceFloat64V(rv.Interface().([]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceFloat64V(v []float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceUintR(rv reflect.Value) {
	fastpathTV.EncSliceUintV(rv.Interface().([]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceUintV(v []uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceUint16R(rv reflect.Value) {
	fastpathTV.EncSliceUint16V(rv.Interface().([]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceUint16V(v []uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceUint32R(rv reflect.Value) {
	fastpathTV.EncSliceUint32V(rv.Interface().([]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceUint32V(v []uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceUint64R(rv reflect.Value) {
	fastpathTV.EncSliceUint64V(rv.Interface().([]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceUint64V(v []uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceIntR(rv reflect.Value) {
	fastpathTV.EncSliceIntV(rv.Interface().([]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceIntV(v []int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceInt8R(rv reflect.Value) {
	fastpathTV.EncSliceInt8V(rv.Interface().([]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceInt8V(v []int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceInt16R(rv reflect.Value) {
	fastpathTV.EncSliceInt16V(rv.Interface().([]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceInt16V(v []int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceInt32R(rv reflect.Value) {
	fastpathTV.EncSliceInt32V(rv.Interface().([]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceInt32V(v []int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceInt64R(rv reflect.Value) {
	fastpathTV.EncSliceInt64V(rv.Interface().([]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceInt64V(v []int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncSliceBoolR(rv reflect.Value) {
	fastpathTV.EncSliceBoolV(rv.Interface().([]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncSliceBoolV(v []bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeArrayStart(len(v))
	for _, v2 := range v {
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfIntfR(rv reflect.Value) {
	fastpathTV.EncMapIntfIntfV(rv.Interface().(map[interface{}]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfIntfV(v map[interface{}]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfStringR(rv reflect.Value) {
	fastpathTV.EncMapIntfStringV(rv.Interface().(map[interface{}]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfStringV(v map[interface{}]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfUintR(rv reflect.Value) {
	fastpathTV.EncMapIntfUintV(rv.Interface().(map[interface{}]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfUintV(v map[interface{}]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfUint8R(rv reflect.Value) {
	fastpathTV.EncMapIntfUint8V(rv.Interface().(map[interface{}]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfUint8V(v map[interface{}]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfUint16R(rv reflect.Value) {
	fastpathTV.EncMapIntfUint16V(rv.Interface().(map[interface{}]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfUint16V(v map[interface{}]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfUint32R(rv reflect.Value) {
	fastpathTV.EncMapIntfUint32V(rv.Interface().(map[interface{}]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfUint32V(v map[interface{}]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfUint64R(rv reflect.Value) {
	fastpathTV.EncMapIntfUint64V(rv.Interface().(map[interface{}]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfUint64V(v map[interface{}]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfIntR(rv reflect.Value) {
	fastpathTV.EncMapIntfIntV(rv.Interface().(map[interface{}]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfIntV(v map[interface{}]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfInt8R(rv reflect.Value) {
	fastpathTV.EncMapIntfInt8V(rv.Interface().(map[interface{}]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfInt8V(v map[interface{}]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfInt16R(rv reflect.Value) {
	fastpathTV.EncMapIntfInt16V(rv.Interface().(map[interface{}]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfInt16V(v map[interface{}]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfInt32R(rv reflect.Value) {
	fastpathTV.EncMapIntfInt32V(rv.Interface().(map[interface{}]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfInt32V(v map[interface{}]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfInt64R(rv reflect.Value) {
	fastpathTV.EncMapIntfInt64V(rv.Interface().(map[interface{}]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfInt64V(v map[interface{}]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfFloat32R(rv reflect.Value) {
	fastpathTV.EncMapIntfFloat32V(rv.Interface().(map[interface{}]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfFloat32V(v map[interface{}]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfFloat64R(rv reflect.Value) {
	fastpathTV.EncMapIntfFloat64V(rv.Interface().(map[interface{}]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfFloat64V(v map[interface{}]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntfBoolR(rv reflect.Value) {
	fastpathTV.EncMapIntfBoolV(rv.Interface().(map[interface{}]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntfBoolV(v map[interface{}]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		e.encode(k2)
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringIntfR(rv reflect.Value) {
	fastpathTV.EncMapStringIntfV(rv.Interface().(map[string]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringIntfV(v map[string]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringStringR(rv reflect.Value) {
	fastpathTV.EncMapStringStringV(rv.Interface().(map[string]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringStringV(v map[string]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringUintR(rv reflect.Value) {
	fastpathTV.EncMapStringUintV(rv.Interface().(map[string]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringUintV(v map[string]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringUint8R(rv reflect.Value) {
	fastpathTV.EncMapStringUint8V(rv.Interface().(map[string]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringUint8V(v map[string]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringUint16R(rv reflect.Value) {
	fastpathTV.EncMapStringUint16V(rv.Interface().(map[string]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringUint16V(v map[string]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringUint32R(rv reflect.Value) {
	fastpathTV.EncMapStringUint32V(rv.Interface().(map[string]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringUint32V(v map[string]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringUint64R(rv reflect.Value) {
	fastpathTV.EncMapStringUint64V(rv.Interface().(map[string]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringUint64V(v map[string]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringIntR(rv reflect.Value) {
	fastpathTV.EncMapStringIntV(rv.Interface().(map[string]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringIntV(v map[string]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringInt8R(rv reflect.Value) {
	fastpathTV.EncMapStringInt8V(rv.Interface().(map[string]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringInt8V(v map[string]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringInt16R(rv reflect.Value) {
	fastpathTV.EncMapStringInt16V(rv.Interface().(map[string]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringInt16V(v map[string]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringInt32R(rv reflect.Value) {
	fastpathTV.EncMapStringInt32V(rv.Interface().(map[string]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringInt32V(v map[string]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringInt64R(rv reflect.Value) {
	fastpathTV.EncMapStringInt64V(rv.Interface().(map[string]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringInt64V(v map[string]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringFloat32R(rv reflect.Value) {
	fastpathTV.EncMapStringFloat32V(rv.Interface().(map[string]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringFloat32V(v map[string]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringFloat64R(rv reflect.Value) {
	fastpathTV.EncMapStringFloat64V(rv.Interface().(map[string]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringFloat64V(v map[string]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapStringBoolR(rv reflect.Value) {
	fastpathTV.EncMapStringBoolV(rv.Interface().(map[string]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapStringBoolV(v map[string]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			ee.EncodeSymbol(k2)
		} else {
			ee.EncodeString(c_UTF8, k2)
		}
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32IntfR(rv reflect.Value) {
	fastpathTV.EncMapFloat32IntfV(rv.Interface().(map[float32]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32IntfV(v map[float32]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32StringR(rv reflect.Value) {
	fastpathTV.EncMapFloat32StringV(rv.Interface().(map[float32]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32StringV(v map[float32]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32UintR(rv reflect.Value) {
	fastpathTV.EncMapFloat32UintV(rv.Interface().(map[float32]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32UintV(v map[float32]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Uint8R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint8V(rv.Interface().(map[float32]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Uint8V(v map[float32]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Uint16R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint16V(rv.Interface().(map[float32]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Uint16V(v map[float32]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Uint32R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint32V(rv.Interface().(map[float32]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Uint32V(v map[float32]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Uint64R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint64V(rv.Interface().(map[float32]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Uint64V(v map[float32]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32IntR(rv reflect.Value) {
	fastpathTV.EncMapFloat32IntV(rv.Interface().(map[float32]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32IntV(v map[float32]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Int8R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Int8V(rv.Interface().(map[float32]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Int8V(v map[float32]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Int16R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Int16V(rv.Interface().(map[float32]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Int16V(v map[float32]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Int32R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Int32V(rv.Interface().(map[float32]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Int32V(v map[float32]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Int64R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Int64V(rv.Interface().(map[float32]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Int64V(v map[float32]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Float32R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Float32V(rv.Interface().(map[float32]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Float32V(v map[float32]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32Float64R(rv reflect.Value) {
	fastpathTV.EncMapFloat32Float64V(rv.Interface().(map[float32]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32Float64V(v map[float32]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat32BoolR(rv reflect.Value) {
	fastpathTV.EncMapFloat32BoolV(rv.Interface().(map[float32]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat32BoolV(v map[float32]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat32(k2)
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64IntfR(rv reflect.Value) {
	fastpathTV.EncMapFloat64IntfV(rv.Interface().(map[float64]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64IntfV(v map[float64]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64StringR(rv reflect.Value) {
	fastpathTV.EncMapFloat64StringV(rv.Interface().(map[float64]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64StringV(v map[float64]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64UintR(rv reflect.Value) {
	fastpathTV.EncMapFloat64UintV(rv.Interface().(map[float64]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64UintV(v map[float64]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Uint8R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint8V(rv.Interface().(map[float64]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Uint8V(v map[float64]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Uint16R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint16V(rv.Interface().(map[float64]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Uint16V(v map[float64]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Uint32R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint32V(rv.Interface().(map[float64]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Uint32V(v map[float64]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Uint64R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint64V(rv.Interface().(map[float64]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Uint64V(v map[float64]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64IntR(rv reflect.Value) {
	fastpathTV.EncMapFloat64IntV(rv.Interface().(map[float64]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64IntV(v map[float64]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Int8R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Int8V(rv.Interface().(map[float64]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Int8V(v map[float64]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Int16R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Int16V(rv.Interface().(map[float64]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Int16V(v map[float64]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Int32R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Int32V(rv.Interface().(map[float64]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Int32V(v map[float64]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Int64R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Int64V(rv.Interface().(map[float64]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Int64V(v map[float64]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Float32R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Float32V(rv.Interface().(map[float64]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Float32V(v map[float64]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64Float64R(rv reflect.Value) {
	fastpathTV.EncMapFloat64Float64V(rv.Interface().(map[float64]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64Float64V(v map[float64]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapFloat64BoolR(rv reflect.Value) {
	fastpathTV.EncMapFloat64BoolV(rv.Interface().(map[float64]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapFloat64BoolV(v map[float64]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeFloat64(k2)
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintIntfR(rv reflect.Value) {
	fastpathTV.EncMapUintIntfV(rv.Interface().(map[uint]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintIntfV(v map[uint]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintStringR(rv reflect.Value) {
	fastpathTV.EncMapUintStringV(rv.Interface().(map[uint]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintStringV(v map[uint]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintUintR(rv reflect.Value) {
	fastpathTV.EncMapUintUintV(rv.Interface().(map[uint]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintUintV(v map[uint]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintUint8R(rv reflect.Value) {
	fastpathTV.EncMapUintUint8V(rv.Interface().(map[uint]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintUint8V(v map[uint]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintUint16R(rv reflect.Value) {
	fastpathTV.EncMapUintUint16V(rv.Interface().(map[uint]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintUint16V(v map[uint]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintUint32R(rv reflect.Value) {
	fastpathTV.EncMapUintUint32V(rv.Interface().(map[uint]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintUint32V(v map[uint]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintUint64R(rv reflect.Value) {
	fastpathTV.EncMapUintUint64V(rv.Interface().(map[uint]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintUint64V(v map[uint]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintIntR(rv reflect.Value) {
	fastpathTV.EncMapUintIntV(rv.Interface().(map[uint]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintIntV(v map[uint]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintInt8R(rv reflect.Value) {
	fastpathTV.EncMapUintInt8V(rv.Interface().(map[uint]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintInt8V(v map[uint]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintInt16R(rv reflect.Value) {
	fastpathTV.EncMapUintInt16V(rv.Interface().(map[uint]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintInt16V(v map[uint]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintInt32R(rv reflect.Value) {
	fastpathTV.EncMapUintInt32V(rv.Interface().(map[uint]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintInt32V(v map[uint]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintInt64R(rv reflect.Value) {
	fastpathTV.EncMapUintInt64V(rv.Interface().(map[uint]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintInt64V(v map[uint]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintFloat32R(rv reflect.Value) {
	fastpathTV.EncMapUintFloat32V(rv.Interface().(map[uint]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintFloat32V(v map[uint]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintFloat64R(rv reflect.Value) {
	fastpathTV.EncMapUintFloat64V(rv.Interface().(map[uint]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintFloat64V(v map[uint]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUintBoolR(rv reflect.Value) {
	fastpathTV.EncMapUintBoolV(rv.Interface().(map[uint]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUintBoolV(v map[uint]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8IntfR(rv reflect.Value) {
	fastpathTV.EncMapUint8IntfV(rv.Interface().(map[uint8]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8IntfV(v map[uint8]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8StringR(rv reflect.Value) {
	fastpathTV.EncMapUint8StringV(rv.Interface().(map[uint8]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8StringV(v map[uint8]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8UintR(rv reflect.Value) {
	fastpathTV.EncMapUint8UintV(rv.Interface().(map[uint8]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8UintV(v map[uint8]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Uint8R(rv reflect.Value) {
	fastpathTV.EncMapUint8Uint8V(rv.Interface().(map[uint8]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Uint8V(v map[uint8]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Uint16R(rv reflect.Value) {
	fastpathTV.EncMapUint8Uint16V(rv.Interface().(map[uint8]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Uint16V(v map[uint8]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Uint32R(rv reflect.Value) {
	fastpathTV.EncMapUint8Uint32V(rv.Interface().(map[uint8]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Uint32V(v map[uint8]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Uint64R(rv reflect.Value) {
	fastpathTV.EncMapUint8Uint64V(rv.Interface().(map[uint8]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Uint64V(v map[uint8]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8IntR(rv reflect.Value) {
	fastpathTV.EncMapUint8IntV(rv.Interface().(map[uint8]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8IntV(v map[uint8]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Int8R(rv reflect.Value) {
	fastpathTV.EncMapUint8Int8V(rv.Interface().(map[uint8]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Int8V(v map[uint8]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Int16R(rv reflect.Value) {
	fastpathTV.EncMapUint8Int16V(rv.Interface().(map[uint8]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Int16V(v map[uint8]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Int32R(rv reflect.Value) {
	fastpathTV.EncMapUint8Int32V(rv.Interface().(map[uint8]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Int32V(v map[uint8]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Int64R(rv reflect.Value) {
	fastpathTV.EncMapUint8Int64V(rv.Interface().(map[uint8]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Int64V(v map[uint8]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Float32R(rv reflect.Value) {
	fastpathTV.EncMapUint8Float32V(rv.Interface().(map[uint8]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Float32V(v map[uint8]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8Float64R(rv reflect.Value) {
	fastpathTV.EncMapUint8Float64V(rv.Interface().(map[uint8]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8Float64V(v map[uint8]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint8BoolR(rv reflect.Value) {
	fastpathTV.EncMapUint8BoolV(rv.Interface().(map[uint8]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint8BoolV(v map[uint8]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16IntfR(rv reflect.Value) {
	fastpathTV.EncMapUint16IntfV(rv.Interface().(map[uint16]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16IntfV(v map[uint16]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16StringR(rv reflect.Value) {
	fastpathTV.EncMapUint16StringV(rv.Interface().(map[uint16]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16StringV(v map[uint16]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16UintR(rv reflect.Value) {
	fastpathTV.EncMapUint16UintV(rv.Interface().(map[uint16]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16UintV(v map[uint16]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Uint8R(rv reflect.Value) {
	fastpathTV.EncMapUint16Uint8V(rv.Interface().(map[uint16]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Uint8V(v map[uint16]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Uint16R(rv reflect.Value) {
	fastpathTV.EncMapUint16Uint16V(rv.Interface().(map[uint16]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Uint16V(v map[uint16]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Uint32R(rv reflect.Value) {
	fastpathTV.EncMapUint16Uint32V(rv.Interface().(map[uint16]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Uint32V(v map[uint16]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Uint64R(rv reflect.Value) {
	fastpathTV.EncMapUint16Uint64V(rv.Interface().(map[uint16]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Uint64V(v map[uint16]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16IntR(rv reflect.Value) {
	fastpathTV.EncMapUint16IntV(rv.Interface().(map[uint16]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16IntV(v map[uint16]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Int8R(rv reflect.Value) {
	fastpathTV.EncMapUint16Int8V(rv.Interface().(map[uint16]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Int8V(v map[uint16]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Int16R(rv reflect.Value) {
	fastpathTV.EncMapUint16Int16V(rv.Interface().(map[uint16]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Int16V(v map[uint16]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Int32R(rv reflect.Value) {
	fastpathTV.EncMapUint16Int32V(rv.Interface().(map[uint16]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Int32V(v map[uint16]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Int64R(rv reflect.Value) {
	fastpathTV.EncMapUint16Int64V(rv.Interface().(map[uint16]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Int64V(v map[uint16]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Float32R(rv reflect.Value) {
	fastpathTV.EncMapUint16Float32V(rv.Interface().(map[uint16]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Float32V(v map[uint16]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16Float64R(rv reflect.Value) {
	fastpathTV.EncMapUint16Float64V(rv.Interface().(map[uint16]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16Float64V(v map[uint16]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint16BoolR(rv reflect.Value) {
	fastpathTV.EncMapUint16BoolV(rv.Interface().(map[uint16]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint16BoolV(v map[uint16]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32IntfR(rv reflect.Value) {
	fastpathTV.EncMapUint32IntfV(rv.Interface().(map[uint32]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32IntfV(v map[uint32]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32StringR(rv reflect.Value) {
	fastpathTV.EncMapUint32StringV(rv.Interface().(map[uint32]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32StringV(v map[uint32]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32UintR(rv reflect.Value) {
	fastpathTV.EncMapUint32UintV(rv.Interface().(map[uint32]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32UintV(v map[uint32]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Uint8R(rv reflect.Value) {
	fastpathTV.EncMapUint32Uint8V(rv.Interface().(map[uint32]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Uint8V(v map[uint32]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Uint16R(rv reflect.Value) {
	fastpathTV.EncMapUint32Uint16V(rv.Interface().(map[uint32]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Uint16V(v map[uint32]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Uint32R(rv reflect.Value) {
	fastpathTV.EncMapUint32Uint32V(rv.Interface().(map[uint32]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Uint32V(v map[uint32]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Uint64R(rv reflect.Value) {
	fastpathTV.EncMapUint32Uint64V(rv.Interface().(map[uint32]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Uint64V(v map[uint32]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32IntR(rv reflect.Value) {
	fastpathTV.EncMapUint32IntV(rv.Interface().(map[uint32]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32IntV(v map[uint32]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Int8R(rv reflect.Value) {
	fastpathTV.EncMapUint32Int8V(rv.Interface().(map[uint32]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Int8V(v map[uint32]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Int16R(rv reflect.Value) {
	fastpathTV.EncMapUint32Int16V(rv.Interface().(map[uint32]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Int16V(v map[uint32]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Int32R(rv reflect.Value) {
	fastpathTV.EncMapUint32Int32V(rv.Interface().(map[uint32]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Int32V(v map[uint32]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Int64R(rv reflect.Value) {
	fastpathTV.EncMapUint32Int64V(rv.Interface().(map[uint32]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Int64V(v map[uint32]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Float32R(rv reflect.Value) {
	fastpathTV.EncMapUint32Float32V(rv.Interface().(map[uint32]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Float32V(v map[uint32]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32Float64R(rv reflect.Value) {
	fastpathTV.EncMapUint32Float64V(rv.Interface().(map[uint32]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32Float64V(v map[uint32]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint32BoolR(rv reflect.Value) {
	fastpathTV.EncMapUint32BoolV(rv.Interface().(map[uint32]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint32BoolV(v map[uint32]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64IntfR(rv reflect.Value) {
	fastpathTV.EncMapUint64IntfV(rv.Interface().(map[uint64]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64IntfV(v map[uint64]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64StringR(rv reflect.Value) {
	fastpathTV.EncMapUint64StringV(rv.Interface().(map[uint64]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64StringV(v map[uint64]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64UintR(rv reflect.Value) {
	fastpathTV.EncMapUint64UintV(rv.Interface().(map[uint64]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64UintV(v map[uint64]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Uint8R(rv reflect.Value) {
	fastpathTV.EncMapUint64Uint8V(rv.Interface().(map[uint64]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Uint8V(v map[uint64]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Uint16R(rv reflect.Value) {
	fastpathTV.EncMapUint64Uint16V(rv.Interface().(map[uint64]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Uint16V(v map[uint64]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Uint32R(rv reflect.Value) {
	fastpathTV.EncMapUint64Uint32V(rv.Interface().(map[uint64]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Uint32V(v map[uint64]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Uint64R(rv reflect.Value) {
	fastpathTV.EncMapUint64Uint64V(rv.Interface().(map[uint64]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Uint64V(v map[uint64]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64IntR(rv reflect.Value) {
	fastpathTV.EncMapUint64IntV(rv.Interface().(map[uint64]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64IntV(v map[uint64]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Int8R(rv reflect.Value) {
	fastpathTV.EncMapUint64Int8V(rv.Interface().(map[uint64]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Int8V(v map[uint64]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Int16R(rv reflect.Value) {
	fastpathTV.EncMapUint64Int16V(rv.Interface().(map[uint64]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Int16V(v map[uint64]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Int32R(rv reflect.Value) {
	fastpathTV.EncMapUint64Int32V(rv.Interface().(map[uint64]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Int32V(v map[uint64]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Int64R(rv reflect.Value) {
	fastpathTV.EncMapUint64Int64V(rv.Interface().(map[uint64]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Int64V(v map[uint64]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Float32R(rv reflect.Value) {
	fastpathTV.EncMapUint64Float32V(rv.Interface().(map[uint64]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Float32V(v map[uint64]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64Float64R(rv reflect.Value) {
	fastpathTV.EncMapUint64Float64V(rv.Interface().(map[uint64]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64Float64V(v map[uint64]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapUint64BoolR(rv reflect.Value) {
	fastpathTV.EncMapUint64BoolV(rv.Interface().(map[uint64]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapUint64BoolV(v map[uint64]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeUint(uint64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntIntfR(rv reflect.Value) {
	fastpathTV.EncMapIntIntfV(rv.Interface().(map[int]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntIntfV(v map[int]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntStringR(rv reflect.Value) {
	fastpathTV.EncMapIntStringV(rv.Interface().(map[int]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntStringV(v map[int]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntUintR(rv reflect.Value) {
	fastpathTV.EncMapIntUintV(rv.Interface().(map[int]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntUintV(v map[int]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntUint8R(rv reflect.Value) {
	fastpathTV.EncMapIntUint8V(rv.Interface().(map[int]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntUint8V(v map[int]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntUint16R(rv reflect.Value) {
	fastpathTV.EncMapIntUint16V(rv.Interface().(map[int]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntUint16V(v map[int]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntUint32R(rv reflect.Value) {
	fastpathTV.EncMapIntUint32V(rv.Interface().(map[int]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntUint32V(v map[int]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntUint64R(rv reflect.Value) {
	fastpathTV.EncMapIntUint64V(rv.Interface().(map[int]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntUint64V(v map[int]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntIntR(rv reflect.Value) {
	fastpathTV.EncMapIntIntV(rv.Interface().(map[int]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntIntV(v map[int]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntInt8R(rv reflect.Value) {
	fastpathTV.EncMapIntInt8V(rv.Interface().(map[int]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntInt8V(v map[int]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntInt16R(rv reflect.Value) {
	fastpathTV.EncMapIntInt16V(rv.Interface().(map[int]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntInt16V(v map[int]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntInt32R(rv reflect.Value) {
	fastpathTV.EncMapIntInt32V(rv.Interface().(map[int]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntInt32V(v map[int]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntInt64R(rv reflect.Value) {
	fastpathTV.EncMapIntInt64V(rv.Interface().(map[int]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntInt64V(v map[int]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntFloat32R(rv reflect.Value) {
	fastpathTV.EncMapIntFloat32V(rv.Interface().(map[int]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntFloat32V(v map[int]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntFloat64R(rv reflect.Value) {
	fastpathTV.EncMapIntFloat64V(rv.Interface().(map[int]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntFloat64V(v map[int]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapIntBoolR(rv reflect.Value) {
	fastpathTV.EncMapIntBoolV(rv.Interface().(map[int]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapIntBoolV(v map[int]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8IntfR(rv reflect.Value) {
	fastpathTV.EncMapInt8IntfV(rv.Interface().(map[int8]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8IntfV(v map[int8]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8StringR(rv reflect.Value) {
	fastpathTV.EncMapInt8StringV(rv.Interface().(map[int8]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8StringV(v map[int8]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8UintR(rv reflect.Value) {
	fastpathTV.EncMapInt8UintV(rv.Interface().(map[int8]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8UintV(v map[int8]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Uint8R(rv reflect.Value) {
	fastpathTV.EncMapInt8Uint8V(rv.Interface().(map[int8]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Uint8V(v map[int8]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Uint16R(rv reflect.Value) {
	fastpathTV.EncMapInt8Uint16V(rv.Interface().(map[int8]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Uint16V(v map[int8]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Uint32R(rv reflect.Value) {
	fastpathTV.EncMapInt8Uint32V(rv.Interface().(map[int8]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Uint32V(v map[int8]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Uint64R(rv reflect.Value) {
	fastpathTV.EncMapInt8Uint64V(rv.Interface().(map[int8]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Uint64V(v map[int8]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8IntR(rv reflect.Value) {
	fastpathTV.EncMapInt8IntV(rv.Interface().(map[int8]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8IntV(v map[int8]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Int8R(rv reflect.Value) {
	fastpathTV.EncMapInt8Int8V(rv.Interface().(map[int8]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Int8V(v map[int8]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Int16R(rv reflect.Value) {
	fastpathTV.EncMapInt8Int16V(rv.Interface().(map[int8]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Int16V(v map[int8]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Int32R(rv reflect.Value) {
	fastpathTV.EncMapInt8Int32V(rv.Interface().(map[int8]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Int32V(v map[int8]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Int64R(rv reflect.Value) {
	fastpathTV.EncMapInt8Int64V(rv.Interface().(map[int8]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Int64V(v map[int8]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Float32R(rv reflect.Value) {
	fastpathTV.EncMapInt8Float32V(rv.Interface().(map[int8]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Float32V(v map[int8]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8Float64R(rv reflect.Value) {
	fastpathTV.EncMapInt8Float64V(rv.Interface().(map[int8]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8Float64V(v map[int8]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt8BoolR(rv reflect.Value) {
	fastpathTV.EncMapInt8BoolV(rv.Interface().(map[int8]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt8BoolV(v map[int8]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16IntfR(rv reflect.Value) {
	fastpathTV.EncMapInt16IntfV(rv.Interface().(map[int16]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16IntfV(v map[int16]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16StringR(rv reflect.Value) {
	fastpathTV.EncMapInt16StringV(rv.Interface().(map[int16]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16StringV(v map[int16]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16UintR(rv reflect.Value) {
	fastpathTV.EncMapInt16UintV(rv.Interface().(map[int16]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16UintV(v map[int16]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Uint8R(rv reflect.Value) {
	fastpathTV.EncMapInt16Uint8V(rv.Interface().(map[int16]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Uint8V(v map[int16]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Uint16R(rv reflect.Value) {
	fastpathTV.EncMapInt16Uint16V(rv.Interface().(map[int16]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Uint16V(v map[int16]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Uint32R(rv reflect.Value) {
	fastpathTV.EncMapInt16Uint32V(rv.Interface().(map[int16]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Uint32V(v map[int16]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Uint64R(rv reflect.Value) {
	fastpathTV.EncMapInt16Uint64V(rv.Interface().(map[int16]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Uint64V(v map[int16]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16IntR(rv reflect.Value) {
	fastpathTV.EncMapInt16IntV(rv.Interface().(map[int16]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16IntV(v map[int16]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Int8R(rv reflect.Value) {
	fastpathTV.EncMapInt16Int8V(rv.Interface().(map[int16]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Int8V(v map[int16]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Int16R(rv reflect.Value) {
	fastpathTV.EncMapInt16Int16V(rv.Interface().(map[int16]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Int16V(v map[int16]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Int32R(rv reflect.Value) {
	fastpathTV.EncMapInt16Int32V(rv.Interface().(map[int16]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Int32V(v map[int16]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Int64R(rv reflect.Value) {
	fastpathTV.EncMapInt16Int64V(rv.Interface().(map[int16]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Int64V(v map[int16]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Float32R(rv reflect.Value) {
	fastpathTV.EncMapInt16Float32V(rv.Interface().(map[int16]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Float32V(v map[int16]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16Float64R(rv reflect.Value) {
	fastpathTV.EncMapInt16Float64V(rv.Interface().(map[int16]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16Float64V(v map[int16]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt16BoolR(rv reflect.Value) {
	fastpathTV.EncMapInt16BoolV(rv.Interface().(map[int16]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt16BoolV(v map[int16]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32IntfR(rv reflect.Value) {
	fastpathTV.EncMapInt32IntfV(rv.Interface().(map[int32]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32IntfV(v map[int32]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32StringR(rv reflect.Value) {
	fastpathTV.EncMapInt32StringV(rv.Interface().(map[int32]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32StringV(v map[int32]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32UintR(rv reflect.Value) {
	fastpathTV.EncMapInt32UintV(rv.Interface().(map[int32]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32UintV(v map[int32]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Uint8R(rv reflect.Value) {
	fastpathTV.EncMapInt32Uint8V(rv.Interface().(map[int32]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Uint8V(v map[int32]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Uint16R(rv reflect.Value) {
	fastpathTV.EncMapInt32Uint16V(rv.Interface().(map[int32]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Uint16V(v map[int32]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Uint32R(rv reflect.Value) {
	fastpathTV.EncMapInt32Uint32V(rv.Interface().(map[int32]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Uint32V(v map[int32]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Uint64R(rv reflect.Value) {
	fastpathTV.EncMapInt32Uint64V(rv.Interface().(map[int32]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Uint64V(v map[int32]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32IntR(rv reflect.Value) {
	fastpathTV.EncMapInt32IntV(rv.Interface().(map[int32]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32IntV(v map[int32]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Int8R(rv reflect.Value) {
	fastpathTV.EncMapInt32Int8V(rv.Interface().(map[int32]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Int8V(v map[int32]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Int16R(rv reflect.Value) {
	fastpathTV.EncMapInt32Int16V(rv.Interface().(map[int32]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Int16V(v map[int32]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Int32R(rv reflect.Value) {
	fastpathTV.EncMapInt32Int32V(rv.Interface().(map[int32]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Int32V(v map[int32]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Int64R(rv reflect.Value) {
	fastpathTV.EncMapInt32Int64V(rv.Interface().(map[int32]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Int64V(v map[int32]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Float32R(rv reflect.Value) {
	fastpathTV.EncMapInt32Float32V(rv.Interface().(map[int32]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Float32V(v map[int32]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32Float64R(rv reflect.Value) {
	fastpathTV.EncMapInt32Float64V(rv.Interface().(map[int32]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32Float64V(v map[int32]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt32BoolR(rv reflect.Value) {
	fastpathTV.EncMapInt32BoolV(rv.Interface().(map[int32]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt32BoolV(v map[int32]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64IntfR(rv reflect.Value) {
	fastpathTV.EncMapInt64IntfV(rv.Interface().(map[int64]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64IntfV(v map[int64]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64StringR(rv reflect.Value) {
	fastpathTV.EncMapInt64StringV(rv.Interface().(map[int64]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64StringV(v map[int64]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64UintR(rv reflect.Value) {
	fastpathTV.EncMapInt64UintV(rv.Interface().(map[int64]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64UintV(v map[int64]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Uint8R(rv reflect.Value) {
	fastpathTV.EncMapInt64Uint8V(rv.Interface().(map[int64]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Uint8V(v map[int64]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Uint16R(rv reflect.Value) {
	fastpathTV.EncMapInt64Uint16V(rv.Interface().(map[int64]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Uint16V(v map[int64]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Uint32R(rv reflect.Value) {
	fastpathTV.EncMapInt64Uint32V(rv.Interface().(map[int64]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Uint32V(v map[int64]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Uint64R(rv reflect.Value) {
	fastpathTV.EncMapInt64Uint64V(rv.Interface().(map[int64]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Uint64V(v map[int64]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64IntR(rv reflect.Value) {
	fastpathTV.EncMapInt64IntV(rv.Interface().(map[int64]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64IntV(v map[int64]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Int8R(rv reflect.Value) {
	fastpathTV.EncMapInt64Int8V(rv.Interface().(map[int64]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Int8V(v map[int64]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Int16R(rv reflect.Value) {
	fastpathTV.EncMapInt64Int16V(rv.Interface().(map[int64]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Int16V(v map[int64]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Int32R(rv reflect.Value) {
	fastpathTV.EncMapInt64Int32V(rv.Interface().(map[int64]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Int32V(v map[int64]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Int64R(rv reflect.Value) {
	fastpathTV.EncMapInt64Int64V(rv.Interface().(map[int64]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Int64V(v map[int64]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Float32R(rv reflect.Value) {
	fastpathTV.EncMapInt64Float32V(rv.Interface().(map[int64]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Float32V(v map[int64]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64Float64R(rv reflect.Value) {
	fastpathTV.EncMapInt64Float64V(rv.Interface().(map[int64]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64Float64V(v map[int64]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapInt64BoolR(rv reflect.Value) {
	fastpathTV.EncMapInt64BoolV(rv.Interface().(map[int64]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapInt64BoolV(v map[int64]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeInt(int64(k2))
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolIntfR(rv reflect.Value) {
	fastpathTV.EncMapBoolIntfV(rv.Interface().(map[bool]interface{}), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolIntfV(v map[bool]interface{}, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		e.encode(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolStringR(rv reflect.Value) {
	fastpathTV.EncMapBoolStringV(rv.Interface().(map[bool]string), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolStringV(v map[bool]string, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeString(c_UTF8, v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolUintR(rv reflect.Value) {
	fastpathTV.EncMapBoolUintV(rv.Interface().(map[bool]uint), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolUintV(v map[bool]uint, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolUint8R(rv reflect.Value) {
	fastpathTV.EncMapBoolUint8V(rv.Interface().(map[bool]uint8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolUint8V(v map[bool]uint8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolUint16R(rv reflect.Value) {
	fastpathTV.EncMapBoolUint16V(rv.Interface().(map[bool]uint16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolUint16V(v map[bool]uint16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolUint32R(rv reflect.Value) {
	fastpathTV.EncMapBoolUint32V(rv.Interface().(map[bool]uint32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolUint32V(v map[bool]uint32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolUint64R(rv reflect.Value) {
	fastpathTV.EncMapBoolUint64V(rv.Interface().(map[bool]uint64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolUint64V(v map[bool]uint64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeUint(uint64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolIntR(rv reflect.Value) {
	fastpathTV.EncMapBoolIntV(rv.Interface().(map[bool]int), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolIntV(v map[bool]int, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolInt8R(rv reflect.Value) {
	fastpathTV.EncMapBoolInt8V(rv.Interface().(map[bool]int8), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolInt8V(v map[bool]int8, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolInt16R(rv reflect.Value) {
	fastpathTV.EncMapBoolInt16V(rv.Interface().(map[bool]int16), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolInt16V(v map[bool]int16, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolInt32R(rv reflect.Value) {
	fastpathTV.EncMapBoolInt32V(rv.Interface().(map[bool]int32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolInt32V(v map[bool]int32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolInt64R(rv reflect.Value) {
	fastpathTV.EncMapBoolInt64V(rv.Interface().(map[bool]int64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolInt64V(v map[bool]int64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeInt(int64(v2))
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolFloat32R(rv reflect.Value) {
	fastpathTV.EncMapBoolFloat32V(rv.Interface().(map[bool]float32), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolFloat32V(v map[bool]float32, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeFloat32(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolFloat64R(rv reflect.Value) {
	fastpathTV.EncMapBoolFloat64V(rv.Interface().(map[bool]float64), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolFloat64V(v map[bool]float64, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeFloat64(v2)
	}
	ee.EncodeEnd()
}

func (f *encFnInfo) fastpathEncMapBoolBoolR(rv reflect.Value) {
	fastpathTV.EncMapBoolBoolV(rv.Interface().(map[bool]bool), fastpathCheckNilFalse, f.e)
}
func (_ fastpathT) EncMapBoolBoolV(v map[bool]bool, checkNil bool, e *Encoder) {
	ee := e.e
	if checkNil && v == nil {
		ee.EncodeNil()
		return
	}
	ee.EncodeMapStart(len(v))

	for k2, v2 := range v {
		ee.EncodeBool(k2)
		ee.EncodeBool(v2)
	}
	ee.EncodeEnd()
}

// -- decode

// -- -- fast path type switch
func fastpathDecodeTypeSwitch(iv interface{}, d *Decoder) bool {
	switch v := iv.(type) {

	case []interface{}:
		fastpathTV.DecSliceIntfV(v, fastpathCheckNilFalse, false, d)
	case *[]interface{}:
		v2, changed2 := fastpathTV.DecSliceIntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]interface{}:
		fastpathTV.DecMapIntfIntfV(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]interface{}:
		v2, changed2 := fastpathTV.DecMapIntfIntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]string:
		fastpathTV.DecMapIntfStringV(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]string:
		v2, changed2 := fastpathTV.DecMapIntfStringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]uint:
		fastpathTV.DecMapIntfUintV(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]uint:
		v2, changed2 := fastpathTV.DecMapIntfUintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]uint8:
		fastpathTV.DecMapIntfUint8V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]uint8:
		v2, changed2 := fastpathTV.DecMapIntfUint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]uint16:
		fastpathTV.DecMapIntfUint16V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]uint16:
		v2, changed2 := fastpathTV.DecMapIntfUint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]uint32:
		fastpathTV.DecMapIntfUint32V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]uint32:
		v2, changed2 := fastpathTV.DecMapIntfUint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]uint64:
		fastpathTV.DecMapIntfUint64V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]uint64:
		v2, changed2 := fastpathTV.DecMapIntfUint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]int:
		fastpathTV.DecMapIntfIntV(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]int:
		v2, changed2 := fastpathTV.DecMapIntfIntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]int8:
		fastpathTV.DecMapIntfInt8V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]int8:
		v2, changed2 := fastpathTV.DecMapIntfInt8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]int16:
		fastpathTV.DecMapIntfInt16V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]int16:
		v2, changed2 := fastpathTV.DecMapIntfInt16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]int32:
		fastpathTV.DecMapIntfInt32V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]int32:
		v2, changed2 := fastpathTV.DecMapIntfInt32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]int64:
		fastpathTV.DecMapIntfInt64V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]int64:
		v2, changed2 := fastpathTV.DecMapIntfInt64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]float32:
		fastpathTV.DecMapIntfFloat32V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]float32:
		v2, changed2 := fastpathTV.DecMapIntfFloat32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]float64:
		fastpathTV.DecMapIntfFloat64V(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]float64:
		v2, changed2 := fastpathTV.DecMapIntfFloat64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[interface{}]bool:
		fastpathTV.DecMapIntfBoolV(v, fastpathCheckNilFalse, false, d)
	case *map[interface{}]bool:
		v2, changed2 := fastpathTV.DecMapIntfBoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []string:
		fastpathTV.DecSliceStringV(v, fastpathCheckNilFalse, false, d)
	case *[]string:
		v2, changed2 := fastpathTV.DecSliceStringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]interface{}:
		fastpathTV.DecMapStringIntfV(v, fastpathCheckNilFalse, false, d)
	case *map[string]interface{}:
		v2, changed2 := fastpathTV.DecMapStringIntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]string:
		fastpathTV.DecMapStringStringV(v, fastpathCheckNilFalse, false, d)
	case *map[string]string:
		v2, changed2 := fastpathTV.DecMapStringStringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]uint:
		fastpathTV.DecMapStringUintV(v, fastpathCheckNilFalse, false, d)
	case *map[string]uint:
		v2, changed2 := fastpathTV.DecMapStringUintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]uint8:
		fastpathTV.DecMapStringUint8V(v, fastpathCheckNilFalse, false, d)
	case *map[string]uint8:
		v2, changed2 := fastpathTV.DecMapStringUint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]uint16:
		fastpathTV.DecMapStringUint16V(v, fastpathCheckNilFalse, false, d)
	case *map[string]uint16:
		v2, changed2 := fastpathTV.DecMapStringUint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]uint32:
		fastpathTV.DecMapStringUint32V(v, fastpathCheckNilFalse, false, d)
	case *map[string]uint32:
		v2, changed2 := fastpathTV.DecMapStringUint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]uint64:
		fastpathTV.DecMapStringUint64V(v, fastpathCheckNilFalse, false, d)
	case *map[string]uint64:
		v2, changed2 := fastpathTV.DecMapStringUint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]int:
		fastpathTV.DecMapStringIntV(v, fastpathCheckNilFalse, false, d)
	case *map[string]int:
		v2, changed2 := fastpathTV.DecMapStringIntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]int8:
		fastpathTV.DecMapStringInt8V(v, fastpathCheckNilFalse, false, d)
	case *map[string]int8:
		v2, changed2 := fastpathTV.DecMapStringInt8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]int16:
		fastpathTV.DecMapStringInt16V(v, fastpathCheckNilFalse, false, d)
	case *map[string]int16:
		v2, changed2 := fastpathTV.DecMapStringInt16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]int32:
		fastpathTV.DecMapStringInt32V(v, fastpathCheckNilFalse, false, d)
	case *map[string]int32:
		v2, changed2 := fastpathTV.DecMapStringInt32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]int64:
		fastpathTV.DecMapStringInt64V(v, fastpathCheckNilFalse, false, d)
	case *map[string]int64:
		v2, changed2 := fastpathTV.DecMapStringInt64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]float32:
		fastpathTV.DecMapStringFloat32V(v, fastpathCheckNilFalse, false, d)
	case *map[string]float32:
		v2, changed2 := fastpathTV.DecMapStringFloat32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]float64:
		fastpathTV.DecMapStringFloat64V(v, fastpathCheckNilFalse, false, d)
	case *map[string]float64:
		v2, changed2 := fastpathTV.DecMapStringFloat64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[string]bool:
		fastpathTV.DecMapStringBoolV(v, fastpathCheckNilFalse, false, d)
	case *map[string]bool:
		v2, changed2 := fastpathTV.DecMapStringBoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []float32:
		fastpathTV.DecSliceFloat32V(v, fastpathCheckNilFalse, false, d)
	case *[]float32:
		v2, changed2 := fastpathTV.DecSliceFloat32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]interface{}:
		fastpathTV.DecMapFloat32IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[float32]interface{}:
		v2, changed2 := fastpathTV.DecMapFloat32IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]string:
		fastpathTV.DecMapFloat32StringV(v, fastpathCheckNilFalse, false, d)
	case *map[float32]string:
		v2, changed2 := fastpathTV.DecMapFloat32StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]uint:
		fastpathTV.DecMapFloat32UintV(v, fastpathCheckNilFalse, false, d)
	case *map[float32]uint:
		v2, changed2 := fastpathTV.DecMapFloat32UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]uint8:
		fastpathTV.DecMapFloat32Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]uint8:
		v2, changed2 := fastpathTV.DecMapFloat32Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]uint16:
		fastpathTV.DecMapFloat32Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]uint16:
		v2, changed2 := fastpathTV.DecMapFloat32Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]uint32:
		fastpathTV.DecMapFloat32Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]uint32:
		v2, changed2 := fastpathTV.DecMapFloat32Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]uint64:
		fastpathTV.DecMapFloat32Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]uint64:
		v2, changed2 := fastpathTV.DecMapFloat32Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]int:
		fastpathTV.DecMapFloat32IntV(v, fastpathCheckNilFalse, false, d)
	case *map[float32]int:
		v2, changed2 := fastpathTV.DecMapFloat32IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]int8:
		fastpathTV.DecMapFloat32Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]int8:
		v2, changed2 := fastpathTV.DecMapFloat32Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]int16:
		fastpathTV.DecMapFloat32Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]int16:
		v2, changed2 := fastpathTV.DecMapFloat32Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]int32:
		fastpathTV.DecMapFloat32Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]int32:
		v2, changed2 := fastpathTV.DecMapFloat32Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]int64:
		fastpathTV.DecMapFloat32Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]int64:
		v2, changed2 := fastpathTV.DecMapFloat32Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]float32:
		fastpathTV.DecMapFloat32Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]float32:
		v2, changed2 := fastpathTV.DecMapFloat32Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]float64:
		fastpathTV.DecMapFloat32Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[float32]float64:
		v2, changed2 := fastpathTV.DecMapFloat32Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float32]bool:
		fastpathTV.DecMapFloat32BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[float32]bool:
		v2, changed2 := fastpathTV.DecMapFloat32BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []float64:
		fastpathTV.DecSliceFloat64V(v, fastpathCheckNilFalse, false, d)
	case *[]float64:
		v2, changed2 := fastpathTV.DecSliceFloat64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]interface{}:
		fastpathTV.DecMapFloat64IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[float64]interface{}:
		v2, changed2 := fastpathTV.DecMapFloat64IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]string:
		fastpathTV.DecMapFloat64StringV(v, fastpathCheckNilFalse, false, d)
	case *map[float64]string:
		v2, changed2 := fastpathTV.DecMapFloat64StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]uint:
		fastpathTV.DecMapFloat64UintV(v, fastpathCheckNilFalse, false, d)
	case *map[float64]uint:
		v2, changed2 := fastpathTV.DecMapFloat64UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]uint8:
		fastpathTV.DecMapFloat64Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]uint8:
		v2, changed2 := fastpathTV.DecMapFloat64Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]uint16:
		fastpathTV.DecMapFloat64Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]uint16:
		v2, changed2 := fastpathTV.DecMapFloat64Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]uint32:
		fastpathTV.DecMapFloat64Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]uint32:
		v2, changed2 := fastpathTV.DecMapFloat64Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]uint64:
		fastpathTV.DecMapFloat64Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]uint64:
		v2, changed2 := fastpathTV.DecMapFloat64Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]int:
		fastpathTV.DecMapFloat64IntV(v, fastpathCheckNilFalse, false, d)
	case *map[float64]int:
		v2, changed2 := fastpathTV.DecMapFloat64IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]int8:
		fastpathTV.DecMapFloat64Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]int8:
		v2, changed2 := fastpathTV.DecMapFloat64Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]int16:
		fastpathTV.DecMapFloat64Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]int16:
		v2, changed2 := fastpathTV.DecMapFloat64Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]int32:
		fastpathTV.DecMapFloat64Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]int32:
		v2, changed2 := fastpathTV.DecMapFloat64Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]int64:
		fastpathTV.DecMapFloat64Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]int64:
		v2, changed2 := fastpathTV.DecMapFloat64Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]float32:
		fastpathTV.DecMapFloat64Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]float32:
		v2, changed2 := fastpathTV.DecMapFloat64Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]float64:
		fastpathTV.DecMapFloat64Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[float64]float64:
		v2, changed2 := fastpathTV.DecMapFloat64Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[float64]bool:
		fastpathTV.DecMapFloat64BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[float64]bool:
		v2, changed2 := fastpathTV.DecMapFloat64BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []uint:
		fastpathTV.DecSliceUintV(v, fastpathCheckNilFalse, false, d)
	case *[]uint:
		v2, changed2 := fastpathTV.DecSliceUintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]interface{}:
		fastpathTV.DecMapUintIntfV(v, fastpathCheckNilFalse, false, d)
	case *map[uint]interface{}:
		v2, changed2 := fastpathTV.DecMapUintIntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]string:
		fastpathTV.DecMapUintStringV(v, fastpathCheckNilFalse, false, d)
	case *map[uint]string:
		v2, changed2 := fastpathTV.DecMapUintStringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]uint:
		fastpathTV.DecMapUintUintV(v, fastpathCheckNilFalse, false, d)
	case *map[uint]uint:
		v2, changed2 := fastpathTV.DecMapUintUintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]uint8:
		fastpathTV.DecMapUintUint8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]uint8:
		v2, changed2 := fastpathTV.DecMapUintUint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]uint16:
		fastpathTV.DecMapUintUint16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]uint16:
		v2, changed2 := fastpathTV.DecMapUintUint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]uint32:
		fastpathTV.DecMapUintUint32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]uint32:
		v2, changed2 := fastpathTV.DecMapUintUint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]uint64:
		fastpathTV.DecMapUintUint64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]uint64:
		v2, changed2 := fastpathTV.DecMapUintUint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]int:
		fastpathTV.DecMapUintIntV(v, fastpathCheckNilFalse, false, d)
	case *map[uint]int:
		v2, changed2 := fastpathTV.DecMapUintIntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]int8:
		fastpathTV.DecMapUintInt8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]int8:
		v2, changed2 := fastpathTV.DecMapUintInt8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]int16:
		fastpathTV.DecMapUintInt16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]int16:
		v2, changed2 := fastpathTV.DecMapUintInt16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]int32:
		fastpathTV.DecMapUintInt32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]int32:
		v2, changed2 := fastpathTV.DecMapUintInt32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]int64:
		fastpathTV.DecMapUintInt64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]int64:
		v2, changed2 := fastpathTV.DecMapUintInt64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]float32:
		fastpathTV.DecMapUintFloat32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]float32:
		v2, changed2 := fastpathTV.DecMapUintFloat32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]float64:
		fastpathTV.DecMapUintFloat64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint]float64:
		v2, changed2 := fastpathTV.DecMapUintFloat64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint]bool:
		fastpathTV.DecMapUintBoolV(v, fastpathCheckNilFalse, false, d)
	case *map[uint]bool:
		v2, changed2 := fastpathTV.DecMapUintBoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]interface{}:
		fastpathTV.DecMapUint8IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]interface{}:
		v2, changed2 := fastpathTV.DecMapUint8IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]string:
		fastpathTV.DecMapUint8StringV(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]string:
		v2, changed2 := fastpathTV.DecMapUint8StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]uint:
		fastpathTV.DecMapUint8UintV(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]uint:
		v2, changed2 := fastpathTV.DecMapUint8UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]uint8:
		fastpathTV.DecMapUint8Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]uint8:
		v2, changed2 := fastpathTV.DecMapUint8Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]uint16:
		fastpathTV.DecMapUint8Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]uint16:
		v2, changed2 := fastpathTV.DecMapUint8Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]uint32:
		fastpathTV.DecMapUint8Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]uint32:
		v2, changed2 := fastpathTV.DecMapUint8Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]uint64:
		fastpathTV.DecMapUint8Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]uint64:
		v2, changed2 := fastpathTV.DecMapUint8Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]int:
		fastpathTV.DecMapUint8IntV(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]int:
		v2, changed2 := fastpathTV.DecMapUint8IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]int8:
		fastpathTV.DecMapUint8Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]int8:
		v2, changed2 := fastpathTV.DecMapUint8Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]int16:
		fastpathTV.DecMapUint8Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]int16:
		v2, changed2 := fastpathTV.DecMapUint8Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]int32:
		fastpathTV.DecMapUint8Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]int32:
		v2, changed2 := fastpathTV.DecMapUint8Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]int64:
		fastpathTV.DecMapUint8Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]int64:
		v2, changed2 := fastpathTV.DecMapUint8Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]float32:
		fastpathTV.DecMapUint8Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]float32:
		v2, changed2 := fastpathTV.DecMapUint8Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]float64:
		fastpathTV.DecMapUint8Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]float64:
		v2, changed2 := fastpathTV.DecMapUint8Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint8]bool:
		fastpathTV.DecMapUint8BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[uint8]bool:
		v2, changed2 := fastpathTV.DecMapUint8BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []uint16:
		fastpathTV.DecSliceUint16V(v, fastpathCheckNilFalse, false, d)
	case *[]uint16:
		v2, changed2 := fastpathTV.DecSliceUint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]interface{}:
		fastpathTV.DecMapUint16IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]interface{}:
		v2, changed2 := fastpathTV.DecMapUint16IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]string:
		fastpathTV.DecMapUint16StringV(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]string:
		v2, changed2 := fastpathTV.DecMapUint16StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]uint:
		fastpathTV.DecMapUint16UintV(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]uint:
		v2, changed2 := fastpathTV.DecMapUint16UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]uint8:
		fastpathTV.DecMapUint16Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]uint8:
		v2, changed2 := fastpathTV.DecMapUint16Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]uint16:
		fastpathTV.DecMapUint16Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]uint16:
		v2, changed2 := fastpathTV.DecMapUint16Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]uint32:
		fastpathTV.DecMapUint16Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]uint32:
		v2, changed2 := fastpathTV.DecMapUint16Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]uint64:
		fastpathTV.DecMapUint16Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]uint64:
		v2, changed2 := fastpathTV.DecMapUint16Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]int:
		fastpathTV.DecMapUint16IntV(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]int:
		v2, changed2 := fastpathTV.DecMapUint16IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]int8:
		fastpathTV.DecMapUint16Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]int8:
		v2, changed2 := fastpathTV.DecMapUint16Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]int16:
		fastpathTV.DecMapUint16Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]int16:
		v2, changed2 := fastpathTV.DecMapUint16Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]int32:
		fastpathTV.DecMapUint16Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]int32:
		v2, changed2 := fastpathTV.DecMapUint16Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]int64:
		fastpathTV.DecMapUint16Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]int64:
		v2, changed2 := fastpathTV.DecMapUint16Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]float32:
		fastpathTV.DecMapUint16Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]float32:
		v2, changed2 := fastpathTV.DecMapUint16Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]float64:
		fastpathTV.DecMapUint16Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]float64:
		v2, changed2 := fastpathTV.DecMapUint16Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint16]bool:
		fastpathTV.DecMapUint16BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[uint16]bool:
		v2, changed2 := fastpathTV.DecMapUint16BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []uint32:
		fastpathTV.DecSliceUint32V(v, fastpathCheckNilFalse, false, d)
	case *[]uint32:
		v2, changed2 := fastpathTV.DecSliceUint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]interface{}:
		fastpathTV.DecMapUint32IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]interface{}:
		v2, changed2 := fastpathTV.DecMapUint32IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]string:
		fastpathTV.DecMapUint32StringV(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]string:
		v2, changed2 := fastpathTV.DecMapUint32StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]uint:
		fastpathTV.DecMapUint32UintV(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]uint:
		v2, changed2 := fastpathTV.DecMapUint32UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]uint8:
		fastpathTV.DecMapUint32Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]uint8:
		v2, changed2 := fastpathTV.DecMapUint32Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]uint16:
		fastpathTV.DecMapUint32Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]uint16:
		v2, changed2 := fastpathTV.DecMapUint32Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]uint32:
		fastpathTV.DecMapUint32Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]uint32:
		v2, changed2 := fastpathTV.DecMapUint32Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]uint64:
		fastpathTV.DecMapUint32Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]uint64:
		v2, changed2 := fastpathTV.DecMapUint32Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]int:
		fastpathTV.DecMapUint32IntV(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]int:
		v2, changed2 := fastpathTV.DecMapUint32IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]int8:
		fastpathTV.DecMapUint32Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]int8:
		v2, changed2 := fastpathTV.DecMapUint32Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]int16:
		fastpathTV.DecMapUint32Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]int16:
		v2, changed2 := fastpathTV.DecMapUint32Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]int32:
		fastpathTV.DecMapUint32Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]int32:
		v2, changed2 := fastpathTV.DecMapUint32Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]int64:
		fastpathTV.DecMapUint32Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]int64:
		v2, changed2 := fastpathTV.DecMapUint32Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]float32:
		fastpathTV.DecMapUint32Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]float32:
		v2, changed2 := fastpathTV.DecMapUint32Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]float64:
		fastpathTV.DecMapUint32Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]float64:
		v2, changed2 := fastpathTV.DecMapUint32Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint32]bool:
		fastpathTV.DecMapUint32BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[uint32]bool:
		v2, changed2 := fastpathTV.DecMapUint32BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []uint64:
		fastpathTV.DecSliceUint64V(v, fastpathCheckNilFalse, false, d)
	case *[]uint64:
		v2, changed2 := fastpathTV.DecSliceUint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]interface{}:
		fastpathTV.DecMapUint64IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]interface{}:
		v2, changed2 := fastpathTV.DecMapUint64IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]string:
		fastpathTV.DecMapUint64StringV(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]string:
		v2, changed2 := fastpathTV.DecMapUint64StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]uint:
		fastpathTV.DecMapUint64UintV(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]uint:
		v2, changed2 := fastpathTV.DecMapUint64UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]uint8:
		fastpathTV.DecMapUint64Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]uint8:
		v2, changed2 := fastpathTV.DecMapUint64Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]uint16:
		fastpathTV.DecMapUint64Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]uint16:
		v2, changed2 := fastpathTV.DecMapUint64Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]uint32:
		fastpathTV.DecMapUint64Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]uint32:
		v2, changed2 := fastpathTV.DecMapUint64Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]uint64:
		fastpathTV.DecMapUint64Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]uint64:
		v2, changed2 := fastpathTV.DecMapUint64Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]int:
		fastpathTV.DecMapUint64IntV(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]int:
		v2, changed2 := fastpathTV.DecMapUint64IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]int8:
		fastpathTV.DecMapUint64Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]int8:
		v2, changed2 := fastpathTV.DecMapUint64Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]int16:
		fastpathTV.DecMapUint64Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]int16:
		v2, changed2 := fastpathTV.DecMapUint64Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]int32:
		fastpathTV.DecMapUint64Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]int32:
		v2, changed2 := fastpathTV.DecMapUint64Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]int64:
		fastpathTV.DecMapUint64Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]int64:
		v2, changed2 := fastpathTV.DecMapUint64Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]float32:
		fastpathTV.DecMapUint64Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]float32:
		v2, changed2 := fastpathTV.DecMapUint64Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]float64:
		fastpathTV.DecMapUint64Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]float64:
		v2, changed2 := fastpathTV.DecMapUint64Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[uint64]bool:
		fastpathTV.DecMapUint64BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[uint64]bool:
		v2, changed2 := fastpathTV.DecMapUint64BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []int:
		fastpathTV.DecSliceIntV(v, fastpathCheckNilFalse, false, d)
	case *[]int:
		v2, changed2 := fastpathTV.DecSliceIntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]interface{}:
		fastpathTV.DecMapIntIntfV(v, fastpathCheckNilFalse, false, d)
	case *map[int]interface{}:
		v2, changed2 := fastpathTV.DecMapIntIntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]string:
		fastpathTV.DecMapIntStringV(v, fastpathCheckNilFalse, false, d)
	case *map[int]string:
		v2, changed2 := fastpathTV.DecMapIntStringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]uint:
		fastpathTV.DecMapIntUintV(v, fastpathCheckNilFalse, false, d)
	case *map[int]uint:
		v2, changed2 := fastpathTV.DecMapIntUintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]uint8:
		fastpathTV.DecMapIntUint8V(v, fastpathCheckNilFalse, false, d)
	case *map[int]uint8:
		v2, changed2 := fastpathTV.DecMapIntUint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]uint16:
		fastpathTV.DecMapIntUint16V(v, fastpathCheckNilFalse, false, d)
	case *map[int]uint16:
		v2, changed2 := fastpathTV.DecMapIntUint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]uint32:
		fastpathTV.DecMapIntUint32V(v, fastpathCheckNilFalse, false, d)
	case *map[int]uint32:
		v2, changed2 := fastpathTV.DecMapIntUint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]uint64:
		fastpathTV.DecMapIntUint64V(v, fastpathCheckNilFalse, false, d)
	case *map[int]uint64:
		v2, changed2 := fastpathTV.DecMapIntUint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]int:
		fastpathTV.DecMapIntIntV(v, fastpathCheckNilFalse, false, d)
	case *map[int]int:
		v2, changed2 := fastpathTV.DecMapIntIntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]int8:
		fastpathTV.DecMapIntInt8V(v, fastpathCheckNilFalse, false, d)
	case *map[int]int8:
		v2, changed2 := fastpathTV.DecMapIntInt8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]int16:
		fastpathTV.DecMapIntInt16V(v, fastpathCheckNilFalse, false, d)
	case *map[int]int16:
		v2, changed2 := fastpathTV.DecMapIntInt16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]int32:
		fastpathTV.DecMapIntInt32V(v, fastpathCheckNilFalse, false, d)
	case *map[int]int32:
		v2, changed2 := fastpathTV.DecMapIntInt32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]int64:
		fastpathTV.DecMapIntInt64V(v, fastpathCheckNilFalse, false, d)
	case *map[int]int64:
		v2, changed2 := fastpathTV.DecMapIntInt64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]float32:
		fastpathTV.DecMapIntFloat32V(v, fastpathCheckNilFalse, false, d)
	case *map[int]float32:
		v2, changed2 := fastpathTV.DecMapIntFloat32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]float64:
		fastpathTV.DecMapIntFloat64V(v, fastpathCheckNilFalse, false, d)
	case *map[int]float64:
		v2, changed2 := fastpathTV.DecMapIntFloat64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int]bool:
		fastpathTV.DecMapIntBoolV(v, fastpathCheckNilFalse, false, d)
	case *map[int]bool:
		v2, changed2 := fastpathTV.DecMapIntBoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []int8:
		fastpathTV.DecSliceInt8V(v, fastpathCheckNilFalse, false, d)
	case *[]int8:
		v2, changed2 := fastpathTV.DecSliceInt8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]interface{}:
		fastpathTV.DecMapInt8IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[int8]interface{}:
		v2, changed2 := fastpathTV.DecMapInt8IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]string:
		fastpathTV.DecMapInt8StringV(v, fastpathCheckNilFalse, false, d)
	case *map[int8]string:
		v2, changed2 := fastpathTV.DecMapInt8StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]uint:
		fastpathTV.DecMapInt8UintV(v, fastpathCheckNilFalse, false, d)
	case *map[int8]uint:
		v2, changed2 := fastpathTV.DecMapInt8UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]uint8:
		fastpathTV.DecMapInt8Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]uint8:
		v2, changed2 := fastpathTV.DecMapInt8Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]uint16:
		fastpathTV.DecMapInt8Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]uint16:
		v2, changed2 := fastpathTV.DecMapInt8Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]uint32:
		fastpathTV.DecMapInt8Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]uint32:
		v2, changed2 := fastpathTV.DecMapInt8Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]uint64:
		fastpathTV.DecMapInt8Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]uint64:
		v2, changed2 := fastpathTV.DecMapInt8Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]int:
		fastpathTV.DecMapInt8IntV(v, fastpathCheckNilFalse, false, d)
	case *map[int8]int:
		v2, changed2 := fastpathTV.DecMapInt8IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]int8:
		fastpathTV.DecMapInt8Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]int8:
		v2, changed2 := fastpathTV.DecMapInt8Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]int16:
		fastpathTV.DecMapInt8Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]int16:
		v2, changed2 := fastpathTV.DecMapInt8Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]int32:
		fastpathTV.DecMapInt8Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]int32:
		v2, changed2 := fastpathTV.DecMapInt8Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]int64:
		fastpathTV.DecMapInt8Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]int64:
		v2, changed2 := fastpathTV.DecMapInt8Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]float32:
		fastpathTV.DecMapInt8Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]float32:
		v2, changed2 := fastpathTV.DecMapInt8Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]float64:
		fastpathTV.DecMapInt8Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[int8]float64:
		v2, changed2 := fastpathTV.DecMapInt8Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int8]bool:
		fastpathTV.DecMapInt8BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[int8]bool:
		v2, changed2 := fastpathTV.DecMapInt8BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []int16:
		fastpathTV.DecSliceInt16V(v, fastpathCheckNilFalse, false, d)
	case *[]int16:
		v2, changed2 := fastpathTV.DecSliceInt16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]interface{}:
		fastpathTV.DecMapInt16IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[int16]interface{}:
		v2, changed2 := fastpathTV.DecMapInt16IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]string:
		fastpathTV.DecMapInt16StringV(v, fastpathCheckNilFalse, false, d)
	case *map[int16]string:
		v2, changed2 := fastpathTV.DecMapInt16StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]uint:
		fastpathTV.DecMapInt16UintV(v, fastpathCheckNilFalse, false, d)
	case *map[int16]uint:
		v2, changed2 := fastpathTV.DecMapInt16UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]uint8:
		fastpathTV.DecMapInt16Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]uint8:
		v2, changed2 := fastpathTV.DecMapInt16Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]uint16:
		fastpathTV.DecMapInt16Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]uint16:
		v2, changed2 := fastpathTV.DecMapInt16Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]uint32:
		fastpathTV.DecMapInt16Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]uint32:
		v2, changed2 := fastpathTV.DecMapInt16Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]uint64:
		fastpathTV.DecMapInt16Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]uint64:
		v2, changed2 := fastpathTV.DecMapInt16Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]int:
		fastpathTV.DecMapInt16IntV(v, fastpathCheckNilFalse, false, d)
	case *map[int16]int:
		v2, changed2 := fastpathTV.DecMapInt16IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]int8:
		fastpathTV.DecMapInt16Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]int8:
		v2, changed2 := fastpathTV.DecMapInt16Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]int16:
		fastpathTV.DecMapInt16Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]int16:
		v2, changed2 := fastpathTV.DecMapInt16Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]int32:
		fastpathTV.DecMapInt16Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]int32:
		v2, changed2 := fastpathTV.DecMapInt16Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]int64:
		fastpathTV.DecMapInt16Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]int64:
		v2, changed2 := fastpathTV.DecMapInt16Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]float32:
		fastpathTV.DecMapInt16Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]float32:
		v2, changed2 := fastpathTV.DecMapInt16Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]float64:
		fastpathTV.DecMapInt16Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[int16]float64:
		v2, changed2 := fastpathTV.DecMapInt16Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int16]bool:
		fastpathTV.DecMapInt16BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[int16]bool:
		v2, changed2 := fastpathTV.DecMapInt16BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []int32:
		fastpathTV.DecSliceInt32V(v, fastpathCheckNilFalse, false, d)
	case *[]int32:
		v2, changed2 := fastpathTV.DecSliceInt32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]interface{}:
		fastpathTV.DecMapInt32IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[int32]interface{}:
		v2, changed2 := fastpathTV.DecMapInt32IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]string:
		fastpathTV.DecMapInt32StringV(v, fastpathCheckNilFalse, false, d)
	case *map[int32]string:
		v2, changed2 := fastpathTV.DecMapInt32StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]uint:
		fastpathTV.DecMapInt32UintV(v, fastpathCheckNilFalse, false, d)
	case *map[int32]uint:
		v2, changed2 := fastpathTV.DecMapInt32UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]uint8:
		fastpathTV.DecMapInt32Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]uint8:
		v2, changed2 := fastpathTV.DecMapInt32Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]uint16:
		fastpathTV.DecMapInt32Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]uint16:
		v2, changed2 := fastpathTV.DecMapInt32Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]uint32:
		fastpathTV.DecMapInt32Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]uint32:
		v2, changed2 := fastpathTV.DecMapInt32Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]uint64:
		fastpathTV.DecMapInt32Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]uint64:
		v2, changed2 := fastpathTV.DecMapInt32Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]int:
		fastpathTV.DecMapInt32IntV(v, fastpathCheckNilFalse, false, d)
	case *map[int32]int:
		v2, changed2 := fastpathTV.DecMapInt32IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]int8:
		fastpathTV.DecMapInt32Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]int8:
		v2, changed2 := fastpathTV.DecMapInt32Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]int16:
		fastpathTV.DecMapInt32Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]int16:
		v2, changed2 := fastpathTV.DecMapInt32Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]int32:
		fastpathTV.DecMapInt32Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]int32:
		v2, changed2 := fastpathTV.DecMapInt32Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]int64:
		fastpathTV.DecMapInt32Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]int64:
		v2, changed2 := fastpathTV.DecMapInt32Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]float32:
		fastpathTV.DecMapInt32Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]float32:
		v2, changed2 := fastpathTV.DecMapInt32Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]float64:
		fastpathTV.DecMapInt32Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[int32]float64:
		v2, changed2 := fastpathTV.DecMapInt32Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int32]bool:
		fastpathTV.DecMapInt32BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[int32]bool:
		v2, changed2 := fastpathTV.DecMapInt32BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []int64:
		fastpathTV.DecSliceInt64V(v, fastpathCheckNilFalse, false, d)
	case *[]int64:
		v2, changed2 := fastpathTV.DecSliceInt64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]interface{}:
		fastpathTV.DecMapInt64IntfV(v, fastpathCheckNilFalse, false, d)
	case *map[int64]interface{}:
		v2, changed2 := fastpathTV.DecMapInt64IntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]string:
		fastpathTV.DecMapInt64StringV(v, fastpathCheckNilFalse, false, d)
	case *map[int64]string:
		v2, changed2 := fastpathTV.DecMapInt64StringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]uint:
		fastpathTV.DecMapInt64UintV(v, fastpathCheckNilFalse, false, d)
	case *map[int64]uint:
		v2, changed2 := fastpathTV.DecMapInt64UintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]uint8:
		fastpathTV.DecMapInt64Uint8V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]uint8:
		v2, changed2 := fastpathTV.DecMapInt64Uint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]uint16:
		fastpathTV.DecMapInt64Uint16V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]uint16:
		v2, changed2 := fastpathTV.DecMapInt64Uint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]uint32:
		fastpathTV.DecMapInt64Uint32V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]uint32:
		v2, changed2 := fastpathTV.DecMapInt64Uint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]uint64:
		fastpathTV.DecMapInt64Uint64V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]uint64:
		v2, changed2 := fastpathTV.DecMapInt64Uint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]int:
		fastpathTV.DecMapInt64IntV(v, fastpathCheckNilFalse, false, d)
	case *map[int64]int:
		v2, changed2 := fastpathTV.DecMapInt64IntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]int8:
		fastpathTV.DecMapInt64Int8V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]int8:
		v2, changed2 := fastpathTV.DecMapInt64Int8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]int16:
		fastpathTV.DecMapInt64Int16V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]int16:
		v2, changed2 := fastpathTV.DecMapInt64Int16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]int32:
		fastpathTV.DecMapInt64Int32V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]int32:
		v2, changed2 := fastpathTV.DecMapInt64Int32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]int64:
		fastpathTV.DecMapInt64Int64V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]int64:
		v2, changed2 := fastpathTV.DecMapInt64Int64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]float32:
		fastpathTV.DecMapInt64Float32V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]float32:
		v2, changed2 := fastpathTV.DecMapInt64Float32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]float64:
		fastpathTV.DecMapInt64Float64V(v, fastpathCheckNilFalse, false, d)
	case *map[int64]float64:
		v2, changed2 := fastpathTV.DecMapInt64Float64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[int64]bool:
		fastpathTV.DecMapInt64BoolV(v, fastpathCheckNilFalse, false, d)
	case *map[int64]bool:
		v2, changed2 := fastpathTV.DecMapInt64BoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case []bool:
		fastpathTV.DecSliceBoolV(v, fastpathCheckNilFalse, false, d)
	case *[]bool:
		v2, changed2 := fastpathTV.DecSliceBoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]interface{}:
		fastpathTV.DecMapBoolIntfV(v, fastpathCheckNilFalse, false, d)
	case *map[bool]interface{}:
		v2, changed2 := fastpathTV.DecMapBoolIntfV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]string:
		fastpathTV.DecMapBoolStringV(v, fastpathCheckNilFalse, false, d)
	case *map[bool]string:
		v2, changed2 := fastpathTV.DecMapBoolStringV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]uint:
		fastpathTV.DecMapBoolUintV(v, fastpathCheckNilFalse, false, d)
	case *map[bool]uint:
		v2, changed2 := fastpathTV.DecMapBoolUintV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]uint8:
		fastpathTV.DecMapBoolUint8V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]uint8:
		v2, changed2 := fastpathTV.DecMapBoolUint8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]uint16:
		fastpathTV.DecMapBoolUint16V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]uint16:
		v2, changed2 := fastpathTV.DecMapBoolUint16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]uint32:
		fastpathTV.DecMapBoolUint32V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]uint32:
		v2, changed2 := fastpathTV.DecMapBoolUint32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]uint64:
		fastpathTV.DecMapBoolUint64V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]uint64:
		v2, changed2 := fastpathTV.DecMapBoolUint64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]int:
		fastpathTV.DecMapBoolIntV(v, fastpathCheckNilFalse, false, d)
	case *map[bool]int:
		v2, changed2 := fastpathTV.DecMapBoolIntV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]int8:
		fastpathTV.DecMapBoolInt8V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]int8:
		v2, changed2 := fastpathTV.DecMapBoolInt8V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]int16:
		fastpathTV.DecMapBoolInt16V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]int16:
		v2, changed2 := fastpathTV.DecMapBoolInt16V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]int32:
		fastpathTV.DecMapBoolInt32V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]int32:
		v2, changed2 := fastpathTV.DecMapBoolInt32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]int64:
		fastpathTV.DecMapBoolInt64V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]int64:
		v2, changed2 := fastpathTV.DecMapBoolInt64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]float32:
		fastpathTV.DecMapBoolFloat32V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]float32:
		v2, changed2 := fastpathTV.DecMapBoolFloat32V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]float64:
		fastpathTV.DecMapBoolFloat64V(v, fastpathCheckNilFalse, false, d)
	case *map[bool]float64:
		v2, changed2 := fastpathTV.DecMapBoolFloat64V(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	case map[bool]bool:
		fastpathTV.DecMapBoolBoolV(v, fastpathCheckNilFalse, false, d)
	case *map[bool]bool:
		v2, changed2 := fastpathTV.DecMapBoolBoolV(*v, fastpathCheckNilFalse, true, d)
		if changed2 {
			*v = v2
		}

	default:
		return false
	}
	return true
}

// -- -- fast path functions

func (f *decFnInfo) fastpathDecSliceIntfR(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]interface{})
		v, changed := fastpathTV.DecSliceIntfV(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]interface{})
		fastpathTV.DecSliceIntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceIntfX(vp *[]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceIntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceIntfV(v []interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ []interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 16); xtrunc {
			x2read = xlen
		}
		v = make([]interface{}, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 16); xtrunc {
					x2read = xlen
				}
				v = make([]interface{}, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			d.decode(&v[j])
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, nil)
				d.decode(&v[j])
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, nil)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				d.decode(&v[j])

			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceStringR(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]string)
		v, changed := fastpathTV.DecSliceStringV(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]string)
		fastpathTV.DecSliceStringV(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceStringX(vp *[]string, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceStringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceStringV(v []string, checkNil bool, canChange bool,
	d *Decoder) (_ []string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 16); xtrunc {
			x2read = xlen
		}
		v = make([]string, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 16); xtrunc {
					x2read = xlen
				}
				v = make([]string, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = dd.DecodeString()
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, "")
				v[j] = dd.DecodeString()
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, "")
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = dd.DecodeString()
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceFloat32R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]float32)
		v, changed := fastpathTV.DecSliceFloat32V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]float32)
		fastpathTV.DecSliceFloat32V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceFloat32X(vp *[]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceFloat32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceFloat32V(v []float32, checkNil bool, canChange bool,
	d *Decoder) (_ []float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 4); xtrunc {
			x2read = xlen
		}
		v = make([]float32, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 4); xtrunc {
					x2read = xlen
				}
				v = make([]float32, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = float32(dd.DecodeFloat(true))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = float32(dd.DecodeFloat(true))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = float32(dd.DecodeFloat(true))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceFloat64R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]float64)
		v, changed := fastpathTV.DecSliceFloat64V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]float64)
		fastpathTV.DecSliceFloat64V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceFloat64X(vp *[]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceFloat64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceFloat64V(v []float64, checkNil bool, canChange bool,
	d *Decoder) (_ []float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
			x2read = xlen
		}
		v = make([]float64, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
					x2read = xlen
				}
				v = make([]float64, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = dd.DecodeFloat(false)
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = dd.DecodeFloat(false)
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = dd.DecodeFloat(false)
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceUintR(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]uint)
		v, changed := fastpathTV.DecSliceUintV(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]uint)
		fastpathTV.DecSliceUintV(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceUintX(vp *[]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceUintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUintV(v []uint, checkNil bool, canChange bool,
	d *Decoder) (_ []uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
			x2read = xlen
		}
		v = make([]uint, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
					x2read = xlen
				}
				v = make([]uint, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = uint(dd.DecodeUint(uintBitsize))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = uint(dd.DecodeUint(uintBitsize))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = uint(dd.DecodeUint(uintBitsize))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceUint16R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]uint16)
		v, changed := fastpathTV.DecSliceUint16V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]uint16)
		fastpathTV.DecSliceUint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceUint16X(vp *[]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceUint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUint16V(v []uint16, checkNil bool, canChange bool,
	d *Decoder) (_ []uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 2); xtrunc {
			x2read = xlen
		}
		v = make([]uint16, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 2); xtrunc {
					x2read = xlen
				}
				v = make([]uint16, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = uint16(dd.DecodeUint(16))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = uint16(dd.DecodeUint(16))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = uint16(dd.DecodeUint(16))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceUint32R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]uint32)
		v, changed := fastpathTV.DecSliceUint32V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]uint32)
		fastpathTV.DecSliceUint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceUint32X(vp *[]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceUint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUint32V(v []uint32, checkNil bool, canChange bool,
	d *Decoder) (_ []uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 4); xtrunc {
			x2read = xlen
		}
		v = make([]uint32, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 4); xtrunc {
					x2read = xlen
				}
				v = make([]uint32, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = uint32(dd.DecodeUint(32))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = uint32(dd.DecodeUint(32))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = uint32(dd.DecodeUint(32))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceUint64R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]uint64)
		v, changed := fastpathTV.DecSliceUint64V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]uint64)
		fastpathTV.DecSliceUint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceUint64X(vp *[]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceUint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUint64V(v []uint64, checkNil bool, canChange bool,
	d *Decoder) (_ []uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
			x2read = xlen
		}
		v = make([]uint64, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
					x2read = xlen
				}
				v = make([]uint64, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = dd.DecodeUint(64)
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = dd.DecodeUint(64)
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = dd.DecodeUint(64)
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceIntR(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]int)
		v, changed := fastpathTV.DecSliceIntV(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]int)
		fastpathTV.DecSliceIntV(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceIntX(vp *[]int, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceIntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceIntV(v []int, checkNil bool, canChange bool,
	d *Decoder) (_ []int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
			x2read = xlen
		}
		v = make([]int, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
					x2read = xlen
				}
				v = make([]int, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = int(dd.DecodeInt(intBitsize))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = int(dd.DecodeInt(intBitsize))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = int(dd.DecodeInt(intBitsize))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceInt8R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]int8)
		v, changed := fastpathTV.DecSliceInt8V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]int8)
		fastpathTV.DecSliceInt8V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceInt8X(vp *[]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceInt8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt8V(v []int8, checkNil bool, canChange bool,
	d *Decoder) (_ []int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 1); xtrunc {
			x2read = xlen
		}
		v = make([]int8, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 1); xtrunc {
					x2read = xlen
				}
				v = make([]int8, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = int8(dd.DecodeInt(8))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = int8(dd.DecodeInt(8))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = int8(dd.DecodeInt(8))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceInt16R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]int16)
		v, changed := fastpathTV.DecSliceInt16V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]int16)
		fastpathTV.DecSliceInt16V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceInt16X(vp *[]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceInt16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt16V(v []int16, checkNil bool, canChange bool,
	d *Decoder) (_ []int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 2); xtrunc {
			x2read = xlen
		}
		v = make([]int16, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 2); xtrunc {
					x2read = xlen
				}
				v = make([]int16, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = int16(dd.DecodeInt(16))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = int16(dd.DecodeInt(16))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = int16(dd.DecodeInt(16))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceInt32R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]int32)
		v, changed := fastpathTV.DecSliceInt32V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]int32)
		fastpathTV.DecSliceInt32V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceInt32X(vp *[]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceInt32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt32V(v []int32, checkNil bool, canChange bool,
	d *Decoder) (_ []int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 4); xtrunc {
			x2read = xlen
		}
		v = make([]int32, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 4); xtrunc {
					x2read = xlen
				}
				v = make([]int32, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = int32(dd.DecodeInt(32))
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = int32(dd.DecodeInt(32))
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = int32(dd.DecodeInt(32))
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceInt64R(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]int64)
		v, changed := fastpathTV.DecSliceInt64V(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]int64)
		fastpathTV.DecSliceInt64V(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceInt64X(vp *[]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceInt64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt64V(v []int64, checkNil bool, canChange bool,
	d *Decoder) (_ []int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
			x2read = xlen
		}
		v = make([]int64, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 8); xtrunc {
					x2read = xlen
				}
				v = make([]int64, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = dd.DecodeInt(64)
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, 0)
				v[j] = dd.DecodeInt(64)
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, 0)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = dd.DecodeInt(64)
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecSliceBoolR(rv reflect.Value) {
	array := f.seq == seqTypeArray
	if !array && rv.CanAddr() {
		vp := rv.Addr().Interface().(*[]bool)
		v, changed := fastpathTV.DecSliceBoolV(*vp, fastpathCheckNilFalse, !array, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().([]bool)
		fastpathTV.DecSliceBoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}

func (f fastpathT) DecSliceBoolX(vp *[]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecSliceBoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceBoolV(v []bool, checkNil bool, canChange bool,
	d *Decoder) (_ []bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	slh, containerLenS := d.decSliceHelperStart()
	x2read := containerLenS
	var xtrunc bool
	if canChange && v == nil {
		var xlen int
		if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 1); xtrunc {
			x2read = xlen
		}
		v = make([]bool, xlen)
		changed = true
	}
	if containerLenS == 0 {
		if canChange && len(v) != 0 {
			v = v[:0]
			changed = true
		}
		return v, changed
	}

	if containerLenS > 0 {
		if containerLenS > cap(v) {
			if canChange {
				var xlen int
				if xlen, xtrunc = decInferLen(containerLenS, d.h.MaxInitLen, 1); xtrunc {
					x2read = xlen
				}
				v = make([]bool, xlen)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), containerLenS)
				x2read = len(v)
			}
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}

		j := 0
		for ; j < x2read; j++ {
			v[j] = dd.DecodeBool()
		}
		if xtrunc {
			for ; j < containerLenS; j++ {
				v = append(v, false)
				v[j] = dd.DecodeBool()
			}
		} else if !canChange {
			for ; j < containerLenS; j++ {
				d.swallow()
			}
		}
	} else {
		j := 0
		for ; !dd.CheckBreak(); j++ {
			if j >= len(v) {
				if canChange {
					v = append(v, false)
					changed = true
				} else {
					d.arrayCannotExpand(len(v), j+1)
				}
			}
			if j < len(v) {
				v[j] = dd.DecodeBool()
			} else {
				d.swallow()
			}
		}
		slh.End()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfIntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]interface{})
		v, changed := fastpathTV.DecMapIntfIntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]interface{})
		fastpathTV.DecMapIntfIntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfIntfX(vp *map[interface{}]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfIntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfIntfV(v map[interface{}]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[interface{}]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfStringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]string)
		v, changed := fastpathTV.DecMapIntfStringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]string)
		fastpathTV.DecMapIntfStringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfStringX(vp *map[interface{}]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfStringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfStringV(v map[interface{}]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[interface{}]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfUintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]uint)
		v, changed := fastpathTV.DecMapIntfUintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]uint)
		fastpathTV.DecMapIntfUintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfUintX(vp *map[interface{}]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfUintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUintV(v map[interface{}]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfUint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]uint8)
		v, changed := fastpathTV.DecMapIntfUint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]uint8)
		fastpathTV.DecMapIntfUint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfUint8X(vp *map[interface{}]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfUint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint8V(v map[interface{}]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[interface{}]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfUint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]uint16)
		v, changed := fastpathTV.DecMapIntfUint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]uint16)
		fastpathTV.DecMapIntfUint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfUint16X(vp *map[interface{}]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfUint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint16V(v map[interface{}]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[interface{}]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfUint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]uint32)
		v, changed := fastpathTV.DecMapIntfUint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]uint32)
		fastpathTV.DecMapIntfUint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfUint32X(vp *map[interface{}]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfUint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint32V(v map[interface{}]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[interface{}]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfUint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]uint64)
		v, changed := fastpathTV.DecMapIntfUint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]uint64)
		fastpathTV.DecMapIntfUint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfUint64X(vp *map[interface{}]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfUint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint64V(v map[interface{}]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfIntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]int)
		v, changed := fastpathTV.DecMapIntfIntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]int)
		fastpathTV.DecMapIntfIntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfIntX(vp *map[interface{}]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfIntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfIntV(v map[interface{}]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfInt8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]int8)
		v, changed := fastpathTV.DecMapIntfInt8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]int8)
		fastpathTV.DecMapIntfInt8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfInt8X(vp *map[interface{}]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfInt8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt8V(v map[interface{}]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[interface{}]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfInt16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]int16)
		v, changed := fastpathTV.DecMapIntfInt16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]int16)
		fastpathTV.DecMapIntfInt16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfInt16X(vp *map[interface{}]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfInt16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt16V(v map[interface{}]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[interface{}]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfInt32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]int32)
		v, changed := fastpathTV.DecMapIntfInt32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]int32)
		fastpathTV.DecMapIntfInt32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfInt32X(vp *map[interface{}]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfInt32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt32V(v map[interface{}]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[interface{}]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfInt64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]int64)
		v, changed := fastpathTV.DecMapIntfInt64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]int64)
		fastpathTV.DecMapIntfInt64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfInt64X(vp *map[interface{}]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfInt64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt64V(v map[interface{}]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfFloat32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]float32)
		v, changed := fastpathTV.DecMapIntfFloat32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]float32)
		fastpathTV.DecMapIntfFloat32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfFloat32X(vp *map[interface{}]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfFloat32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfFloat32V(v map[interface{}]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[interface{}]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfFloat64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]float64)
		v, changed := fastpathTV.DecMapIntfFloat64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]float64)
		fastpathTV.DecMapIntfFloat64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfFloat64X(vp *map[interface{}]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfFloat64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfFloat64V(v map[interface{}]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntfBoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[interface{}]bool)
		v, changed := fastpathTV.DecMapIntfBoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[interface{}]bool)
		fastpathTV.DecMapIntfBoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntfBoolX(vp *map[interface{}]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntfBoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfBoolV(v map[interface{}]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[interface{}]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[interface{}]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			var mk interface{}
			d.decode(&mk)
			if bv, bok := mk.([]byte); bok {
				mk = string(bv)
			}
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringIntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]interface{})
		v, changed := fastpathTV.DecMapStringIntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]interface{})
		fastpathTV.DecMapStringIntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringIntfX(vp *map[string]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringIntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringIntfV(v map[string]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[string]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringStringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]string)
		v, changed := fastpathTV.DecMapStringStringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]string)
		fastpathTV.DecMapStringStringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringStringX(vp *map[string]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringStringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringStringV(v map[string]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[string]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringUintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]uint)
		v, changed := fastpathTV.DecMapStringUintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]uint)
		fastpathTV.DecMapStringUintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringUintX(vp *map[string]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringUintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUintV(v map[string]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringUint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]uint8)
		v, changed := fastpathTV.DecMapStringUint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]uint8)
		fastpathTV.DecMapStringUint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringUint8X(vp *map[string]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringUint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint8V(v map[string]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[string]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringUint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]uint16)
		v, changed := fastpathTV.DecMapStringUint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]uint16)
		fastpathTV.DecMapStringUint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringUint16X(vp *map[string]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringUint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint16V(v map[string]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[string]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringUint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]uint32)
		v, changed := fastpathTV.DecMapStringUint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]uint32)
		fastpathTV.DecMapStringUint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringUint32X(vp *map[string]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringUint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint32V(v map[string]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[string]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringUint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]uint64)
		v, changed := fastpathTV.DecMapStringUint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]uint64)
		fastpathTV.DecMapStringUint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringUint64X(vp *map[string]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringUint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint64V(v map[string]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringIntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]int)
		v, changed := fastpathTV.DecMapStringIntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]int)
		fastpathTV.DecMapStringIntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringIntX(vp *map[string]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringIntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringIntV(v map[string]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringInt8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]int8)
		v, changed := fastpathTV.DecMapStringInt8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]int8)
		fastpathTV.DecMapStringInt8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringInt8X(vp *map[string]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringInt8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt8V(v map[string]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[string]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringInt16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]int16)
		v, changed := fastpathTV.DecMapStringInt16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]int16)
		fastpathTV.DecMapStringInt16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringInt16X(vp *map[string]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringInt16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt16V(v map[string]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[string]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringInt32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]int32)
		v, changed := fastpathTV.DecMapStringInt32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]int32)
		fastpathTV.DecMapStringInt32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringInt32X(vp *map[string]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringInt32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt32V(v map[string]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[string]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringInt64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]int64)
		v, changed := fastpathTV.DecMapStringInt64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]int64)
		fastpathTV.DecMapStringInt64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringInt64X(vp *map[string]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringInt64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt64V(v map[string]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringFloat32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]float32)
		v, changed := fastpathTV.DecMapStringFloat32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]float32)
		fastpathTV.DecMapStringFloat32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringFloat32X(vp *map[string]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringFloat32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringFloat32V(v map[string]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[string]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringFloat64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]float64)
		v, changed := fastpathTV.DecMapStringFloat64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]float64)
		fastpathTV.DecMapStringFloat64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringFloat64X(vp *map[string]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringFloat64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringFloat64V(v map[string]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapStringBoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[string]bool)
		v, changed := fastpathTV.DecMapStringBoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[string]bool)
		fastpathTV.DecMapStringBoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapStringBoolX(vp *map[string]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapStringBoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringBoolV(v map[string]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[string]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[string]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeString()
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]interface{})
		v, changed := fastpathTV.DecMapFloat32IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]interface{})
		fastpathTV.DecMapFloat32IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32IntfX(vp *map[float32]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32IntfV(v map[float32]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[float32]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]string)
		v, changed := fastpathTV.DecMapFloat32StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]string)
		fastpathTV.DecMapFloat32StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32StringX(vp *map[float32]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32StringV(v map[float32]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[float32]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]uint)
		v, changed := fastpathTV.DecMapFloat32UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]uint)
		fastpathTV.DecMapFloat32UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32UintX(vp *map[float32]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32UintV(v map[float32]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]uint8)
		v, changed := fastpathTV.DecMapFloat32Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]uint8)
		fastpathTV.DecMapFloat32Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Uint8X(vp *map[float32]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint8V(v map[float32]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[float32]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]uint16)
		v, changed := fastpathTV.DecMapFloat32Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]uint16)
		fastpathTV.DecMapFloat32Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Uint16X(vp *map[float32]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint16V(v map[float32]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[float32]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]uint32)
		v, changed := fastpathTV.DecMapFloat32Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]uint32)
		fastpathTV.DecMapFloat32Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Uint32X(vp *map[float32]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint32V(v map[float32]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[float32]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]uint64)
		v, changed := fastpathTV.DecMapFloat32Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]uint64)
		fastpathTV.DecMapFloat32Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Uint64X(vp *map[float32]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint64V(v map[float32]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]int)
		v, changed := fastpathTV.DecMapFloat32IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]int)
		fastpathTV.DecMapFloat32IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32IntX(vp *map[float32]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32IntV(v map[float32]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]int8)
		v, changed := fastpathTV.DecMapFloat32Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]int8)
		fastpathTV.DecMapFloat32Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Int8X(vp *map[float32]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int8V(v map[float32]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[float32]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]int16)
		v, changed := fastpathTV.DecMapFloat32Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]int16)
		fastpathTV.DecMapFloat32Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Int16X(vp *map[float32]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int16V(v map[float32]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[float32]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]int32)
		v, changed := fastpathTV.DecMapFloat32Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]int32)
		fastpathTV.DecMapFloat32Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Int32X(vp *map[float32]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int32V(v map[float32]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[float32]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]int64)
		v, changed := fastpathTV.DecMapFloat32Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]int64)
		fastpathTV.DecMapFloat32Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Int64X(vp *map[float32]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int64V(v map[float32]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]float32)
		v, changed := fastpathTV.DecMapFloat32Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]float32)
		fastpathTV.DecMapFloat32Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Float32X(vp *map[float32]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Float32V(v map[float32]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[float32]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]float64)
		v, changed := fastpathTV.DecMapFloat32Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]float64)
		fastpathTV.DecMapFloat32Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32Float64X(vp *map[float32]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Float64V(v map[float32]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat32BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float32]bool)
		v, changed := fastpathTV.DecMapFloat32BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float32]bool)
		fastpathTV.DecMapFloat32BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat32BoolX(vp *map[float32]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat32BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32BoolV(v map[float32]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[float32]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[float32]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := float32(dd.DecodeFloat(true))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]interface{})
		v, changed := fastpathTV.DecMapFloat64IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]interface{})
		fastpathTV.DecMapFloat64IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64IntfX(vp *map[float64]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64IntfV(v map[float64]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[float64]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]string)
		v, changed := fastpathTV.DecMapFloat64StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]string)
		fastpathTV.DecMapFloat64StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64StringX(vp *map[float64]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64StringV(v map[float64]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[float64]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]uint)
		v, changed := fastpathTV.DecMapFloat64UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]uint)
		fastpathTV.DecMapFloat64UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64UintX(vp *map[float64]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64UintV(v map[float64]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]uint8)
		v, changed := fastpathTV.DecMapFloat64Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]uint8)
		fastpathTV.DecMapFloat64Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Uint8X(vp *map[float64]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint8V(v map[float64]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[float64]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]uint16)
		v, changed := fastpathTV.DecMapFloat64Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]uint16)
		fastpathTV.DecMapFloat64Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Uint16X(vp *map[float64]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint16V(v map[float64]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[float64]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]uint32)
		v, changed := fastpathTV.DecMapFloat64Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]uint32)
		fastpathTV.DecMapFloat64Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Uint32X(vp *map[float64]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint32V(v map[float64]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float64]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]uint64)
		v, changed := fastpathTV.DecMapFloat64Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]uint64)
		fastpathTV.DecMapFloat64Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Uint64X(vp *map[float64]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint64V(v map[float64]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]int)
		v, changed := fastpathTV.DecMapFloat64IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]int)
		fastpathTV.DecMapFloat64IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64IntX(vp *map[float64]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64IntV(v map[float64]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]int8)
		v, changed := fastpathTV.DecMapFloat64Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]int8)
		fastpathTV.DecMapFloat64Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Int8X(vp *map[float64]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int8V(v map[float64]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[float64]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]int16)
		v, changed := fastpathTV.DecMapFloat64Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]int16)
		fastpathTV.DecMapFloat64Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Int16X(vp *map[float64]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int16V(v map[float64]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[float64]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]int32)
		v, changed := fastpathTV.DecMapFloat64Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]int32)
		fastpathTV.DecMapFloat64Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Int32X(vp *map[float64]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int32V(v map[float64]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float64]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]int64)
		v, changed := fastpathTV.DecMapFloat64Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]int64)
		fastpathTV.DecMapFloat64Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Int64X(vp *map[float64]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int64V(v map[float64]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]float32)
		v, changed := fastpathTV.DecMapFloat64Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]float32)
		fastpathTV.DecMapFloat64Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Float32X(vp *map[float64]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Float32V(v map[float64]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float64]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]float64)
		v, changed := fastpathTV.DecMapFloat64Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]float64)
		fastpathTV.DecMapFloat64Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64Float64X(vp *map[float64]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Float64V(v map[float64]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapFloat64BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[float64]bool)
		v, changed := fastpathTV.DecMapFloat64BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[float64]bool)
		fastpathTV.DecMapFloat64BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapFloat64BoolX(vp *map[float64]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapFloat64BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64BoolV(v map[float64]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[float64]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[float64]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeFloat(false)
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintIntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]interface{})
		v, changed := fastpathTV.DecMapUintIntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]interface{})
		fastpathTV.DecMapUintIntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintIntfX(vp *map[uint]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintIntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintIntfV(v map[uint]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintStringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]string)
		v, changed := fastpathTV.DecMapUintStringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]string)
		fastpathTV.DecMapUintStringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintStringX(vp *map[uint]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintStringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintStringV(v map[uint]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintUintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]uint)
		v, changed := fastpathTV.DecMapUintUintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]uint)
		fastpathTV.DecMapUintUintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintUintX(vp *map[uint]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintUintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUintV(v map[uint]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintUint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]uint8)
		v, changed := fastpathTV.DecMapUintUint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]uint8)
		fastpathTV.DecMapUintUint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintUint8X(vp *map[uint]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintUint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint8V(v map[uint]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintUint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]uint16)
		v, changed := fastpathTV.DecMapUintUint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]uint16)
		fastpathTV.DecMapUintUint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintUint16X(vp *map[uint]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintUint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint16V(v map[uint]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintUint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]uint32)
		v, changed := fastpathTV.DecMapUintUint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]uint32)
		fastpathTV.DecMapUintUint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintUint32X(vp *map[uint]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintUint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint32V(v map[uint]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintUint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]uint64)
		v, changed := fastpathTV.DecMapUintUint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]uint64)
		fastpathTV.DecMapUintUint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintUint64X(vp *map[uint]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintUint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint64V(v map[uint]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintIntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]int)
		v, changed := fastpathTV.DecMapUintIntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]int)
		fastpathTV.DecMapUintIntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintIntX(vp *map[uint]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintIntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintIntV(v map[uint]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintInt8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]int8)
		v, changed := fastpathTV.DecMapUintInt8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]int8)
		fastpathTV.DecMapUintInt8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintInt8X(vp *map[uint]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintInt8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt8V(v map[uint]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintInt16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]int16)
		v, changed := fastpathTV.DecMapUintInt16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]int16)
		fastpathTV.DecMapUintInt16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintInt16X(vp *map[uint]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintInt16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt16V(v map[uint]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintInt32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]int32)
		v, changed := fastpathTV.DecMapUintInt32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]int32)
		fastpathTV.DecMapUintInt32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintInt32X(vp *map[uint]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintInt32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt32V(v map[uint]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintInt64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]int64)
		v, changed := fastpathTV.DecMapUintInt64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]int64)
		fastpathTV.DecMapUintInt64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintInt64X(vp *map[uint]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintInt64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt64V(v map[uint]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintFloat32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]float32)
		v, changed := fastpathTV.DecMapUintFloat32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]float32)
		fastpathTV.DecMapUintFloat32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintFloat32X(vp *map[uint]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintFloat32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintFloat32V(v map[uint]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintFloat64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]float64)
		v, changed := fastpathTV.DecMapUintFloat64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]float64)
		fastpathTV.DecMapUintFloat64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintFloat64X(vp *map[uint]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintFloat64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintFloat64V(v map[uint]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUintBoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint]bool)
		v, changed := fastpathTV.DecMapUintBoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint]bool)
		fastpathTV.DecMapUintBoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUintBoolX(vp *map[uint]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUintBoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintBoolV(v map[uint]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint(dd.DecodeUint(uintBitsize))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]interface{})
		v, changed := fastpathTV.DecMapUint8IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]interface{})
		fastpathTV.DecMapUint8IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8IntfX(vp *map[uint8]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8IntfV(v map[uint8]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[uint8]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]string)
		v, changed := fastpathTV.DecMapUint8StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]string)
		fastpathTV.DecMapUint8StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8StringX(vp *map[uint8]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8StringV(v map[uint8]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[uint8]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]uint)
		v, changed := fastpathTV.DecMapUint8UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]uint)
		fastpathTV.DecMapUint8UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8UintX(vp *map[uint8]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8UintV(v map[uint8]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]uint8)
		v, changed := fastpathTV.DecMapUint8Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]uint8)
		fastpathTV.DecMapUint8Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Uint8X(vp *map[uint8]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint8V(v map[uint8]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[uint8]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]uint16)
		v, changed := fastpathTV.DecMapUint8Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]uint16)
		fastpathTV.DecMapUint8Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Uint16X(vp *map[uint8]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint16V(v map[uint8]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint8]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]uint32)
		v, changed := fastpathTV.DecMapUint8Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]uint32)
		fastpathTV.DecMapUint8Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Uint32X(vp *map[uint8]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint32V(v map[uint8]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint8]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]uint64)
		v, changed := fastpathTV.DecMapUint8Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]uint64)
		fastpathTV.DecMapUint8Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Uint64X(vp *map[uint8]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint64V(v map[uint8]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]int)
		v, changed := fastpathTV.DecMapUint8IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]int)
		fastpathTV.DecMapUint8IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8IntX(vp *map[uint8]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8IntV(v map[uint8]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]int8)
		v, changed := fastpathTV.DecMapUint8Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]int8)
		fastpathTV.DecMapUint8Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Int8X(vp *map[uint8]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int8V(v map[uint8]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[uint8]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]int16)
		v, changed := fastpathTV.DecMapUint8Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]int16)
		fastpathTV.DecMapUint8Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Int16X(vp *map[uint8]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int16V(v map[uint8]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint8]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]int32)
		v, changed := fastpathTV.DecMapUint8Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]int32)
		fastpathTV.DecMapUint8Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Int32X(vp *map[uint8]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int32V(v map[uint8]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint8]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]int64)
		v, changed := fastpathTV.DecMapUint8Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]int64)
		fastpathTV.DecMapUint8Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Int64X(vp *map[uint8]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int64V(v map[uint8]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]float32)
		v, changed := fastpathTV.DecMapUint8Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]float32)
		fastpathTV.DecMapUint8Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Float32X(vp *map[uint8]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Float32V(v map[uint8]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint8]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]float64)
		v, changed := fastpathTV.DecMapUint8Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]float64)
		fastpathTV.DecMapUint8Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8Float64X(vp *map[uint8]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Float64V(v map[uint8]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint8BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint8]bool)
		v, changed := fastpathTV.DecMapUint8BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint8]bool)
		fastpathTV.DecMapUint8BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint8BoolX(vp *map[uint8]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint8BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8BoolV(v map[uint8]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint8]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[uint8]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint8(dd.DecodeUint(8))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]interface{})
		v, changed := fastpathTV.DecMapUint16IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]interface{})
		fastpathTV.DecMapUint16IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16IntfX(vp *map[uint16]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16IntfV(v map[uint16]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[uint16]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]string)
		v, changed := fastpathTV.DecMapUint16StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]string)
		fastpathTV.DecMapUint16StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16StringX(vp *map[uint16]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16StringV(v map[uint16]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[uint16]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]uint)
		v, changed := fastpathTV.DecMapUint16UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]uint)
		fastpathTV.DecMapUint16UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16UintX(vp *map[uint16]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16UintV(v map[uint16]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]uint8)
		v, changed := fastpathTV.DecMapUint16Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]uint8)
		fastpathTV.DecMapUint16Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Uint8X(vp *map[uint16]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint8V(v map[uint16]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint16]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]uint16)
		v, changed := fastpathTV.DecMapUint16Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]uint16)
		fastpathTV.DecMapUint16Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Uint16X(vp *map[uint16]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint16V(v map[uint16]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[uint16]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]uint32)
		v, changed := fastpathTV.DecMapUint16Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]uint32)
		fastpathTV.DecMapUint16Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Uint32X(vp *map[uint16]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint32V(v map[uint16]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint16]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]uint64)
		v, changed := fastpathTV.DecMapUint16Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]uint64)
		fastpathTV.DecMapUint16Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Uint64X(vp *map[uint16]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint64V(v map[uint16]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]int)
		v, changed := fastpathTV.DecMapUint16IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]int)
		fastpathTV.DecMapUint16IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16IntX(vp *map[uint16]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16IntV(v map[uint16]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]int8)
		v, changed := fastpathTV.DecMapUint16Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]int8)
		fastpathTV.DecMapUint16Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Int8X(vp *map[uint16]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int8V(v map[uint16]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint16]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]int16)
		v, changed := fastpathTV.DecMapUint16Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]int16)
		fastpathTV.DecMapUint16Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Int16X(vp *map[uint16]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int16V(v map[uint16]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[uint16]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]int32)
		v, changed := fastpathTV.DecMapUint16Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]int32)
		fastpathTV.DecMapUint16Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Int32X(vp *map[uint16]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int32V(v map[uint16]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint16]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]int64)
		v, changed := fastpathTV.DecMapUint16Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]int64)
		fastpathTV.DecMapUint16Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Int64X(vp *map[uint16]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int64V(v map[uint16]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]float32)
		v, changed := fastpathTV.DecMapUint16Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]float32)
		fastpathTV.DecMapUint16Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Float32X(vp *map[uint16]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Float32V(v map[uint16]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint16]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]float64)
		v, changed := fastpathTV.DecMapUint16Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]float64)
		fastpathTV.DecMapUint16Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16Float64X(vp *map[uint16]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Float64V(v map[uint16]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint16BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint16]bool)
		v, changed := fastpathTV.DecMapUint16BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint16]bool)
		fastpathTV.DecMapUint16BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint16BoolX(vp *map[uint16]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint16BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16BoolV(v map[uint16]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint16]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint16]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint16(dd.DecodeUint(16))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]interface{})
		v, changed := fastpathTV.DecMapUint32IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]interface{})
		fastpathTV.DecMapUint32IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32IntfX(vp *map[uint32]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32IntfV(v map[uint32]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[uint32]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]string)
		v, changed := fastpathTV.DecMapUint32StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]string)
		fastpathTV.DecMapUint32StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32StringX(vp *map[uint32]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32StringV(v map[uint32]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[uint32]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]uint)
		v, changed := fastpathTV.DecMapUint32UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]uint)
		fastpathTV.DecMapUint32UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32UintX(vp *map[uint32]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32UintV(v map[uint32]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]uint8)
		v, changed := fastpathTV.DecMapUint32Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]uint8)
		fastpathTV.DecMapUint32Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Uint8X(vp *map[uint32]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint8V(v map[uint32]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint32]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]uint16)
		v, changed := fastpathTV.DecMapUint32Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]uint16)
		fastpathTV.DecMapUint32Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Uint16X(vp *map[uint32]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint16V(v map[uint32]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint32]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]uint32)
		v, changed := fastpathTV.DecMapUint32Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]uint32)
		fastpathTV.DecMapUint32Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Uint32X(vp *map[uint32]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint32V(v map[uint32]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[uint32]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]uint64)
		v, changed := fastpathTV.DecMapUint32Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]uint64)
		fastpathTV.DecMapUint32Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Uint64X(vp *map[uint32]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint64V(v map[uint32]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]int)
		v, changed := fastpathTV.DecMapUint32IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]int)
		fastpathTV.DecMapUint32IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32IntX(vp *map[uint32]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32IntV(v map[uint32]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]int8)
		v, changed := fastpathTV.DecMapUint32Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]int8)
		fastpathTV.DecMapUint32Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Int8X(vp *map[uint32]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int8V(v map[uint32]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint32]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]int16)
		v, changed := fastpathTV.DecMapUint32Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]int16)
		fastpathTV.DecMapUint32Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Int16X(vp *map[uint32]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int16V(v map[uint32]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint32]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]int32)
		v, changed := fastpathTV.DecMapUint32Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]int32)
		fastpathTV.DecMapUint32Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Int32X(vp *map[uint32]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int32V(v map[uint32]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[uint32]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]int64)
		v, changed := fastpathTV.DecMapUint32Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]int64)
		fastpathTV.DecMapUint32Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Int64X(vp *map[uint32]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int64V(v map[uint32]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]float32)
		v, changed := fastpathTV.DecMapUint32Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]float32)
		fastpathTV.DecMapUint32Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Float32X(vp *map[uint32]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Float32V(v map[uint32]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[uint32]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]float64)
		v, changed := fastpathTV.DecMapUint32Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]float64)
		fastpathTV.DecMapUint32Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32Float64X(vp *map[uint32]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Float64V(v map[uint32]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint32BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint32]bool)
		v, changed := fastpathTV.DecMapUint32BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint32]bool)
		fastpathTV.DecMapUint32BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint32BoolX(vp *map[uint32]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint32BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32BoolV(v map[uint32]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint32]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint32]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := uint32(dd.DecodeUint(32))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]interface{})
		v, changed := fastpathTV.DecMapUint64IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]interface{})
		fastpathTV.DecMapUint64IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64IntfX(vp *map[uint64]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64IntfV(v map[uint64]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint64]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]string)
		v, changed := fastpathTV.DecMapUint64StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]string)
		fastpathTV.DecMapUint64StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64StringX(vp *map[uint64]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64StringV(v map[uint64]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint64]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]uint)
		v, changed := fastpathTV.DecMapUint64UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]uint)
		fastpathTV.DecMapUint64UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64UintX(vp *map[uint64]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64UintV(v map[uint64]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]uint8)
		v, changed := fastpathTV.DecMapUint64Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]uint8)
		fastpathTV.DecMapUint64Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Uint8X(vp *map[uint64]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint8V(v map[uint64]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint64]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]uint16)
		v, changed := fastpathTV.DecMapUint64Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]uint16)
		fastpathTV.DecMapUint64Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Uint16X(vp *map[uint64]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint16V(v map[uint64]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint64]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]uint32)
		v, changed := fastpathTV.DecMapUint64Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]uint32)
		fastpathTV.DecMapUint64Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Uint32X(vp *map[uint64]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint32V(v map[uint64]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint64]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]uint64)
		v, changed := fastpathTV.DecMapUint64Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]uint64)
		fastpathTV.DecMapUint64Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Uint64X(vp *map[uint64]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint64V(v map[uint64]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]int)
		v, changed := fastpathTV.DecMapUint64IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]int)
		fastpathTV.DecMapUint64IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64IntX(vp *map[uint64]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64IntV(v map[uint64]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]int8)
		v, changed := fastpathTV.DecMapUint64Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]int8)
		fastpathTV.DecMapUint64Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Int8X(vp *map[uint64]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int8V(v map[uint64]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint64]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]int16)
		v, changed := fastpathTV.DecMapUint64Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]int16)
		fastpathTV.DecMapUint64Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Int16X(vp *map[uint64]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int16V(v map[uint64]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint64]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]int32)
		v, changed := fastpathTV.DecMapUint64Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]int32)
		fastpathTV.DecMapUint64Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Int32X(vp *map[uint64]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int32V(v map[uint64]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint64]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]int64)
		v, changed := fastpathTV.DecMapUint64Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]int64)
		fastpathTV.DecMapUint64Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Int64X(vp *map[uint64]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int64V(v map[uint64]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]float32)
		v, changed := fastpathTV.DecMapUint64Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]float32)
		fastpathTV.DecMapUint64Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Float32X(vp *map[uint64]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Float32V(v map[uint64]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint64]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]float64)
		v, changed := fastpathTV.DecMapUint64Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]float64)
		fastpathTV.DecMapUint64Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64Float64X(vp *map[uint64]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Float64V(v map[uint64]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapUint64BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[uint64]bool)
		v, changed := fastpathTV.DecMapUint64BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[uint64]bool)
		fastpathTV.DecMapUint64BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapUint64BoolX(vp *map[uint64]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapUint64BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64BoolV(v map[uint64]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[uint64]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint64]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeUint(64)
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntIntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]interface{})
		v, changed := fastpathTV.DecMapIntIntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]interface{})
		fastpathTV.DecMapIntIntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntIntfX(vp *map[int]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntIntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntIntfV(v map[int]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntStringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]string)
		v, changed := fastpathTV.DecMapIntStringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]string)
		fastpathTV.DecMapIntStringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntStringX(vp *map[int]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntStringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntStringV(v map[int]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntUintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]uint)
		v, changed := fastpathTV.DecMapIntUintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]uint)
		fastpathTV.DecMapIntUintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntUintX(vp *map[int]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntUintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUintV(v map[int]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntUint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]uint8)
		v, changed := fastpathTV.DecMapIntUint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]uint8)
		fastpathTV.DecMapIntUint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntUint8X(vp *map[int]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntUint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint8V(v map[int]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntUint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]uint16)
		v, changed := fastpathTV.DecMapIntUint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]uint16)
		fastpathTV.DecMapIntUint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntUint16X(vp *map[int]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntUint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint16V(v map[int]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntUint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]uint32)
		v, changed := fastpathTV.DecMapIntUint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]uint32)
		fastpathTV.DecMapIntUint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntUint32X(vp *map[int]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntUint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint32V(v map[int]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntUint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]uint64)
		v, changed := fastpathTV.DecMapIntUint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]uint64)
		fastpathTV.DecMapIntUint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntUint64X(vp *map[int]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntUint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint64V(v map[int]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntIntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]int)
		v, changed := fastpathTV.DecMapIntIntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]int)
		fastpathTV.DecMapIntIntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntIntX(vp *map[int]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntIntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntIntV(v map[int]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntInt8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]int8)
		v, changed := fastpathTV.DecMapIntInt8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]int8)
		fastpathTV.DecMapIntInt8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntInt8X(vp *map[int]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntInt8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt8V(v map[int]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntInt16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]int16)
		v, changed := fastpathTV.DecMapIntInt16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]int16)
		fastpathTV.DecMapIntInt16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntInt16X(vp *map[int]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntInt16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt16V(v map[int]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntInt32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]int32)
		v, changed := fastpathTV.DecMapIntInt32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]int32)
		fastpathTV.DecMapIntInt32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntInt32X(vp *map[int]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntInt32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt32V(v map[int]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntInt64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]int64)
		v, changed := fastpathTV.DecMapIntInt64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]int64)
		fastpathTV.DecMapIntInt64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntInt64X(vp *map[int]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntInt64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt64V(v map[int]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntFloat32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]float32)
		v, changed := fastpathTV.DecMapIntFloat32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]float32)
		fastpathTV.DecMapIntFloat32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntFloat32X(vp *map[int]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntFloat32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntFloat32V(v map[int]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntFloat64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]float64)
		v, changed := fastpathTV.DecMapIntFloat64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]float64)
		fastpathTV.DecMapIntFloat64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntFloat64X(vp *map[int]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntFloat64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntFloat64V(v map[int]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapIntBoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int]bool)
		v, changed := fastpathTV.DecMapIntBoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int]bool)
		fastpathTV.DecMapIntBoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapIntBoolX(vp *map[int]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapIntBoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntBoolV(v map[int]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[int]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int(dd.DecodeInt(intBitsize))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]interface{})
		v, changed := fastpathTV.DecMapInt8IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]interface{})
		fastpathTV.DecMapInt8IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8IntfX(vp *map[int8]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8IntfV(v map[int8]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[int8]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]string)
		v, changed := fastpathTV.DecMapInt8StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]string)
		fastpathTV.DecMapInt8StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8StringX(vp *map[int8]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8StringV(v map[int8]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[int8]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]uint)
		v, changed := fastpathTV.DecMapInt8UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]uint)
		fastpathTV.DecMapInt8UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8UintX(vp *map[int8]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8UintV(v map[int8]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]uint8)
		v, changed := fastpathTV.DecMapInt8Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]uint8)
		fastpathTV.DecMapInt8Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Uint8X(vp *map[int8]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint8V(v map[int8]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[int8]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]uint16)
		v, changed := fastpathTV.DecMapInt8Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]uint16)
		fastpathTV.DecMapInt8Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Uint16X(vp *map[int8]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint16V(v map[int8]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int8]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]uint32)
		v, changed := fastpathTV.DecMapInt8Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]uint32)
		fastpathTV.DecMapInt8Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Uint32X(vp *map[int8]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint32V(v map[int8]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int8]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]uint64)
		v, changed := fastpathTV.DecMapInt8Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]uint64)
		fastpathTV.DecMapInt8Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Uint64X(vp *map[int8]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint64V(v map[int8]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]int)
		v, changed := fastpathTV.DecMapInt8IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]int)
		fastpathTV.DecMapInt8IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8IntX(vp *map[int8]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8IntV(v map[int8]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]int8)
		v, changed := fastpathTV.DecMapInt8Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]int8)
		fastpathTV.DecMapInt8Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Int8X(vp *map[int8]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int8V(v map[int8]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[int8]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]int16)
		v, changed := fastpathTV.DecMapInt8Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]int16)
		fastpathTV.DecMapInt8Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Int16X(vp *map[int8]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int16V(v map[int8]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int8]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]int32)
		v, changed := fastpathTV.DecMapInt8Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]int32)
		fastpathTV.DecMapInt8Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Int32X(vp *map[int8]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int32V(v map[int8]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int8]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]int64)
		v, changed := fastpathTV.DecMapInt8Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]int64)
		fastpathTV.DecMapInt8Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Int64X(vp *map[int8]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int64V(v map[int8]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]float32)
		v, changed := fastpathTV.DecMapInt8Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]float32)
		fastpathTV.DecMapInt8Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Float32X(vp *map[int8]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Float32V(v map[int8]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int8]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]float64)
		v, changed := fastpathTV.DecMapInt8Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]float64)
		fastpathTV.DecMapInt8Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8Float64X(vp *map[int8]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Float64V(v map[int8]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt8BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int8]bool)
		v, changed := fastpathTV.DecMapInt8BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int8]bool)
		fastpathTV.DecMapInt8BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt8BoolX(vp *map[int8]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt8BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8BoolV(v map[int8]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[int8]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[int8]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int8(dd.DecodeInt(8))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]interface{})
		v, changed := fastpathTV.DecMapInt16IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]interface{})
		fastpathTV.DecMapInt16IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16IntfX(vp *map[int16]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16IntfV(v map[int16]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[int16]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]string)
		v, changed := fastpathTV.DecMapInt16StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]string)
		fastpathTV.DecMapInt16StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16StringX(vp *map[int16]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16StringV(v map[int16]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[int16]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]uint)
		v, changed := fastpathTV.DecMapInt16UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]uint)
		fastpathTV.DecMapInt16UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16UintX(vp *map[int16]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16UintV(v map[int16]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]uint8)
		v, changed := fastpathTV.DecMapInt16Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]uint8)
		fastpathTV.DecMapInt16Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Uint8X(vp *map[int16]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint8V(v map[int16]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int16]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]uint16)
		v, changed := fastpathTV.DecMapInt16Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]uint16)
		fastpathTV.DecMapInt16Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Uint16X(vp *map[int16]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint16V(v map[int16]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[int16]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]uint32)
		v, changed := fastpathTV.DecMapInt16Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]uint32)
		fastpathTV.DecMapInt16Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Uint32X(vp *map[int16]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint32V(v map[int16]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int16]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]uint64)
		v, changed := fastpathTV.DecMapInt16Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]uint64)
		fastpathTV.DecMapInt16Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Uint64X(vp *map[int16]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint64V(v map[int16]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]int)
		v, changed := fastpathTV.DecMapInt16IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]int)
		fastpathTV.DecMapInt16IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16IntX(vp *map[int16]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16IntV(v map[int16]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]int8)
		v, changed := fastpathTV.DecMapInt16Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]int8)
		fastpathTV.DecMapInt16Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Int8X(vp *map[int16]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int8V(v map[int16]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int16]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]int16)
		v, changed := fastpathTV.DecMapInt16Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]int16)
		fastpathTV.DecMapInt16Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Int16X(vp *map[int16]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int16V(v map[int16]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[int16]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]int32)
		v, changed := fastpathTV.DecMapInt16Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]int32)
		fastpathTV.DecMapInt16Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Int32X(vp *map[int16]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int32V(v map[int16]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int16]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]int64)
		v, changed := fastpathTV.DecMapInt16Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]int64)
		fastpathTV.DecMapInt16Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Int64X(vp *map[int16]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int64V(v map[int16]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]float32)
		v, changed := fastpathTV.DecMapInt16Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]float32)
		fastpathTV.DecMapInt16Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Float32X(vp *map[int16]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Float32V(v map[int16]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int16]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]float64)
		v, changed := fastpathTV.DecMapInt16Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]float64)
		fastpathTV.DecMapInt16Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16Float64X(vp *map[int16]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Float64V(v map[int16]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt16BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int16]bool)
		v, changed := fastpathTV.DecMapInt16BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int16]bool)
		fastpathTV.DecMapInt16BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt16BoolX(vp *map[int16]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt16BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16BoolV(v map[int16]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[int16]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int16]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int16(dd.DecodeInt(16))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]interface{})
		v, changed := fastpathTV.DecMapInt32IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]interface{})
		fastpathTV.DecMapInt32IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32IntfX(vp *map[int32]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32IntfV(v map[int32]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[int32]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]string)
		v, changed := fastpathTV.DecMapInt32StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]string)
		fastpathTV.DecMapInt32StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32StringX(vp *map[int32]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32StringV(v map[int32]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[int32]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]uint)
		v, changed := fastpathTV.DecMapInt32UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]uint)
		fastpathTV.DecMapInt32UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32UintX(vp *map[int32]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32UintV(v map[int32]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]uint8)
		v, changed := fastpathTV.DecMapInt32Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]uint8)
		fastpathTV.DecMapInt32Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Uint8X(vp *map[int32]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint8V(v map[int32]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int32]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]uint16)
		v, changed := fastpathTV.DecMapInt32Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]uint16)
		fastpathTV.DecMapInt32Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Uint16X(vp *map[int32]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint16V(v map[int32]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int32]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]uint32)
		v, changed := fastpathTV.DecMapInt32Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]uint32)
		fastpathTV.DecMapInt32Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Uint32X(vp *map[int32]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint32V(v map[int32]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[int32]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]uint64)
		v, changed := fastpathTV.DecMapInt32Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]uint64)
		fastpathTV.DecMapInt32Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Uint64X(vp *map[int32]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint64V(v map[int32]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]int)
		v, changed := fastpathTV.DecMapInt32IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]int)
		fastpathTV.DecMapInt32IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32IntX(vp *map[int32]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32IntV(v map[int32]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]int8)
		v, changed := fastpathTV.DecMapInt32Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]int8)
		fastpathTV.DecMapInt32Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Int8X(vp *map[int32]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int8V(v map[int32]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int32]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]int16)
		v, changed := fastpathTV.DecMapInt32Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]int16)
		fastpathTV.DecMapInt32Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Int16X(vp *map[int32]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int16V(v map[int32]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int32]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]int32)
		v, changed := fastpathTV.DecMapInt32Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]int32)
		fastpathTV.DecMapInt32Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Int32X(vp *map[int32]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int32V(v map[int32]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[int32]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]int64)
		v, changed := fastpathTV.DecMapInt32Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]int64)
		fastpathTV.DecMapInt32Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Int64X(vp *map[int32]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int64V(v map[int32]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]float32)
		v, changed := fastpathTV.DecMapInt32Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]float32)
		fastpathTV.DecMapInt32Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Float32X(vp *map[int32]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Float32V(v map[int32]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[int32]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]float64)
		v, changed := fastpathTV.DecMapInt32Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]float64)
		fastpathTV.DecMapInt32Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32Float64X(vp *map[int32]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Float64V(v map[int32]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt32BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int32]bool)
		v, changed := fastpathTV.DecMapInt32BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int32]bool)
		fastpathTV.DecMapInt32BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt32BoolX(vp *map[int32]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt32BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32BoolV(v map[int32]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[int32]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int32]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := int32(dd.DecodeInt(32))
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64IntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]interface{})
		v, changed := fastpathTV.DecMapInt64IntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]interface{})
		fastpathTV.DecMapInt64IntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64IntfX(vp *map[int64]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64IntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64IntfV(v map[int64]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int64]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64StringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]string)
		v, changed := fastpathTV.DecMapInt64StringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]string)
		fastpathTV.DecMapInt64StringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64StringX(vp *map[int64]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64StringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64StringV(v map[int64]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int64]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64UintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]uint)
		v, changed := fastpathTV.DecMapInt64UintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]uint)
		fastpathTV.DecMapInt64UintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64UintX(vp *map[int64]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64UintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64UintV(v map[int64]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Uint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]uint8)
		v, changed := fastpathTV.DecMapInt64Uint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]uint8)
		fastpathTV.DecMapInt64Uint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Uint8X(vp *map[int64]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Uint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint8V(v map[int64]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int64]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Uint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]uint16)
		v, changed := fastpathTV.DecMapInt64Uint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]uint16)
		fastpathTV.DecMapInt64Uint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Uint16X(vp *map[int64]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Uint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint16V(v map[int64]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int64]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Uint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]uint32)
		v, changed := fastpathTV.DecMapInt64Uint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]uint32)
		fastpathTV.DecMapInt64Uint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Uint32X(vp *map[int64]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Uint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint32V(v map[int64]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int64]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Uint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]uint64)
		v, changed := fastpathTV.DecMapInt64Uint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]uint64)
		fastpathTV.DecMapInt64Uint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Uint64X(vp *map[int64]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Uint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint64V(v map[int64]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64IntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]int)
		v, changed := fastpathTV.DecMapInt64IntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]int)
		fastpathTV.DecMapInt64IntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64IntX(vp *map[int64]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64IntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64IntV(v map[int64]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Int8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]int8)
		v, changed := fastpathTV.DecMapInt64Int8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]int8)
		fastpathTV.DecMapInt64Int8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Int8X(vp *map[int64]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Int8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int8V(v map[int64]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int64]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Int16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]int16)
		v, changed := fastpathTV.DecMapInt64Int16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]int16)
		fastpathTV.DecMapInt64Int16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Int16X(vp *map[int64]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Int16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int16V(v map[int64]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int64]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Int32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]int32)
		v, changed := fastpathTV.DecMapInt64Int32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]int32)
		fastpathTV.DecMapInt64Int32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Int32X(vp *map[int64]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Int32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int32V(v map[int64]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int64]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Int64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]int64)
		v, changed := fastpathTV.DecMapInt64Int64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]int64)
		fastpathTV.DecMapInt64Int64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Int64X(vp *map[int64]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Int64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int64V(v map[int64]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Float32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]float32)
		v, changed := fastpathTV.DecMapInt64Float32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]float32)
		fastpathTV.DecMapInt64Float32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Float32X(vp *map[int64]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Float32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Float32V(v map[int64]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int64]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64Float64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]float64)
		v, changed := fastpathTV.DecMapInt64Float64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]float64)
		fastpathTV.DecMapInt64Float64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64Float64X(vp *map[int64]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64Float64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Float64V(v map[int64]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapInt64BoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[int64]bool)
		v, changed := fastpathTV.DecMapInt64BoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[int64]bool)
		fastpathTV.DecMapInt64BoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapInt64BoolX(vp *map[int64]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapInt64BoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64BoolV(v map[int64]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[int64]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int64]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeInt(64)
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolIntfR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]interface{})
		v, changed := fastpathTV.DecMapBoolIntfV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]interface{})
		fastpathTV.DecMapBoolIntfV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolIntfX(vp *map[bool]interface{}, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolIntfV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolIntfV(v map[bool]interface{}, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]interface{}, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[bool]interface{}, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			d.decode(&mv)

			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolStringR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]string)
		v, changed := fastpathTV.DecMapBoolStringV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]string)
		fastpathTV.DecMapBoolStringV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolStringX(vp *map[bool]string, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolStringV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolStringV(v map[bool]string, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]string, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[bool]string, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeString()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolUintR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]uint)
		v, changed := fastpathTV.DecMapBoolUintV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]uint)
		fastpathTV.DecMapBoolUintV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolUintX(vp *map[bool]uint, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolUintV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUintV(v map[bool]uint, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]uint, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]uint, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint(dd.DecodeUint(uintBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolUint8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]uint8)
		v, changed := fastpathTV.DecMapBoolUint8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]uint8)
		fastpathTV.DecMapBoolUint8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolUint8X(vp *map[bool]uint8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolUint8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint8V(v map[bool]uint8, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]uint8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[bool]uint8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint8(dd.DecodeUint(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolUint16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]uint16)
		v, changed := fastpathTV.DecMapBoolUint16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]uint16)
		fastpathTV.DecMapBoolUint16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolUint16X(vp *map[bool]uint16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolUint16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint16V(v map[bool]uint16, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]uint16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[bool]uint16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint16(dd.DecodeUint(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolUint32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]uint32)
		v, changed := fastpathTV.DecMapBoolUint32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]uint32)
		fastpathTV.DecMapBoolUint32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolUint32X(vp *map[bool]uint32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolUint32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint32V(v map[bool]uint32, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]uint32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[bool]uint32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = uint32(dd.DecodeUint(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolUint64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]uint64)
		v, changed := fastpathTV.DecMapBoolUint64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]uint64)
		fastpathTV.DecMapBoolUint64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolUint64X(vp *map[bool]uint64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolUint64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint64V(v map[bool]uint64, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]uint64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]uint64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeUint(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolIntR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]int)
		v, changed := fastpathTV.DecMapBoolIntV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]int)
		fastpathTV.DecMapBoolIntV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolIntX(vp *map[bool]int, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolIntV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolIntV(v map[bool]int, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]int, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]int, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int(dd.DecodeInt(intBitsize))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolInt8R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]int8)
		v, changed := fastpathTV.DecMapBoolInt8V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]int8)
		fastpathTV.DecMapBoolInt8V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolInt8X(vp *map[bool]int8, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolInt8V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt8V(v map[bool]int8, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]int8, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[bool]int8, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int8(dd.DecodeInt(8))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolInt16R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]int16)
		v, changed := fastpathTV.DecMapBoolInt16V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]int16)
		fastpathTV.DecMapBoolInt16V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolInt16X(vp *map[bool]int16, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolInt16V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt16V(v map[bool]int16, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]int16, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[bool]int16, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int16(dd.DecodeInt(16))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolInt32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]int32)
		v, changed := fastpathTV.DecMapBoolInt32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]int32)
		fastpathTV.DecMapBoolInt32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolInt32X(vp *map[bool]int32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolInt32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt32V(v map[bool]int32, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]int32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[bool]int32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = int32(dd.DecodeInt(32))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolInt64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]int64)
		v, changed := fastpathTV.DecMapBoolInt64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]int64)
		fastpathTV.DecMapBoolInt64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolInt64X(vp *map[bool]int64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolInt64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt64V(v map[bool]int64, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]int64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]int64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeInt(64)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolFloat32R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]float32)
		v, changed := fastpathTV.DecMapBoolFloat32V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]float32)
		fastpathTV.DecMapBoolFloat32V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolFloat32X(vp *map[bool]float32, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolFloat32V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolFloat32V(v map[bool]float32, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]float32, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[bool]float32, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = float32(dd.DecodeFloat(true))
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolFloat64R(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]float64)
		v, changed := fastpathTV.DecMapBoolFloat64V(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]float64)
		fastpathTV.DecMapBoolFloat64V(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolFloat64X(vp *map[bool]float64, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolFloat64V(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolFloat64V(v map[bool]float64, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]float64, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]float64, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeFloat(false)
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}

func (f *decFnInfo) fastpathDecMapBoolBoolR(rv reflect.Value) {
	if rv.CanAddr() {
		vp := rv.Addr().Interface().(*map[bool]bool)
		v, changed := fastpathTV.DecMapBoolBoolV(*vp, fastpathCheckNilFalse, true, f.d)
		if changed {
			*vp = v
		}
	} else {
		v := rv.Interface().(map[bool]bool)
		fastpathTV.DecMapBoolBoolV(v, fastpathCheckNilFalse, false, f.d)
	}
}
func (f fastpathT) DecMapBoolBoolX(vp *map[bool]bool, checkNil bool, d *Decoder) {
	v, changed := f.DecMapBoolBoolV(*vp, checkNil, true, d)
	if changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolBoolV(v map[bool]bool, checkNil bool, canChange bool,
	d *Decoder) (_ map[bool]bool, changed bool) {
	dd := d.d

	if checkNil && dd.TryDecodeAsNil() {
		if v != nil {
			changed = true
		}
		return nil, changed
	}

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen, _ := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[bool]bool, xlen)
		changed = true
	}
	if containerLen > 0 {
		for j := 0; j < containerLen; j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
	} else if containerLen < 0 {
		for j := 0; !dd.CheckBreak(); j++ {
			mk := dd.DecodeBool()
			mv := v[mk]
			mv = dd.DecodeBool()
			if v != nil {
				v[mk] = mv
			}
		}
		dd.ReadEnd()
	}
	return v, changed
}
