// +build !notfastpath

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
//    - symmetrical maps of all builtin types (e.g. str-str, uint8-uint8)
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

const fastpathEnabled = true

type fastpathT struct{}

var fastpathTV fastpathT

type fastpathE struct {
	rtid  uintptr
	rt    reflect.Type
	encfn func(*Encoder, *codecFnInfo, reflect.Value)
	decfn func(*Decoder, *codecFnInfo, reflect.Value)
}

type fastpathA [271]fastpathE

func (x *fastpathA) index(rtid uintptr) int {
	// use binary search to grab the index (adapted from sort/search.go)
	h, i, j := 0, 0, 271 // len(x)
	for i < j {
		h = i + (j-i)/2
		if x[h].rtid < rtid {
			i = h + 1
		} else {
			j = h
		}
	}
	if i < 271 && x[i].rtid == rtid {
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
	i := 0
	fn := func(v interface{},
		fe func(*Encoder, *codecFnInfo, reflect.Value),
		fd func(*Decoder, *codecFnInfo, reflect.Value)) (f fastpathE) {
		xrt := reflect.TypeOf(v)
		xptr := rt2id(xrt)
		if useLookupRecognizedTypes {
			recognizedRtids = append(recognizedRtids, xptr)
			recognizedRtidPtrs = append(recognizedRtidPtrs, rt2id(reflect.PtrTo(xrt)))
		}
		fastpathAV[i] = fastpathE{xptr, xrt, fe, fd}
		i++
		return
	}

	fn([]interface{}(nil), (*Encoder).fastpathEncSliceIntfR, (*Decoder).fastpathDecSliceIntfR)
	fn([]string(nil), (*Encoder).fastpathEncSliceStringR, (*Decoder).fastpathDecSliceStringR)
	fn([]float32(nil), (*Encoder).fastpathEncSliceFloat32R, (*Decoder).fastpathDecSliceFloat32R)
	fn([]float64(nil), (*Encoder).fastpathEncSliceFloat64R, (*Decoder).fastpathDecSliceFloat64R)
	fn([]uint(nil), (*Encoder).fastpathEncSliceUintR, (*Decoder).fastpathDecSliceUintR)
	fn([]uint16(nil), (*Encoder).fastpathEncSliceUint16R, (*Decoder).fastpathDecSliceUint16R)
	fn([]uint32(nil), (*Encoder).fastpathEncSliceUint32R, (*Decoder).fastpathDecSliceUint32R)
	fn([]uint64(nil), (*Encoder).fastpathEncSliceUint64R, (*Decoder).fastpathDecSliceUint64R)
	fn([]uintptr(nil), (*Encoder).fastpathEncSliceUintptrR, (*Decoder).fastpathDecSliceUintptrR)
	fn([]int(nil), (*Encoder).fastpathEncSliceIntR, (*Decoder).fastpathDecSliceIntR)
	fn([]int8(nil), (*Encoder).fastpathEncSliceInt8R, (*Decoder).fastpathDecSliceInt8R)
	fn([]int16(nil), (*Encoder).fastpathEncSliceInt16R, (*Decoder).fastpathDecSliceInt16R)
	fn([]int32(nil), (*Encoder).fastpathEncSliceInt32R, (*Decoder).fastpathDecSliceInt32R)
	fn([]int64(nil), (*Encoder).fastpathEncSliceInt64R, (*Decoder).fastpathDecSliceInt64R)
	fn([]bool(nil), (*Encoder).fastpathEncSliceBoolR, (*Decoder).fastpathDecSliceBoolR)

	fn(map[interface{}]interface{}(nil), (*Encoder).fastpathEncMapIntfIntfR, (*Decoder).fastpathDecMapIntfIntfR)
	fn(map[interface{}]string(nil), (*Encoder).fastpathEncMapIntfStringR, (*Decoder).fastpathDecMapIntfStringR)
	fn(map[interface{}]uint(nil), (*Encoder).fastpathEncMapIntfUintR, (*Decoder).fastpathDecMapIntfUintR)
	fn(map[interface{}]uint8(nil), (*Encoder).fastpathEncMapIntfUint8R, (*Decoder).fastpathDecMapIntfUint8R)
	fn(map[interface{}]uint16(nil), (*Encoder).fastpathEncMapIntfUint16R, (*Decoder).fastpathDecMapIntfUint16R)
	fn(map[interface{}]uint32(nil), (*Encoder).fastpathEncMapIntfUint32R, (*Decoder).fastpathDecMapIntfUint32R)
	fn(map[interface{}]uint64(nil), (*Encoder).fastpathEncMapIntfUint64R, (*Decoder).fastpathDecMapIntfUint64R)
	fn(map[interface{}]uintptr(nil), (*Encoder).fastpathEncMapIntfUintptrR, (*Decoder).fastpathDecMapIntfUintptrR)
	fn(map[interface{}]int(nil), (*Encoder).fastpathEncMapIntfIntR, (*Decoder).fastpathDecMapIntfIntR)
	fn(map[interface{}]int8(nil), (*Encoder).fastpathEncMapIntfInt8R, (*Decoder).fastpathDecMapIntfInt8R)
	fn(map[interface{}]int16(nil), (*Encoder).fastpathEncMapIntfInt16R, (*Decoder).fastpathDecMapIntfInt16R)
	fn(map[interface{}]int32(nil), (*Encoder).fastpathEncMapIntfInt32R, (*Decoder).fastpathDecMapIntfInt32R)
	fn(map[interface{}]int64(nil), (*Encoder).fastpathEncMapIntfInt64R, (*Decoder).fastpathDecMapIntfInt64R)
	fn(map[interface{}]float32(nil), (*Encoder).fastpathEncMapIntfFloat32R, (*Decoder).fastpathDecMapIntfFloat32R)
	fn(map[interface{}]float64(nil), (*Encoder).fastpathEncMapIntfFloat64R, (*Decoder).fastpathDecMapIntfFloat64R)
	fn(map[interface{}]bool(nil), (*Encoder).fastpathEncMapIntfBoolR, (*Decoder).fastpathDecMapIntfBoolR)
	fn(map[string]interface{}(nil), (*Encoder).fastpathEncMapStringIntfR, (*Decoder).fastpathDecMapStringIntfR)
	fn(map[string]string(nil), (*Encoder).fastpathEncMapStringStringR, (*Decoder).fastpathDecMapStringStringR)
	fn(map[string]uint(nil), (*Encoder).fastpathEncMapStringUintR, (*Decoder).fastpathDecMapStringUintR)
	fn(map[string]uint8(nil), (*Encoder).fastpathEncMapStringUint8R, (*Decoder).fastpathDecMapStringUint8R)
	fn(map[string]uint16(nil), (*Encoder).fastpathEncMapStringUint16R, (*Decoder).fastpathDecMapStringUint16R)
	fn(map[string]uint32(nil), (*Encoder).fastpathEncMapStringUint32R, (*Decoder).fastpathDecMapStringUint32R)
	fn(map[string]uint64(nil), (*Encoder).fastpathEncMapStringUint64R, (*Decoder).fastpathDecMapStringUint64R)
	fn(map[string]uintptr(nil), (*Encoder).fastpathEncMapStringUintptrR, (*Decoder).fastpathDecMapStringUintptrR)
	fn(map[string]int(nil), (*Encoder).fastpathEncMapStringIntR, (*Decoder).fastpathDecMapStringIntR)
	fn(map[string]int8(nil), (*Encoder).fastpathEncMapStringInt8R, (*Decoder).fastpathDecMapStringInt8R)
	fn(map[string]int16(nil), (*Encoder).fastpathEncMapStringInt16R, (*Decoder).fastpathDecMapStringInt16R)
	fn(map[string]int32(nil), (*Encoder).fastpathEncMapStringInt32R, (*Decoder).fastpathDecMapStringInt32R)
	fn(map[string]int64(nil), (*Encoder).fastpathEncMapStringInt64R, (*Decoder).fastpathDecMapStringInt64R)
	fn(map[string]float32(nil), (*Encoder).fastpathEncMapStringFloat32R, (*Decoder).fastpathDecMapStringFloat32R)
	fn(map[string]float64(nil), (*Encoder).fastpathEncMapStringFloat64R, (*Decoder).fastpathDecMapStringFloat64R)
	fn(map[string]bool(nil), (*Encoder).fastpathEncMapStringBoolR, (*Decoder).fastpathDecMapStringBoolR)
	fn(map[float32]interface{}(nil), (*Encoder).fastpathEncMapFloat32IntfR, (*Decoder).fastpathDecMapFloat32IntfR)
	fn(map[float32]string(nil), (*Encoder).fastpathEncMapFloat32StringR, (*Decoder).fastpathDecMapFloat32StringR)
	fn(map[float32]uint(nil), (*Encoder).fastpathEncMapFloat32UintR, (*Decoder).fastpathDecMapFloat32UintR)
	fn(map[float32]uint8(nil), (*Encoder).fastpathEncMapFloat32Uint8R, (*Decoder).fastpathDecMapFloat32Uint8R)
	fn(map[float32]uint16(nil), (*Encoder).fastpathEncMapFloat32Uint16R, (*Decoder).fastpathDecMapFloat32Uint16R)
	fn(map[float32]uint32(nil), (*Encoder).fastpathEncMapFloat32Uint32R, (*Decoder).fastpathDecMapFloat32Uint32R)
	fn(map[float32]uint64(nil), (*Encoder).fastpathEncMapFloat32Uint64R, (*Decoder).fastpathDecMapFloat32Uint64R)
	fn(map[float32]uintptr(nil), (*Encoder).fastpathEncMapFloat32UintptrR, (*Decoder).fastpathDecMapFloat32UintptrR)
	fn(map[float32]int(nil), (*Encoder).fastpathEncMapFloat32IntR, (*Decoder).fastpathDecMapFloat32IntR)
	fn(map[float32]int8(nil), (*Encoder).fastpathEncMapFloat32Int8R, (*Decoder).fastpathDecMapFloat32Int8R)
	fn(map[float32]int16(nil), (*Encoder).fastpathEncMapFloat32Int16R, (*Decoder).fastpathDecMapFloat32Int16R)
	fn(map[float32]int32(nil), (*Encoder).fastpathEncMapFloat32Int32R, (*Decoder).fastpathDecMapFloat32Int32R)
	fn(map[float32]int64(nil), (*Encoder).fastpathEncMapFloat32Int64R, (*Decoder).fastpathDecMapFloat32Int64R)
	fn(map[float32]float32(nil), (*Encoder).fastpathEncMapFloat32Float32R, (*Decoder).fastpathDecMapFloat32Float32R)
	fn(map[float32]float64(nil), (*Encoder).fastpathEncMapFloat32Float64R, (*Decoder).fastpathDecMapFloat32Float64R)
	fn(map[float32]bool(nil), (*Encoder).fastpathEncMapFloat32BoolR, (*Decoder).fastpathDecMapFloat32BoolR)
	fn(map[float64]interface{}(nil), (*Encoder).fastpathEncMapFloat64IntfR, (*Decoder).fastpathDecMapFloat64IntfR)
	fn(map[float64]string(nil), (*Encoder).fastpathEncMapFloat64StringR, (*Decoder).fastpathDecMapFloat64StringR)
	fn(map[float64]uint(nil), (*Encoder).fastpathEncMapFloat64UintR, (*Decoder).fastpathDecMapFloat64UintR)
	fn(map[float64]uint8(nil), (*Encoder).fastpathEncMapFloat64Uint8R, (*Decoder).fastpathDecMapFloat64Uint8R)
	fn(map[float64]uint16(nil), (*Encoder).fastpathEncMapFloat64Uint16R, (*Decoder).fastpathDecMapFloat64Uint16R)
	fn(map[float64]uint32(nil), (*Encoder).fastpathEncMapFloat64Uint32R, (*Decoder).fastpathDecMapFloat64Uint32R)
	fn(map[float64]uint64(nil), (*Encoder).fastpathEncMapFloat64Uint64R, (*Decoder).fastpathDecMapFloat64Uint64R)
	fn(map[float64]uintptr(nil), (*Encoder).fastpathEncMapFloat64UintptrR, (*Decoder).fastpathDecMapFloat64UintptrR)
	fn(map[float64]int(nil), (*Encoder).fastpathEncMapFloat64IntR, (*Decoder).fastpathDecMapFloat64IntR)
	fn(map[float64]int8(nil), (*Encoder).fastpathEncMapFloat64Int8R, (*Decoder).fastpathDecMapFloat64Int8R)
	fn(map[float64]int16(nil), (*Encoder).fastpathEncMapFloat64Int16R, (*Decoder).fastpathDecMapFloat64Int16R)
	fn(map[float64]int32(nil), (*Encoder).fastpathEncMapFloat64Int32R, (*Decoder).fastpathDecMapFloat64Int32R)
	fn(map[float64]int64(nil), (*Encoder).fastpathEncMapFloat64Int64R, (*Decoder).fastpathDecMapFloat64Int64R)
	fn(map[float64]float32(nil), (*Encoder).fastpathEncMapFloat64Float32R, (*Decoder).fastpathDecMapFloat64Float32R)
	fn(map[float64]float64(nil), (*Encoder).fastpathEncMapFloat64Float64R, (*Decoder).fastpathDecMapFloat64Float64R)
	fn(map[float64]bool(nil), (*Encoder).fastpathEncMapFloat64BoolR, (*Decoder).fastpathDecMapFloat64BoolR)
	fn(map[uint]interface{}(nil), (*Encoder).fastpathEncMapUintIntfR, (*Decoder).fastpathDecMapUintIntfR)
	fn(map[uint]string(nil), (*Encoder).fastpathEncMapUintStringR, (*Decoder).fastpathDecMapUintStringR)
	fn(map[uint]uint(nil), (*Encoder).fastpathEncMapUintUintR, (*Decoder).fastpathDecMapUintUintR)
	fn(map[uint]uint8(nil), (*Encoder).fastpathEncMapUintUint8R, (*Decoder).fastpathDecMapUintUint8R)
	fn(map[uint]uint16(nil), (*Encoder).fastpathEncMapUintUint16R, (*Decoder).fastpathDecMapUintUint16R)
	fn(map[uint]uint32(nil), (*Encoder).fastpathEncMapUintUint32R, (*Decoder).fastpathDecMapUintUint32R)
	fn(map[uint]uint64(nil), (*Encoder).fastpathEncMapUintUint64R, (*Decoder).fastpathDecMapUintUint64R)
	fn(map[uint]uintptr(nil), (*Encoder).fastpathEncMapUintUintptrR, (*Decoder).fastpathDecMapUintUintptrR)
	fn(map[uint]int(nil), (*Encoder).fastpathEncMapUintIntR, (*Decoder).fastpathDecMapUintIntR)
	fn(map[uint]int8(nil), (*Encoder).fastpathEncMapUintInt8R, (*Decoder).fastpathDecMapUintInt8R)
	fn(map[uint]int16(nil), (*Encoder).fastpathEncMapUintInt16R, (*Decoder).fastpathDecMapUintInt16R)
	fn(map[uint]int32(nil), (*Encoder).fastpathEncMapUintInt32R, (*Decoder).fastpathDecMapUintInt32R)
	fn(map[uint]int64(nil), (*Encoder).fastpathEncMapUintInt64R, (*Decoder).fastpathDecMapUintInt64R)
	fn(map[uint]float32(nil), (*Encoder).fastpathEncMapUintFloat32R, (*Decoder).fastpathDecMapUintFloat32R)
	fn(map[uint]float64(nil), (*Encoder).fastpathEncMapUintFloat64R, (*Decoder).fastpathDecMapUintFloat64R)
	fn(map[uint]bool(nil), (*Encoder).fastpathEncMapUintBoolR, (*Decoder).fastpathDecMapUintBoolR)
	fn(map[uint8]interface{}(nil), (*Encoder).fastpathEncMapUint8IntfR, (*Decoder).fastpathDecMapUint8IntfR)
	fn(map[uint8]string(nil), (*Encoder).fastpathEncMapUint8StringR, (*Decoder).fastpathDecMapUint8StringR)
	fn(map[uint8]uint(nil), (*Encoder).fastpathEncMapUint8UintR, (*Decoder).fastpathDecMapUint8UintR)
	fn(map[uint8]uint8(nil), (*Encoder).fastpathEncMapUint8Uint8R, (*Decoder).fastpathDecMapUint8Uint8R)
	fn(map[uint8]uint16(nil), (*Encoder).fastpathEncMapUint8Uint16R, (*Decoder).fastpathDecMapUint8Uint16R)
	fn(map[uint8]uint32(nil), (*Encoder).fastpathEncMapUint8Uint32R, (*Decoder).fastpathDecMapUint8Uint32R)
	fn(map[uint8]uint64(nil), (*Encoder).fastpathEncMapUint8Uint64R, (*Decoder).fastpathDecMapUint8Uint64R)
	fn(map[uint8]uintptr(nil), (*Encoder).fastpathEncMapUint8UintptrR, (*Decoder).fastpathDecMapUint8UintptrR)
	fn(map[uint8]int(nil), (*Encoder).fastpathEncMapUint8IntR, (*Decoder).fastpathDecMapUint8IntR)
	fn(map[uint8]int8(nil), (*Encoder).fastpathEncMapUint8Int8R, (*Decoder).fastpathDecMapUint8Int8R)
	fn(map[uint8]int16(nil), (*Encoder).fastpathEncMapUint8Int16R, (*Decoder).fastpathDecMapUint8Int16R)
	fn(map[uint8]int32(nil), (*Encoder).fastpathEncMapUint8Int32R, (*Decoder).fastpathDecMapUint8Int32R)
	fn(map[uint8]int64(nil), (*Encoder).fastpathEncMapUint8Int64R, (*Decoder).fastpathDecMapUint8Int64R)
	fn(map[uint8]float32(nil), (*Encoder).fastpathEncMapUint8Float32R, (*Decoder).fastpathDecMapUint8Float32R)
	fn(map[uint8]float64(nil), (*Encoder).fastpathEncMapUint8Float64R, (*Decoder).fastpathDecMapUint8Float64R)
	fn(map[uint8]bool(nil), (*Encoder).fastpathEncMapUint8BoolR, (*Decoder).fastpathDecMapUint8BoolR)
	fn(map[uint16]interface{}(nil), (*Encoder).fastpathEncMapUint16IntfR, (*Decoder).fastpathDecMapUint16IntfR)
	fn(map[uint16]string(nil), (*Encoder).fastpathEncMapUint16StringR, (*Decoder).fastpathDecMapUint16StringR)
	fn(map[uint16]uint(nil), (*Encoder).fastpathEncMapUint16UintR, (*Decoder).fastpathDecMapUint16UintR)
	fn(map[uint16]uint8(nil), (*Encoder).fastpathEncMapUint16Uint8R, (*Decoder).fastpathDecMapUint16Uint8R)
	fn(map[uint16]uint16(nil), (*Encoder).fastpathEncMapUint16Uint16R, (*Decoder).fastpathDecMapUint16Uint16R)
	fn(map[uint16]uint32(nil), (*Encoder).fastpathEncMapUint16Uint32R, (*Decoder).fastpathDecMapUint16Uint32R)
	fn(map[uint16]uint64(nil), (*Encoder).fastpathEncMapUint16Uint64R, (*Decoder).fastpathDecMapUint16Uint64R)
	fn(map[uint16]uintptr(nil), (*Encoder).fastpathEncMapUint16UintptrR, (*Decoder).fastpathDecMapUint16UintptrR)
	fn(map[uint16]int(nil), (*Encoder).fastpathEncMapUint16IntR, (*Decoder).fastpathDecMapUint16IntR)
	fn(map[uint16]int8(nil), (*Encoder).fastpathEncMapUint16Int8R, (*Decoder).fastpathDecMapUint16Int8R)
	fn(map[uint16]int16(nil), (*Encoder).fastpathEncMapUint16Int16R, (*Decoder).fastpathDecMapUint16Int16R)
	fn(map[uint16]int32(nil), (*Encoder).fastpathEncMapUint16Int32R, (*Decoder).fastpathDecMapUint16Int32R)
	fn(map[uint16]int64(nil), (*Encoder).fastpathEncMapUint16Int64R, (*Decoder).fastpathDecMapUint16Int64R)
	fn(map[uint16]float32(nil), (*Encoder).fastpathEncMapUint16Float32R, (*Decoder).fastpathDecMapUint16Float32R)
	fn(map[uint16]float64(nil), (*Encoder).fastpathEncMapUint16Float64R, (*Decoder).fastpathDecMapUint16Float64R)
	fn(map[uint16]bool(nil), (*Encoder).fastpathEncMapUint16BoolR, (*Decoder).fastpathDecMapUint16BoolR)
	fn(map[uint32]interface{}(nil), (*Encoder).fastpathEncMapUint32IntfR, (*Decoder).fastpathDecMapUint32IntfR)
	fn(map[uint32]string(nil), (*Encoder).fastpathEncMapUint32StringR, (*Decoder).fastpathDecMapUint32StringR)
	fn(map[uint32]uint(nil), (*Encoder).fastpathEncMapUint32UintR, (*Decoder).fastpathDecMapUint32UintR)
	fn(map[uint32]uint8(nil), (*Encoder).fastpathEncMapUint32Uint8R, (*Decoder).fastpathDecMapUint32Uint8R)
	fn(map[uint32]uint16(nil), (*Encoder).fastpathEncMapUint32Uint16R, (*Decoder).fastpathDecMapUint32Uint16R)
	fn(map[uint32]uint32(nil), (*Encoder).fastpathEncMapUint32Uint32R, (*Decoder).fastpathDecMapUint32Uint32R)
	fn(map[uint32]uint64(nil), (*Encoder).fastpathEncMapUint32Uint64R, (*Decoder).fastpathDecMapUint32Uint64R)
	fn(map[uint32]uintptr(nil), (*Encoder).fastpathEncMapUint32UintptrR, (*Decoder).fastpathDecMapUint32UintptrR)
	fn(map[uint32]int(nil), (*Encoder).fastpathEncMapUint32IntR, (*Decoder).fastpathDecMapUint32IntR)
	fn(map[uint32]int8(nil), (*Encoder).fastpathEncMapUint32Int8R, (*Decoder).fastpathDecMapUint32Int8R)
	fn(map[uint32]int16(nil), (*Encoder).fastpathEncMapUint32Int16R, (*Decoder).fastpathDecMapUint32Int16R)
	fn(map[uint32]int32(nil), (*Encoder).fastpathEncMapUint32Int32R, (*Decoder).fastpathDecMapUint32Int32R)
	fn(map[uint32]int64(nil), (*Encoder).fastpathEncMapUint32Int64R, (*Decoder).fastpathDecMapUint32Int64R)
	fn(map[uint32]float32(nil), (*Encoder).fastpathEncMapUint32Float32R, (*Decoder).fastpathDecMapUint32Float32R)
	fn(map[uint32]float64(nil), (*Encoder).fastpathEncMapUint32Float64R, (*Decoder).fastpathDecMapUint32Float64R)
	fn(map[uint32]bool(nil), (*Encoder).fastpathEncMapUint32BoolR, (*Decoder).fastpathDecMapUint32BoolR)
	fn(map[uint64]interface{}(nil), (*Encoder).fastpathEncMapUint64IntfR, (*Decoder).fastpathDecMapUint64IntfR)
	fn(map[uint64]string(nil), (*Encoder).fastpathEncMapUint64StringR, (*Decoder).fastpathDecMapUint64StringR)
	fn(map[uint64]uint(nil), (*Encoder).fastpathEncMapUint64UintR, (*Decoder).fastpathDecMapUint64UintR)
	fn(map[uint64]uint8(nil), (*Encoder).fastpathEncMapUint64Uint8R, (*Decoder).fastpathDecMapUint64Uint8R)
	fn(map[uint64]uint16(nil), (*Encoder).fastpathEncMapUint64Uint16R, (*Decoder).fastpathDecMapUint64Uint16R)
	fn(map[uint64]uint32(nil), (*Encoder).fastpathEncMapUint64Uint32R, (*Decoder).fastpathDecMapUint64Uint32R)
	fn(map[uint64]uint64(nil), (*Encoder).fastpathEncMapUint64Uint64R, (*Decoder).fastpathDecMapUint64Uint64R)
	fn(map[uint64]uintptr(nil), (*Encoder).fastpathEncMapUint64UintptrR, (*Decoder).fastpathDecMapUint64UintptrR)
	fn(map[uint64]int(nil), (*Encoder).fastpathEncMapUint64IntR, (*Decoder).fastpathDecMapUint64IntR)
	fn(map[uint64]int8(nil), (*Encoder).fastpathEncMapUint64Int8R, (*Decoder).fastpathDecMapUint64Int8R)
	fn(map[uint64]int16(nil), (*Encoder).fastpathEncMapUint64Int16R, (*Decoder).fastpathDecMapUint64Int16R)
	fn(map[uint64]int32(nil), (*Encoder).fastpathEncMapUint64Int32R, (*Decoder).fastpathDecMapUint64Int32R)
	fn(map[uint64]int64(nil), (*Encoder).fastpathEncMapUint64Int64R, (*Decoder).fastpathDecMapUint64Int64R)
	fn(map[uint64]float32(nil), (*Encoder).fastpathEncMapUint64Float32R, (*Decoder).fastpathDecMapUint64Float32R)
	fn(map[uint64]float64(nil), (*Encoder).fastpathEncMapUint64Float64R, (*Decoder).fastpathDecMapUint64Float64R)
	fn(map[uint64]bool(nil), (*Encoder).fastpathEncMapUint64BoolR, (*Decoder).fastpathDecMapUint64BoolR)
	fn(map[uintptr]interface{}(nil), (*Encoder).fastpathEncMapUintptrIntfR, (*Decoder).fastpathDecMapUintptrIntfR)
	fn(map[uintptr]string(nil), (*Encoder).fastpathEncMapUintptrStringR, (*Decoder).fastpathDecMapUintptrStringR)
	fn(map[uintptr]uint(nil), (*Encoder).fastpathEncMapUintptrUintR, (*Decoder).fastpathDecMapUintptrUintR)
	fn(map[uintptr]uint8(nil), (*Encoder).fastpathEncMapUintptrUint8R, (*Decoder).fastpathDecMapUintptrUint8R)
	fn(map[uintptr]uint16(nil), (*Encoder).fastpathEncMapUintptrUint16R, (*Decoder).fastpathDecMapUintptrUint16R)
	fn(map[uintptr]uint32(nil), (*Encoder).fastpathEncMapUintptrUint32R, (*Decoder).fastpathDecMapUintptrUint32R)
	fn(map[uintptr]uint64(nil), (*Encoder).fastpathEncMapUintptrUint64R, (*Decoder).fastpathDecMapUintptrUint64R)
	fn(map[uintptr]uintptr(nil), (*Encoder).fastpathEncMapUintptrUintptrR, (*Decoder).fastpathDecMapUintptrUintptrR)
	fn(map[uintptr]int(nil), (*Encoder).fastpathEncMapUintptrIntR, (*Decoder).fastpathDecMapUintptrIntR)
	fn(map[uintptr]int8(nil), (*Encoder).fastpathEncMapUintptrInt8R, (*Decoder).fastpathDecMapUintptrInt8R)
	fn(map[uintptr]int16(nil), (*Encoder).fastpathEncMapUintptrInt16R, (*Decoder).fastpathDecMapUintptrInt16R)
	fn(map[uintptr]int32(nil), (*Encoder).fastpathEncMapUintptrInt32R, (*Decoder).fastpathDecMapUintptrInt32R)
	fn(map[uintptr]int64(nil), (*Encoder).fastpathEncMapUintptrInt64R, (*Decoder).fastpathDecMapUintptrInt64R)
	fn(map[uintptr]float32(nil), (*Encoder).fastpathEncMapUintptrFloat32R, (*Decoder).fastpathDecMapUintptrFloat32R)
	fn(map[uintptr]float64(nil), (*Encoder).fastpathEncMapUintptrFloat64R, (*Decoder).fastpathDecMapUintptrFloat64R)
	fn(map[uintptr]bool(nil), (*Encoder).fastpathEncMapUintptrBoolR, (*Decoder).fastpathDecMapUintptrBoolR)
	fn(map[int]interface{}(nil), (*Encoder).fastpathEncMapIntIntfR, (*Decoder).fastpathDecMapIntIntfR)
	fn(map[int]string(nil), (*Encoder).fastpathEncMapIntStringR, (*Decoder).fastpathDecMapIntStringR)
	fn(map[int]uint(nil), (*Encoder).fastpathEncMapIntUintR, (*Decoder).fastpathDecMapIntUintR)
	fn(map[int]uint8(nil), (*Encoder).fastpathEncMapIntUint8R, (*Decoder).fastpathDecMapIntUint8R)
	fn(map[int]uint16(nil), (*Encoder).fastpathEncMapIntUint16R, (*Decoder).fastpathDecMapIntUint16R)
	fn(map[int]uint32(nil), (*Encoder).fastpathEncMapIntUint32R, (*Decoder).fastpathDecMapIntUint32R)
	fn(map[int]uint64(nil), (*Encoder).fastpathEncMapIntUint64R, (*Decoder).fastpathDecMapIntUint64R)
	fn(map[int]uintptr(nil), (*Encoder).fastpathEncMapIntUintptrR, (*Decoder).fastpathDecMapIntUintptrR)
	fn(map[int]int(nil), (*Encoder).fastpathEncMapIntIntR, (*Decoder).fastpathDecMapIntIntR)
	fn(map[int]int8(nil), (*Encoder).fastpathEncMapIntInt8R, (*Decoder).fastpathDecMapIntInt8R)
	fn(map[int]int16(nil), (*Encoder).fastpathEncMapIntInt16R, (*Decoder).fastpathDecMapIntInt16R)
	fn(map[int]int32(nil), (*Encoder).fastpathEncMapIntInt32R, (*Decoder).fastpathDecMapIntInt32R)
	fn(map[int]int64(nil), (*Encoder).fastpathEncMapIntInt64R, (*Decoder).fastpathDecMapIntInt64R)
	fn(map[int]float32(nil), (*Encoder).fastpathEncMapIntFloat32R, (*Decoder).fastpathDecMapIntFloat32R)
	fn(map[int]float64(nil), (*Encoder).fastpathEncMapIntFloat64R, (*Decoder).fastpathDecMapIntFloat64R)
	fn(map[int]bool(nil), (*Encoder).fastpathEncMapIntBoolR, (*Decoder).fastpathDecMapIntBoolR)
	fn(map[int8]interface{}(nil), (*Encoder).fastpathEncMapInt8IntfR, (*Decoder).fastpathDecMapInt8IntfR)
	fn(map[int8]string(nil), (*Encoder).fastpathEncMapInt8StringR, (*Decoder).fastpathDecMapInt8StringR)
	fn(map[int8]uint(nil), (*Encoder).fastpathEncMapInt8UintR, (*Decoder).fastpathDecMapInt8UintR)
	fn(map[int8]uint8(nil), (*Encoder).fastpathEncMapInt8Uint8R, (*Decoder).fastpathDecMapInt8Uint8R)
	fn(map[int8]uint16(nil), (*Encoder).fastpathEncMapInt8Uint16R, (*Decoder).fastpathDecMapInt8Uint16R)
	fn(map[int8]uint32(nil), (*Encoder).fastpathEncMapInt8Uint32R, (*Decoder).fastpathDecMapInt8Uint32R)
	fn(map[int8]uint64(nil), (*Encoder).fastpathEncMapInt8Uint64R, (*Decoder).fastpathDecMapInt8Uint64R)
	fn(map[int8]uintptr(nil), (*Encoder).fastpathEncMapInt8UintptrR, (*Decoder).fastpathDecMapInt8UintptrR)
	fn(map[int8]int(nil), (*Encoder).fastpathEncMapInt8IntR, (*Decoder).fastpathDecMapInt8IntR)
	fn(map[int8]int8(nil), (*Encoder).fastpathEncMapInt8Int8R, (*Decoder).fastpathDecMapInt8Int8R)
	fn(map[int8]int16(nil), (*Encoder).fastpathEncMapInt8Int16R, (*Decoder).fastpathDecMapInt8Int16R)
	fn(map[int8]int32(nil), (*Encoder).fastpathEncMapInt8Int32R, (*Decoder).fastpathDecMapInt8Int32R)
	fn(map[int8]int64(nil), (*Encoder).fastpathEncMapInt8Int64R, (*Decoder).fastpathDecMapInt8Int64R)
	fn(map[int8]float32(nil), (*Encoder).fastpathEncMapInt8Float32R, (*Decoder).fastpathDecMapInt8Float32R)
	fn(map[int8]float64(nil), (*Encoder).fastpathEncMapInt8Float64R, (*Decoder).fastpathDecMapInt8Float64R)
	fn(map[int8]bool(nil), (*Encoder).fastpathEncMapInt8BoolR, (*Decoder).fastpathDecMapInt8BoolR)
	fn(map[int16]interface{}(nil), (*Encoder).fastpathEncMapInt16IntfR, (*Decoder).fastpathDecMapInt16IntfR)
	fn(map[int16]string(nil), (*Encoder).fastpathEncMapInt16StringR, (*Decoder).fastpathDecMapInt16StringR)
	fn(map[int16]uint(nil), (*Encoder).fastpathEncMapInt16UintR, (*Decoder).fastpathDecMapInt16UintR)
	fn(map[int16]uint8(nil), (*Encoder).fastpathEncMapInt16Uint8R, (*Decoder).fastpathDecMapInt16Uint8R)
	fn(map[int16]uint16(nil), (*Encoder).fastpathEncMapInt16Uint16R, (*Decoder).fastpathDecMapInt16Uint16R)
	fn(map[int16]uint32(nil), (*Encoder).fastpathEncMapInt16Uint32R, (*Decoder).fastpathDecMapInt16Uint32R)
	fn(map[int16]uint64(nil), (*Encoder).fastpathEncMapInt16Uint64R, (*Decoder).fastpathDecMapInt16Uint64R)
	fn(map[int16]uintptr(nil), (*Encoder).fastpathEncMapInt16UintptrR, (*Decoder).fastpathDecMapInt16UintptrR)
	fn(map[int16]int(nil), (*Encoder).fastpathEncMapInt16IntR, (*Decoder).fastpathDecMapInt16IntR)
	fn(map[int16]int8(nil), (*Encoder).fastpathEncMapInt16Int8R, (*Decoder).fastpathDecMapInt16Int8R)
	fn(map[int16]int16(nil), (*Encoder).fastpathEncMapInt16Int16R, (*Decoder).fastpathDecMapInt16Int16R)
	fn(map[int16]int32(nil), (*Encoder).fastpathEncMapInt16Int32R, (*Decoder).fastpathDecMapInt16Int32R)
	fn(map[int16]int64(nil), (*Encoder).fastpathEncMapInt16Int64R, (*Decoder).fastpathDecMapInt16Int64R)
	fn(map[int16]float32(nil), (*Encoder).fastpathEncMapInt16Float32R, (*Decoder).fastpathDecMapInt16Float32R)
	fn(map[int16]float64(nil), (*Encoder).fastpathEncMapInt16Float64R, (*Decoder).fastpathDecMapInt16Float64R)
	fn(map[int16]bool(nil), (*Encoder).fastpathEncMapInt16BoolR, (*Decoder).fastpathDecMapInt16BoolR)
	fn(map[int32]interface{}(nil), (*Encoder).fastpathEncMapInt32IntfR, (*Decoder).fastpathDecMapInt32IntfR)
	fn(map[int32]string(nil), (*Encoder).fastpathEncMapInt32StringR, (*Decoder).fastpathDecMapInt32StringR)
	fn(map[int32]uint(nil), (*Encoder).fastpathEncMapInt32UintR, (*Decoder).fastpathDecMapInt32UintR)
	fn(map[int32]uint8(nil), (*Encoder).fastpathEncMapInt32Uint8R, (*Decoder).fastpathDecMapInt32Uint8R)
	fn(map[int32]uint16(nil), (*Encoder).fastpathEncMapInt32Uint16R, (*Decoder).fastpathDecMapInt32Uint16R)
	fn(map[int32]uint32(nil), (*Encoder).fastpathEncMapInt32Uint32R, (*Decoder).fastpathDecMapInt32Uint32R)
	fn(map[int32]uint64(nil), (*Encoder).fastpathEncMapInt32Uint64R, (*Decoder).fastpathDecMapInt32Uint64R)
	fn(map[int32]uintptr(nil), (*Encoder).fastpathEncMapInt32UintptrR, (*Decoder).fastpathDecMapInt32UintptrR)
	fn(map[int32]int(nil), (*Encoder).fastpathEncMapInt32IntR, (*Decoder).fastpathDecMapInt32IntR)
	fn(map[int32]int8(nil), (*Encoder).fastpathEncMapInt32Int8R, (*Decoder).fastpathDecMapInt32Int8R)
	fn(map[int32]int16(nil), (*Encoder).fastpathEncMapInt32Int16R, (*Decoder).fastpathDecMapInt32Int16R)
	fn(map[int32]int32(nil), (*Encoder).fastpathEncMapInt32Int32R, (*Decoder).fastpathDecMapInt32Int32R)
	fn(map[int32]int64(nil), (*Encoder).fastpathEncMapInt32Int64R, (*Decoder).fastpathDecMapInt32Int64R)
	fn(map[int32]float32(nil), (*Encoder).fastpathEncMapInt32Float32R, (*Decoder).fastpathDecMapInt32Float32R)
	fn(map[int32]float64(nil), (*Encoder).fastpathEncMapInt32Float64R, (*Decoder).fastpathDecMapInt32Float64R)
	fn(map[int32]bool(nil), (*Encoder).fastpathEncMapInt32BoolR, (*Decoder).fastpathDecMapInt32BoolR)
	fn(map[int64]interface{}(nil), (*Encoder).fastpathEncMapInt64IntfR, (*Decoder).fastpathDecMapInt64IntfR)
	fn(map[int64]string(nil), (*Encoder).fastpathEncMapInt64StringR, (*Decoder).fastpathDecMapInt64StringR)
	fn(map[int64]uint(nil), (*Encoder).fastpathEncMapInt64UintR, (*Decoder).fastpathDecMapInt64UintR)
	fn(map[int64]uint8(nil), (*Encoder).fastpathEncMapInt64Uint8R, (*Decoder).fastpathDecMapInt64Uint8R)
	fn(map[int64]uint16(nil), (*Encoder).fastpathEncMapInt64Uint16R, (*Decoder).fastpathDecMapInt64Uint16R)
	fn(map[int64]uint32(nil), (*Encoder).fastpathEncMapInt64Uint32R, (*Decoder).fastpathDecMapInt64Uint32R)
	fn(map[int64]uint64(nil), (*Encoder).fastpathEncMapInt64Uint64R, (*Decoder).fastpathDecMapInt64Uint64R)
	fn(map[int64]uintptr(nil), (*Encoder).fastpathEncMapInt64UintptrR, (*Decoder).fastpathDecMapInt64UintptrR)
	fn(map[int64]int(nil), (*Encoder).fastpathEncMapInt64IntR, (*Decoder).fastpathDecMapInt64IntR)
	fn(map[int64]int8(nil), (*Encoder).fastpathEncMapInt64Int8R, (*Decoder).fastpathDecMapInt64Int8R)
	fn(map[int64]int16(nil), (*Encoder).fastpathEncMapInt64Int16R, (*Decoder).fastpathDecMapInt64Int16R)
	fn(map[int64]int32(nil), (*Encoder).fastpathEncMapInt64Int32R, (*Decoder).fastpathDecMapInt64Int32R)
	fn(map[int64]int64(nil), (*Encoder).fastpathEncMapInt64Int64R, (*Decoder).fastpathDecMapInt64Int64R)
	fn(map[int64]float32(nil), (*Encoder).fastpathEncMapInt64Float32R, (*Decoder).fastpathDecMapInt64Float32R)
	fn(map[int64]float64(nil), (*Encoder).fastpathEncMapInt64Float64R, (*Decoder).fastpathDecMapInt64Float64R)
	fn(map[int64]bool(nil), (*Encoder).fastpathEncMapInt64BoolR, (*Decoder).fastpathDecMapInt64BoolR)
	fn(map[bool]interface{}(nil), (*Encoder).fastpathEncMapBoolIntfR, (*Decoder).fastpathDecMapBoolIntfR)
	fn(map[bool]string(nil), (*Encoder).fastpathEncMapBoolStringR, (*Decoder).fastpathDecMapBoolStringR)
	fn(map[bool]uint(nil), (*Encoder).fastpathEncMapBoolUintR, (*Decoder).fastpathDecMapBoolUintR)
	fn(map[bool]uint8(nil), (*Encoder).fastpathEncMapBoolUint8R, (*Decoder).fastpathDecMapBoolUint8R)
	fn(map[bool]uint16(nil), (*Encoder).fastpathEncMapBoolUint16R, (*Decoder).fastpathDecMapBoolUint16R)
	fn(map[bool]uint32(nil), (*Encoder).fastpathEncMapBoolUint32R, (*Decoder).fastpathDecMapBoolUint32R)
	fn(map[bool]uint64(nil), (*Encoder).fastpathEncMapBoolUint64R, (*Decoder).fastpathDecMapBoolUint64R)
	fn(map[bool]uintptr(nil), (*Encoder).fastpathEncMapBoolUintptrR, (*Decoder).fastpathDecMapBoolUintptrR)
	fn(map[bool]int(nil), (*Encoder).fastpathEncMapBoolIntR, (*Decoder).fastpathDecMapBoolIntR)
	fn(map[bool]int8(nil), (*Encoder).fastpathEncMapBoolInt8R, (*Decoder).fastpathDecMapBoolInt8R)
	fn(map[bool]int16(nil), (*Encoder).fastpathEncMapBoolInt16R, (*Decoder).fastpathDecMapBoolInt16R)
	fn(map[bool]int32(nil), (*Encoder).fastpathEncMapBoolInt32R, (*Decoder).fastpathDecMapBoolInt32R)
	fn(map[bool]int64(nil), (*Encoder).fastpathEncMapBoolInt64R, (*Decoder).fastpathDecMapBoolInt64R)
	fn(map[bool]float32(nil), (*Encoder).fastpathEncMapBoolFloat32R, (*Decoder).fastpathDecMapBoolFloat32R)
	fn(map[bool]float64(nil), (*Encoder).fastpathEncMapBoolFloat64R, (*Decoder).fastpathDecMapBoolFloat64R)
	fn(map[bool]bool(nil), (*Encoder).fastpathEncMapBoolBoolR, (*Decoder).fastpathDecMapBoolBoolR)

	sort.Sort(fastpathAslice(fastpathAV[:]))
}

// -- encode

// -- -- fast path type switch
func fastpathEncodeTypeSwitch(iv interface{}, e *Encoder) bool {
	switch v := iv.(type) {

	case []interface{}:
		fastpathTV.EncSliceIntfV(v, e)
	case *[]interface{}:
		fastpathTV.EncSliceIntfV(*v, e)

	case map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(v, e)
	case *map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(*v, e)

	case map[interface{}]string:
		fastpathTV.EncMapIntfStringV(v, e)
	case *map[interface{}]string:
		fastpathTV.EncMapIntfStringV(*v, e)

	case map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(v, e)
	case *map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(*v, e)

	case map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(v, e)
	case *map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(*v, e)

	case map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(v, e)
	case *map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(*v, e)

	case map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(v, e)
	case *map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(*v, e)

	case map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(v, e)
	case *map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(*v, e)

	case map[interface{}]uintptr:
		fastpathTV.EncMapIntfUintptrV(v, e)
	case *map[interface{}]uintptr:
		fastpathTV.EncMapIntfUintptrV(*v, e)

	case map[interface{}]int:
		fastpathTV.EncMapIntfIntV(v, e)
	case *map[interface{}]int:
		fastpathTV.EncMapIntfIntV(*v, e)

	case map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(v, e)
	case *map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(*v, e)

	case map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(v, e)
	case *map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(*v, e)

	case map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(v, e)
	case *map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(*v, e)

	case map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(v, e)
	case *map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(*v, e)

	case map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(v, e)
	case *map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(*v, e)

	case map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(v, e)
	case *map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(*v, e)

	case map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(v, e)
	case *map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(*v, e)

	case []string:
		fastpathTV.EncSliceStringV(v, e)
	case *[]string:
		fastpathTV.EncSliceStringV(*v, e)

	case map[string]interface{}:
		fastpathTV.EncMapStringIntfV(v, e)
	case *map[string]interface{}:
		fastpathTV.EncMapStringIntfV(*v, e)

	case map[string]string:
		fastpathTV.EncMapStringStringV(v, e)
	case *map[string]string:
		fastpathTV.EncMapStringStringV(*v, e)

	case map[string]uint:
		fastpathTV.EncMapStringUintV(v, e)
	case *map[string]uint:
		fastpathTV.EncMapStringUintV(*v, e)

	case map[string]uint8:
		fastpathTV.EncMapStringUint8V(v, e)
	case *map[string]uint8:
		fastpathTV.EncMapStringUint8V(*v, e)

	case map[string]uint16:
		fastpathTV.EncMapStringUint16V(v, e)
	case *map[string]uint16:
		fastpathTV.EncMapStringUint16V(*v, e)

	case map[string]uint32:
		fastpathTV.EncMapStringUint32V(v, e)
	case *map[string]uint32:
		fastpathTV.EncMapStringUint32V(*v, e)

	case map[string]uint64:
		fastpathTV.EncMapStringUint64V(v, e)
	case *map[string]uint64:
		fastpathTV.EncMapStringUint64V(*v, e)

	case map[string]uintptr:
		fastpathTV.EncMapStringUintptrV(v, e)
	case *map[string]uintptr:
		fastpathTV.EncMapStringUintptrV(*v, e)

	case map[string]int:
		fastpathTV.EncMapStringIntV(v, e)
	case *map[string]int:
		fastpathTV.EncMapStringIntV(*v, e)

	case map[string]int8:
		fastpathTV.EncMapStringInt8V(v, e)
	case *map[string]int8:
		fastpathTV.EncMapStringInt8V(*v, e)

	case map[string]int16:
		fastpathTV.EncMapStringInt16V(v, e)
	case *map[string]int16:
		fastpathTV.EncMapStringInt16V(*v, e)

	case map[string]int32:
		fastpathTV.EncMapStringInt32V(v, e)
	case *map[string]int32:
		fastpathTV.EncMapStringInt32V(*v, e)

	case map[string]int64:
		fastpathTV.EncMapStringInt64V(v, e)
	case *map[string]int64:
		fastpathTV.EncMapStringInt64V(*v, e)

	case map[string]float32:
		fastpathTV.EncMapStringFloat32V(v, e)
	case *map[string]float32:
		fastpathTV.EncMapStringFloat32V(*v, e)

	case map[string]float64:
		fastpathTV.EncMapStringFloat64V(v, e)
	case *map[string]float64:
		fastpathTV.EncMapStringFloat64V(*v, e)

	case map[string]bool:
		fastpathTV.EncMapStringBoolV(v, e)
	case *map[string]bool:
		fastpathTV.EncMapStringBoolV(*v, e)

	case []float32:
		fastpathTV.EncSliceFloat32V(v, e)
	case *[]float32:
		fastpathTV.EncSliceFloat32V(*v, e)

	case map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(v, e)
	case *map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(*v, e)

	case map[float32]string:
		fastpathTV.EncMapFloat32StringV(v, e)
	case *map[float32]string:
		fastpathTV.EncMapFloat32StringV(*v, e)

	case map[float32]uint:
		fastpathTV.EncMapFloat32UintV(v, e)
	case *map[float32]uint:
		fastpathTV.EncMapFloat32UintV(*v, e)

	case map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(v, e)
	case *map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(*v, e)

	case map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(v, e)
	case *map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(*v, e)

	case map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(v, e)
	case *map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(*v, e)

	case map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(v, e)
	case *map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(*v, e)

	case map[float32]uintptr:
		fastpathTV.EncMapFloat32UintptrV(v, e)
	case *map[float32]uintptr:
		fastpathTV.EncMapFloat32UintptrV(*v, e)

	case map[float32]int:
		fastpathTV.EncMapFloat32IntV(v, e)
	case *map[float32]int:
		fastpathTV.EncMapFloat32IntV(*v, e)

	case map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(v, e)
	case *map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(*v, e)

	case map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(v, e)
	case *map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(*v, e)

	case map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(v, e)
	case *map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(*v, e)

	case map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(v, e)
	case *map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(*v, e)

	case map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(v, e)
	case *map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(*v, e)

	case map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(v, e)
	case *map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(*v, e)

	case map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(v, e)
	case *map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(*v, e)

	case []float64:
		fastpathTV.EncSliceFloat64V(v, e)
	case *[]float64:
		fastpathTV.EncSliceFloat64V(*v, e)

	case map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(v, e)
	case *map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(*v, e)

	case map[float64]string:
		fastpathTV.EncMapFloat64StringV(v, e)
	case *map[float64]string:
		fastpathTV.EncMapFloat64StringV(*v, e)

	case map[float64]uint:
		fastpathTV.EncMapFloat64UintV(v, e)
	case *map[float64]uint:
		fastpathTV.EncMapFloat64UintV(*v, e)

	case map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(v, e)
	case *map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(*v, e)

	case map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(v, e)
	case *map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(*v, e)

	case map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(v, e)
	case *map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(*v, e)

	case map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(v, e)
	case *map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(*v, e)

	case map[float64]uintptr:
		fastpathTV.EncMapFloat64UintptrV(v, e)
	case *map[float64]uintptr:
		fastpathTV.EncMapFloat64UintptrV(*v, e)

	case map[float64]int:
		fastpathTV.EncMapFloat64IntV(v, e)
	case *map[float64]int:
		fastpathTV.EncMapFloat64IntV(*v, e)

	case map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(v, e)
	case *map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(*v, e)

	case map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(v, e)
	case *map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(*v, e)

	case map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(v, e)
	case *map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(*v, e)

	case map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(v, e)
	case *map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(*v, e)

	case map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(v, e)
	case *map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(*v, e)

	case map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(v, e)
	case *map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(*v, e)

	case map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(v, e)
	case *map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(*v, e)

	case []uint:
		fastpathTV.EncSliceUintV(v, e)
	case *[]uint:
		fastpathTV.EncSliceUintV(*v, e)

	case map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(v, e)
	case *map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(*v, e)

	case map[uint]string:
		fastpathTV.EncMapUintStringV(v, e)
	case *map[uint]string:
		fastpathTV.EncMapUintStringV(*v, e)

	case map[uint]uint:
		fastpathTV.EncMapUintUintV(v, e)
	case *map[uint]uint:
		fastpathTV.EncMapUintUintV(*v, e)

	case map[uint]uint8:
		fastpathTV.EncMapUintUint8V(v, e)
	case *map[uint]uint8:
		fastpathTV.EncMapUintUint8V(*v, e)

	case map[uint]uint16:
		fastpathTV.EncMapUintUint16V(v, e)
	case *map[uint]uint16:
		fastpathTV.EncMapUintUint16V(*v, e)

	case map[uint]uint32:
		fastpathTV.EncMapUintUint32V(v, e)
	case *map[uint]uint32:
		fastpathTV.EncMapUintUint32V(*v, e)

	case map[uint]uint64:
		fastpathTV.EncMapUintUint64V(v, e)
	case *map[uint]uint64:
		fastpathTV.EncMapUintUint64V(*v, e)

	case map[uint]uintptr:
		fastpathTV.EncMapUintUintptrV(v, e)
	case *map[uint]uintptr:
		fastpathTV.EncMapUintUintptrV(*v, e)

	case map[uint]int:
		fastpathTV.EncMapUintIntV(v, e)
	case *map[uint]int:
		fastpathTV.EncMapUintIntV(*v, e)

	case map[uint]int8:
		fastpathTV.EncMapUintInt8V(v, e)
	case *map[uint]int8:
		fastpathTV.EncMapUintInt8V(*v, e)

	case map[uint]int16:
		fastpathTV.EncMapUintInt16V(v, e)
	case *map[uint]int16:
		fastpathTV.EncMapUintInt16V(*v, e)

	case map[uint]int32:
		fastpathTV.EncMapUintInt32V(v, e)
	case *map[uint]int32:
		fastpathTV.EncMapUintInt32V(*v, e)

	case map[uint]int64:
		fastpathTV.EncMapUintInt64V(v, e)
	case *map[uint]int64:
		fastpathTV.EncMapUintInt64V(*v, e)

	case map[uint]float32:
		fastpathTV.EncMapUintFloat32V(v, e)
	case *map[uint]float32:
		fastpathTV.EncMapUintFloat32V(*v, e)

	case map[uint]float64:
		fastpathTV.EncMapUintFloat64V(v, e)
	case *map[uint]float64:
		fastpathTV.EncMapUintFloat64V(*v, e)

	case map[uint]bool:
		fastpathTV.EncMapUintBoolV(v, e)
	case *map[uint]bool:
		fastpathTV.EncMapUintBoolV(*v, e)

	case map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(v, e)
	case *map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(*v, e)

	case map[uint8]string:
		fastpathTV.EncMapUint8StringV(v, e)
	case *map[uint8]string:
		fastpathTV.EncMapUint8StringV(*v, e)

	case map[uint8]uint:
		fastpathTV.EncMapUint8UintV(v, e)
	case *map[uint8]uint:
		fastpathTV.EncMapUint8UintV(*v, e)

	case map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(v, e)
	case *map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(*v, e)

	case map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(v, e)
	case *map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(*v, e)

	case map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(v, e)
	case *map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(*v, e)

	case map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(v, e)
	case *map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(*v, e)

	case map[uint8]uintptr:
		fastpathTV.EncMapUint8UintptrV(v, e)
	case *map[uint8]uintptr:
		fastpathTV.EncMapUint8UintptrV(*v, e)

	case map[uint8]int:
		fastpathTV.EncMapUint8IntV(v, e)
	case *map[uint8]int:
		fastpathTV.EncMapUint8IntV(*v, e)

	case map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(v, e)
	case *map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(*v, e)

	case map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(v, e)
	case *map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(*v, e)

	case map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(v, e)
	case *map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(*v, e)

	case map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(v, e)
	case *map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(*v, e)

	case map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(v, e)
	case *map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(*v, e)

	case map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(v, e)
	case *map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(*v, e)

	case map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(v, e)
	case *map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(*v, e)

	case []uint16:
		fastpathTV.EncSliceUint16V(v, e)
	case *[]uint16:
		fastpathTV.EncSliceUint16V(*v, e)

	case map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(v, e)
	case *map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(*v, e)

	case map[uint16]string:
		fastpathTV.EncMapUint16StringV(v, e)
	case *map[uint16]string:
		fastpathTV.EncMapUint16StringV(*v, e)

	case map[uint16]uint:
		fastpathTV.EncMapUint16UintV(v, e)
	case *map[uint16]uint:
		fastpathTV.EncMapUint16UintV(*v, e)

	case map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(v, e)
	case *map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(*v, e)

	case map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(v, e)
	case *map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(*v, e)

	case map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(v, e)
	case *map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(*v, e)

	case map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(v, e)
	case *map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(*v, e)

	case map[uint16]uintptr:
		fastpathTV.EncMapUint16UintptrV(v, e)
	case *map[uint16]uintptr:
		fastpathTV.EncMapUint16UintptrV(*v, e)

	case map[uint16]int:
		fastpathTV.EncMapUint16IntV(v, e)
	case *map[uint16]int:
		fastpathTV.EncMapUint16IntV(*v, e)

	case map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(v, e)
	case *map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(*v, e)

	case map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(v, e)
	case *map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(*v, e)

	case map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(v, e)
	case *map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(*v, e)

	case map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(v, e)
	case *map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(*v, e)

	case map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(v, e)
	case *map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(*v, e)

	case map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(v, e)
	case *map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(*v, e)

	case map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(v, e)
	case *map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(*v, e)

	case []uint32:
		fastpathTV.EncSliceUint32V(v, e)
	case *[]uint32:
		fastpathTV.EncSliceUint32V(*v, e)

	case map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(v, e)
	case *map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(*v, e)

	case map[uint32]string:
		fastpathTV.EncMapUint32StringV(v, e)
	case *map[uint32]string:
		fastpathTV.EncMapUint32StringV(*v, e)

	case map[uint32]uint:
		fastpathTV.EncMapUint32UintV(v, e)
	case *map[uint32]uint:
		fastpathTV.EncMapUint32UintV(*v, e)

	case map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(v, e)
	case *map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(*v, e)

	case map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(v, e)
	case *map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(*v, e)

	case map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(v, e)
	case *map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(*v, e)

	case map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(v, e)
	case *map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(*v, e)

	case map[uint32]uintptr:
		fastpathTV.EncMapUint32UintptrV(v, e)
	case *map[uint32]uintptr:
		fastpathTV.EncMapUint32UintptrV(*v, e)

	case map[uint32]int:
		fastpathTV.EncMapUint32IntV(v, e)
	case *map[uint32]int:
		fastpathTV.EncMapUint32IntV(*v, e)

	case map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(v, e)
	case *map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(*v, e)

	case map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(v, e)
	case *map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(*v, e)

	case map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(v, e)
	case *map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(*v, e)

	case map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(v, e)
	case *map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(*v, e)

	case map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(v, e)
	case *map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(*v, e)

	case map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(v, e)
	case *map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(*v, e)

	case map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(v, e)
	case *map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(*v, e)

	case []uint64:
		fastpathTV.EncSliceUint64V(v, e)
	case *[]uint64:
		fastpathTV.EncSliceUint64V(*v, e)

	case map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(v, e)
	case *map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(*v, e)

	case map[uint64]string:
		fastpathTV.EncMapUint64StringV(v, e)
	case *map[uint64]string:
		fastpathTV.EncMapUint64StringV(*v, e)

	case map[uint64]uint:
		fastpathTV.EncMapUint64UintV(v, e)
	case *map[uint64]uint:
		fastpathTV.EncMapUint64UintV(*v, e)

	case map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(v, e)
	case *map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(*v, e)

	case map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(v, e)
	case *map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(*v, e)

	case map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(v, e)
	case *map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(*v, e)

	case map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(v, e)
	case *map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(*v, e)

	case map[uint64]uintptr:
		fastpathTV.EncMapUint64UintptrV(v, e)
	case *map[uint64]uintptr:
		fastpathTV.EncMapUint64UintptrV(*v, e)

	case map[uint64]int:
		fastpathTV.EncMapUint64IntV(v, e)
	case *map[uint64]int:
		fastpathTV.EncMapUint64IntV(*v, e)

	case map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(v, e)
	case *map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(*v, e)

	case map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(v, e)
	case *map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(*v, e)

	case map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(v, e)
	case *map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(*v, e)

	case map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(v, e)
	case *map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(*v, e)

	case map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(v, e)
	case *map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(*v, e)

	case map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(v, e)
	case *map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(*v, e)

	case map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(v, e)
	case *map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(*v, e)

	case []uintptr:
		fastpathTV.EncSliceUintptrV(v, e)
	case *[]uintptr:
		fastpathTV.EncSliceUintptrV(*v, e)

	case map[uintptr]interface{}:
		fastpathTV.EncMapUintptrIntfV(v, e)
	case *map[uintptr]interface{}:
		fastpathTV.EncMapUintptrIntfV(*v, e)

	case map[uintptr]string:
		fastpathTV.EncMapUintptrStringV(v, e)
	case *map[uintptr]string:
		fastpathTV.EncMapUintptrStringV(*v, e)

	case map[uintptr]uint:
		fastpathTV.EncMapUintptrUintV(v, e)
	case *map[uintptr]uint:
		fastpathTV.EncMapUintptrUintV(*v, e)

	case map[uintptr]uint8:
		fastpathTV.EncMapUintptrUint8V(v, e)
	case *map[uintptr]uint8:
		fastpathTV.EncMapUintptrUint8V(*v, e)

	case map[uintptr]uint16:
		fastpathTV.EncMapUintptrUint16V(v, e)
	case *map[uintptr]uint16:
		fastpathTV.EncMapUintptrUint16V(*v, e)

	case map[uintptr]uint32:
		fastpathTV.EncMapUintptrUint32V(v, e)
	case *map[uintptr]uint32:
		fastpathTV.EncMapUintptrUint32V(*v, e)

	case map[uintptr]uint64:
		fastpathTV.EncMapUintptrUint64V(v, e)
	case *map[uintptr]uint64:
		fastpathTV.EncMapUintptrUint64V(*v, e)

	case map[uintptr]uintptr:
		fastpathTV.EncMapUintptrUintptrV(v, e)
	case *map[uintptr]uintptr:
		fastpathTV.EncMapUintptrUintptrV(*v, e)

	case map[uintptr]int:
		fastpathTV.EncMapUintptrIntV(v, e)
	case *map[uintptr]int:
		fastpathTV.EncMapUintptrIntV(*v, e)

	case map[uintptr]int8:
		fastpathTV.EncMapUintptrInt8V(v, e)
	case *map[uintptr]int8:
		fastpathTV.EncMapUintptrInt8V(*v, e)

	case map[uintptr]int16:
		fastpathTV.EncMapUintptrInt16V(v, e)
	case *map[uintptr]int16:
		fastpathTV.EncMapUintptrInt16V(*v, e)

	case map[uintptr]int32:
		fastpathTV.EncMapUintptrInt32V(v, e)
	case *map[uintptr]int32:
		fastpathTV.EncMapUintptrInt32V(*v, e)

	case map[uintptr]int64:
		fastpathTV.EncMapUintptrInt64V(v, e)
	case *map[uintptr]int64:
		fastpathTV.EncMapUintptrInt64V(*v, e)

	case map[uintptr]float32:
		fastpathTV.EncMapUintptrFloat32V(v, e)
	case *map[uintptr]float32:
		fastpathTV.EncMapUintptrFloat32V(*v, e)

	case map[uintptr]float64:
		fastpathTV.EncMapUintptrFloat64V(v, e)
	case *map[uintptr]float64:
		fastpathTV.EncMapUintptrFloat64V(*v, e)

	case map[uintptr]bool:
		fastpathTV.EncMapUintptrBoolV(v, e)
	case *map[uintptr]bool:
		fastpathTV.EncMapUintptrBoolV(*v, e)

	case []int:
		fastpathTV.EncSliceIntV(v, e)
	case *[]int:
		fastpathTV.EncSliceIntV(*v, e)

	case map[int]interface{}:
		fastpathTV.EncMapIntIntfV(v, e)
	case *map[int]interface{}:
		fastpathTV.EncMapIntIntfV(*v, e)

	case map[int]string:
		fastpathTV.EncMapIntStringV(v, e)
	case *map[int]string:
		fastpathTV.EncMapIntStringV(*v, e)

	case map[int]uint:
		fastpathTV.EncMapIntUintV(v, e)
	case *map[int]uint:
		fastpathTV.EncMapIntUintV(*v, e)

	case map[int]uint8:
		fastpathTV.EncMapIntUint8V(v, e)
	case *map[int]uint8:
		fastpathTV.EncMapIntUint8V(*v, e)

	case map[int]uint16:
		fastpathTV.EncMapIntUint16V(v, e)
	case *map[int]uint16:
		fastpathTV.EncMapIntUint16V(*v, e)

	case map[int]uint32:
		fastpathTV.EncMapIntUint32V(v, e)
	case *map[int]uint32:
		fastpathTV.EncMapIntUint32V(*v, e)

	case map[int]uint64:
		fastpathTV.EncMapIntUint64V(v, e)
	case *map[int]uint64:
		fastpathTV.EncMapIntUint64V(*v, e)

	case map[int]uintptr:
		fastpathTV.EncMapIntUintptrV(v, e)
	case *map[int]uintptr:
		fastpathTV.EncMapIntUintptrV(*v, e)

	case map[int]int:
		fastpathTV.EncMapIntIntV(v, e)
	case *map[int]int:
		fastpathTV.EncMapIntIntV(*v, e)

	case map[int]int8:
		fastpathTV.EncMapIntInt8V(v, e)
	case *map[int]int8:
		fastpathTV.EncMapIntInt8V(*v, e)

	case map[int]int16:
		fastpathTV.EncMapIntInt16V(v, e)
	case *map[int]int16:
		fastpathTV.EncMapIntInt16V(*v, e)

	case map[int]int32:
		fastpathTV.EncMapIntInt32V(v, e)
	case *map[int]int32:
		fastpathTV.EncMapIntInt32V(*v, e)

	case map[int]int64:
		fastpathTV.EncMapIntInt64V(v, e)
	case *map[int]int64:
		fastpathTV.EncMapIntInt64V(*v, e)

	case map[int]float32:
		fastpathTV.EncMapIntFloat32V(v, e)
	case *map[int]float32:
		fastpathTV.EncMapIntFloat32V(*v, e)

	case map[int]float64:
		fastpathTV.EncMapIntFloat64V(v, e)
	case *map[int]float64:
		fastpathTV.EncMapIntFloat64V(*v, e)

	case map[int]bool:
		fastpathTV.EncMapIntBoolV(v, e)
	case *map[int]bool:
		fastpathTV.EncMapIntBoolV(*v, e)

	case []int8:
		fastpathTV.EncSliceInt8V(v, e)
	case *[]int8:
		fastpathTV.EncSliceInt8V(*v, e)

	case map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(v, e)
	case *map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(*v, e)

	case map[int8]string:
		fastpathTV.EncMapInt8StringV(v, e)
	case *map[int8]string:
		fastpathTV.EncMapInt8StringV(*v, e)

	case map[int8]uint:
		fastpathTV.EncMapInt8UintV(v, e)
	case *map[int8]uint:
		fastpathTV.EncMapInt8UintV(*v, e)

	case map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(v, e)
	case *map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(*v, e)

	case map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(v, e)
	case *map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(*v, e)

	case map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(v, e)
	case *map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(*v, e)

	case map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(v, e)
	case *map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(*v, e)

	case map[int8]uintptr:
		fastpathTV.EncMapInt8UintptrV(v, e)
	case *map[int8]uintptr:
		fastpathTV.EncMapInt8UintptrV(*v, e)

	case map[int8]int:
		fastpathTV.EncMapInt8IntV(v, e)
	case *map[int8]int:
		fastpathTV.EncMapInt8IntV(*v, e)

	case map[int8]int8:
		fastpathTV.EncMapInt8Int8V(v, e)
	case *map[int8]int8:
		fastpathTV.EncMapInt8Int8V(*v, e)

	case map[int8]int16:
		fastpathTV.EncMapInt8Int16V(v, e)
	case *map[int8]int16:
		fastpathTV.EncMapInt8Int16V(*v, e)

	case map[int8]int32:
		fastpathTV.EncMapInt8Int32V(v, e)
	case *map[int8]int32:
		fastpathTV.EncMapInt8Int32V(*v, e)

	case map[int8]int64:
		fastpathTV.EncMapInt8Int64V(v, e)
	case *map[int8]int64:
		fastpathTV.EncMapInt8Int64V(*v, e)

	case map[int8]float32:
		fastpathTV.EncMapInt8Float32V(v, e)
	case *map[int8]float32:
		fastpathTV.EncMapInt8Float32V(*v, e)

	case map[int8]float64:
		fastpathTV.EncMapInt8Float64V(v, e)
	case *map[int8]float64:
		fastpathTV.EncMapInt8Float64V(*v, e)

	case map[int8]bool:
		fastpathTV.EncMapInt8BoolV(v, e)
	case *map[int8]bool:
		fastpathTV.EncMapInt8BoolV(*v, e)

	case []int16:
		fastpathTV.EncSliceInt16V(v, e)
	case *[]int16:
		fastpathTV.EncSliceInt16V(*v, e)

	case map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(v, e)
	case *map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(*v, e)

	case map[int16]string:
		fastpathTV.EncMapInt16StringV(v, e)
	case *map[int16]string:
		fastpathTV.EncMapInt16StringV(*v, e)

	case map[int16]uint:
		fastpathTV.EncMapInt16UintV(v, e)
	case *map[int16]uint:
		fastpathTV.EncMapInt16UintV(*v, e)

	case map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(v, e)
	case *map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(*v, e)

	case map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(v, e)
	case *map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(*v, e)

	case map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(v, e)
	case *map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(*v, e)

	case map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(v, e)
	case *map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(*v, e)

	case map[int16]uintptr:
		fastpathTV.EncMapInt16UintptrV(v, e)
	case *map[int16]uintptr:
		fastpathTV.EncMapInt16UintptrV(*v, e)

	case map[int16]int:
		fastpathTV.EncMapInt16IntV(v, e)
	case *map[int16]int:
		fastpathTV.EncMapInt16IntV(*v, e)

	case map[int16]int8:
		fastpathTV.EncMapInt16Int8V(v, e)
	case *map[int16]int8:
		fastpathTV.EncMapInt16Int8V(*v, e)

	case map[int16]int16:
		fastpathTV.EncMapInt16Int16V(v, e)
	case *map[int16]int16:
		fastpathTV.EncMapInt16Int16V(*v, e)

	case map[int16]int32:
		fastpathTV.EncMapInt16Int32V(v, e)
	case *map[int16]int32:
		fastpathTV.EncMapInt16Int32V(*v, e)

	case map[int16]int64:
		fastpathTV.EncMapInt16Int64V(v, e)
	case *map[int16]int64:
		fastpathTV.EncMapInt16Int64V(*v, e)

	case map[int16]float32:
		fastpathTV.EncMapInt16Float32V(v, e)
	case *map[int16]float32:
		fastpathTV.EncMapInt16Float32V(*v, e)

	case map[int16]float64:
		fastpathTV.EncMapInt16Float64V(v, e)
	case *map[int16]float64:
		fastpathTV.EncMapInt16Float64V(*v, e)

	case map[int16]bool:
		fastpathTV.EncMapInt16BoolV(v, e)
	case *map[int16]bool:
		fastpathTV.EncMapInt16BoolV(*v, e)

	case []int32:
		fastpathTV.EncSliceInt32V(v, e)
	case *[]int32:
		fastpathTV.EncSliceInt32V(*v, e)

	case map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(v, e)
	case *map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(*v, e)

	case map[int32]string:
		fastpathTV.EncMapInt32StringV(v, e)
	case *map[int32]string:
		fastpathTV.EncMapInt32StringV(*v, e)

	case map[int32]uint:
		fastpathTV.EncMapInt32UintV(v, e)
	case *map[int32]uint:
		fastpathTV.EncMapInt32UintV(*v, e)

	case map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(v, e)
	case *map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(*v, e)

	case map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(v, e)
	case *map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(*v, e)

	case map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(v, e)
	case *map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(*v, e)

	case map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(v, e)
	case *map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(*v, e)

	case map[int32]uintptr:
		fastpathTV.EncMapInt32UintptrV(v, e)
	case *map[int32]uintptr:
		fastpathTV.EncMapInt32UintptrV(*v, e)

	case map[int32]int:
		fastpathTV.EncMapInt32IntV(v, e)
	case *map[int32]int:
		fastpathTV.EncMapInt32IntV(*v, e)

	case map[int32]int8:
		fastpathTV.EncMapInt32Int8V(v, e)
	case *map[int32]int8:
		fastpathTV.EncMapInt32Int8V(*v, e)

	case map[int32]int16:
		fastpathTV.EncMapInt32Int16V(v, e)
	case *map[int32]int16:
		fastpathTV.EncMapInt32Int16V(*v, e)

	case map[int32]int32:
		fastpathTV.EncMapInt32Int32V(v, e)
	case *map[int32]int32:
		fastpathTV.EncMapInt32Int32V(*v, e)

	case map[int32]int64:
		fastpathTV.EncMapInt32Int64V(v, e)
	case *map[int32]int64:
		fastpathTV.EncMapInt32Int64V(*v, e)

	case map[int32]float32:
		fastpathTV.EncMapInt32Float32V(v, e)
	case *map[int32]float32:
		fastpathTV.EncMapInt32Float32V(*v, e)

	case map[int32]float64:
		fastpathTV.EncMapInt32Float64V(v, e)
	case *map[int32]float64:
		fastpathTV.EncMapInt32Float64V(*v, e)

	case map[int32]bool:
		fastpathTV.EncMapInt32BoolV(v, e)
	case *map[int32]bool:
		fastpathTV.EncMapInt32BoolV(*v, e)

	case []int64:
		fastpathTV.EncSliceInt64V(v, e)
	case *[]int64:
		fastpathTV.EncSliceInt64V(*v, e)

	case map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(v, e)
	case *map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(*v, e)

	case map[int64]string:
		fastpathTV.EncMapInt64StringV(v, e)
	case *map[int64]string:
		fastpathTV.EncMapInt64StringV(*v, e)

	case map[int64]uint:
		fastpathTV.EncMapInt64UintV(v, e)
	case *map[int64]uint:
		fastpathTV.EncMapInt64UintV(*v, e)

	case map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(v, e)
	case *map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(*v, e)

	case map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(v, e)
	case *map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(*v, e)

	case map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(v, e)
	case *map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(*v, e)

	case map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(v, e)
	case *map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(*v, e)

	case map[int64]uintptr:
		fastpathTV.EncMapInt64UintptrV(v, e)
	case *map[int64]uintptr:
		fastpathTV.EncMapInt64UintptrV(*v, e)

	case map[int64]int:
		fastpathTV.EncMapInt64IntV(v, e)
	case *map[int64]int:
		fastpathTV.EncMapInt64IntV(*v, e)

	case map[int64]int8:
		fastpathTV.EncMapInt64Int8V(v, e)
	case *map[int64]int8:
		fastpathTV.EncMapInt64Int8V(*v, e)

	case map[int64]int16:
		fastpathTV.EncMapInt64Int16V(v, e)
	case *map[int64]int16:
		fastpathTV.EncMapInt64Int16V(*v, e)

	case map[int64]int32:
		fastpathTV.EncMapInt64Int32V(v, e)
	case *map[int64]int32:
		fastpathTV.EncMapInt64Int32V(*v, e)

	case map[int64]int64:
		fastpathTV.EncMapInt64Int64V(v, e)
	case *map[int64]int64:
		fastpathTV.EncMapInt64Int64V(*v, e)

	case map[int64]float32:
		fastpathTV.EncMapInt64Float32V(v, e)
	case *map[int64]float32:
		fastpathTV.EncMapInt64Float32V(*v, e)

	case map[int64]float64:
		fastpathTV.EncMapInt64Float64V(v, e)
	case *map[int64]float64:
		fastpathTV.EncMapInt64Float64V(*v, e)

	case map[int64]bool:
		fastpathTV.EncMapInt64BoolV(v, e)
	case *map[int64]bool:
		fastpathTV.EncMapInt64BoolV(*v, e)

	case []bool:
		fastpathTV.EncSliceBoolV(v, e)
	case *[]bool:
		fastpathTV.EncSliceBoolV(*v, e)

	case map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(v, e)
	case *map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(*v, e)

	case map[bool]string:
		fastpathTV.EncMapBoolStringV(v, e)
	case *map[bool]string:
		fastpathTV.EncMapBoolStringV(*v, e)

	case map[bool]uint:
		fastpathTV.EncMapBoolUintV(v, e)
	case *map[bool]uint:
		fastpathTV.EncMapBoolUintV(*v, e)

	case map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(v, e)
	case *map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(*v, e)

	case map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(v, e)
	case *map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(*v, e)

	case map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(v, e)
	case *map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(*v, e)

	case map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(v, e)
	case *map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(*v, e)

	case map[bool]uintptr:
		fastpathTV.EncMapBoolUintptrV(v, e)
	case *map[bool]uintptr:
		fastpathTV.EncMapBoolUintptrV(*v, e)

	case map[bool]int:
		fastpathTV.EncMapBoolIntV(v, e)
	case *map[bool]int:
		fastpathTV.EncMapBoolIntV(*v, e)

	case map[bool]int8:
		fastpathTV.EncMapBoolInt8V(v, e)
	case *map[bool]int8:
		fastpathTV.EncMapBoolInt8V(*v, e)

	case map[bool]int16:
		fastpathTV.EncMapBoolInt16V(v, e)
	case *map[bool]int16:
		fastpathTV.EncMapBoolInt16V(*v, e)

	case map[bool]int32:
		fastpathTV.EncMapBoolInt32V(v, e)
	case *map[bool]int32:
		fastpathTV.EncMapBoolInt32V(*v, e)

	case map[bool]int64:
		fastpathTV.EncMapBoolInt64V(v, e)
	case *map[bool]int64:
		fastpathTV.EncMapBoolInt64V(*v, e)

	case map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(v, e)
	case *map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(*v, e)

	case map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(v, e)
	case *map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(*v, e)

	case map[bool]bool:
		fastpathTV.EncMapBoolBoolV(v, e)
	case *map[bool]bool:
		fastpathTV.EncMapBoolBoolV(*v, e)

	default:
		_ = v // TODO: workaround https://github.com/golang/go/issues/12927 (remove after go 1.6 release)
		return false
	}
	return true
}

func fastpathEncodeTypeSwitchSlice(iv interface{}, e *Encoder) bool {
	switch v := iv.(type) {

	case []interface{}:
		fastpathTV.EncSliceIntfV(v, e)
	case *[]interface{}:
		fastpathTV.EncSliceIntfV(*v, e)

	case []string:
		fastpathTV.EncSliceStringV(v, e)
	case *[]string:
		fastpathTV.EncSliceStringV(*v, e)

	case []float32:
		fastpathTV.EncSliceFloat32V(v, e)
	case *[]float32:
		fastpathTV.EncSliceFloat32V(*v, e)

	case []float64:
		fastpathTV.EncSliceFloat64V(v, e)
	case *[]float64:
		fastpathTV.EncSliceFloat64V(*v, e)

	case []uint:
		fastpathTV.EncSliceUintV(v, e)
	case *[]uint:
		fastpathTV.EncSliceUintV(*v, e)

	case []uint16:
		fastpathTV.EncSliceUint16V(v, e)
	case *[]uint16:
		fastpathTV.EncSliceUint16V(*v, e)

	case []uint32:
		fastpathTV.EncSliceUint32V(v, e)
	case *[]uint32:
		fastpathTV.EncSliceUint32V(*v, e)

	case []uint64:
		fastpathTV.EncSliceUint64V(v, e)
	case *[]uint64:
		fastpathTV.EncSliceUint64V(*v, e)

	case []uintptr:
		fastpathTV.EncSliceUintptrV(v, e)
	case *[]uintptr:
		fastpathTV.EncSliceUintptrV(*v, e)

	case []int:
		fastpathTV.EncSliceIntV(v, e)
	case *[]int:
		fastpathTV.EncSliceIntV(*v, e)

	case []int8:
		fastpathTV.EncSliceInt8V(v, e)
	case *[]int8:
		fastpathTV.EncSliceInt8V(*v, e)

	case []int16:
		fastpathTV.EncSliceInt16V(v, e)
	case *[]int16:
		fastpathTV.EncSliceInt16V(*v, e)

	case []int32:
		fastpathTV.EncSliceInt32V(v, e)
	case *[]int32:
		fastpathTV.EncSliceInt32V(*v, e)

	case []int64:
		fastpathTV.EncSliceInt64V(v, e)
	case *[]int64:
		fastpathTV.EncSliceInt64V(*v, e)

	case []bool:
		fastpathTV.EncSliceBoolV(v, e)
	case *[]bool:
		fastpathTV.EncSliceBoolV(*v, e)

	default:
		_ = v // TODO: workaround https://github.com/golang/go/issues/12927 (remove after go 1.6 release)
		return false
	}
	return true
}

func fastpathEncodeTypeSwitchMap(iv interface{}, e *Encoder) bool {
	switch v := iv.(type) {

	case map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(v, e)
	case *map[interface{}]interface{}:
		fastpathTV.EncMapIntfIntfV(*v, e)

	case map[interface{}]string:
		fastpathTV.EncMapIntfStringV(v, e)
	case *map[interface{}]string:
		fastpathTV.EncMapIntfStringV(*v, e)

	case map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(v, e)
	case *map[interface{}]uint:
		fastpathTV.EncMapIntfUintV(*v, e)

	case map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(v, e)
	case *map[interface{}]uint8:
		fastpathTV.EncMapIntfUint8V(*v, e)

	case map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(v, e)
	case *map[interface{}]uint16:
		fastpathTV.EncMapIntfUint16V(*v, e)

	case map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(v, e)
	case *map[interface{}]uint32:
		fastpathTV.EncMapIntfUint32V(*v, e)

	case map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(v, e)
	case *map[interface{}]uint64:
		fastpathTV.EncMapIntfUint64V(*v, e)

	case map[interface{}]uintptr:
		fastpathTV.EncMapIntfUintptrV(v, e)
	case *map[interface{}]uintptr:
		fastpathTV.EncMapIntfUintptrV(*v, e)

	case map[interface{}]int:
		fastpathTV.EncMapIntfIntV(v, e)
	case *map[interface{}]int:
		fastpathTV.EncMapIntfIntV(*v, e)

	case map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(v, e)
	case *map[interface{}]int8:
		fastpathTV.EncMapIntfInt8V(*v, e)

	case map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(v, e)
	case *map[interface{}]int16:
		fastpathTV.EncMapIntfInt16V(*v, e)

	case map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(v, e)
	case *map[interface{}]int32:
		fastpathTV.EncMapIntfInt32V(*v, e)

	case map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(v, e)
	case *map[interface{}]int64:
		fastpathTV.EncMapIntfInt64V(*v, e)

	case map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(v, e)
	case *map[interface{}]float32:
		fastpathTV.EncMapIntfFloat32V(*v, e)

	case map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(v, e)
	case *map[interface{}]float64:
		fastpathTV.EncMapIntfFloat64V(*v, e)

	case map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(v, e)
	case *map[interface{}]bool:
		fastpathTV.EncMapIntfBoolV(*v, e)

	case map[string]interface{}:
		fastpathTV.EncMapStringIntfV(v, e)
	case *map[string]interface{}:
		fastpathTV.EncMapStringIntfV(*v, e)

	case map[string]string:
		fastpathTV.EncMapStringStringV(v, e)
	case *map[string]string:
		fastpathTV.EncMapStringStringV(*v, e)

	case map[string]uint:
		fastpathTV.EncMapStringUintV(v, e)
	case *map[string]uint:
		fastpathTV.EncMapStringUintV(*v, e)

	case map[string]uint8:
		fastpathTV.EncMapStringUint8V(v, e)
	case *map[string]uint8:
		fastpathTV.EncMapStringUint8V(*v, e)

	case map[string]uint16:
		fastpathTV.EncMapStringUint16V(v, e)
	case *map[string]uint16:
		fastpathTV.EncMapStringUint16V(*v, e)

	case map[string]uint32:
		fastpathTV.EncMapStringUint32V(v, e)
	case *map[string]uint32:
		fastpathTV.EncMapStringUint32V(*v, e)

	case map[string]uint64:
		fastpathTV.EncMapStringUint64V(v, e)
	case *map[string]uint64:
		fastpathTV.EncMapStringUint64V(*v, e)

	case map[string]uintptr:
		fastpathTV.EncMapStringUintptrV(v, e)
	case *map[string]uintptr:
		fastpathTV.EncMapStringUintptrV(*v, e)

	case map[string]int:
		fastpathTV.EncMapStringIntV(v, e)
	case *map[string]int:
		fastpathTV.EncMapStringIntV(*v, e)

	case map[string]int8:
		fastpathTV.EncMapStringInt8V(v, e)
	case *map[string]int8:
		fastpathTV.EncMapStringInt8V(*v, e)

	case map[string]int16:
		fastpathTV.EncMapStringInt16V(v, e)
	case *map[string]int16:
		fastpathTV.EncMapStringInt16V(*v, e)

	case map[string]int32:
		fastpathTV.EncMapStringInt32V(v, e)
	case *map[string]int32:
		fastpathTV.EncMapStringInt32V(*v, e)

	case map[string]int64:
		fastpathTV.EncMapStringInt64V(v, e)
	case *map[string]int64:
		fastpathTV.EncMapStringInt64V(*v, e)

	case map[string]float32:
		fastpathTV.EncMapStringFloat32V(v, e)
	case *map[string]float32:
		fastpathTV.EncMapStringFloat32V(*v, e)

	case map[string]float64:
		fastpathTV.EncMapStringFloat64V(v, e)
	case *map[string]float64:
		fastpathTV.EncMapStringFloat64V(*v, e)

	case map[string]bool:
		fastpathTV.EncMapStringBoolV(v, e)
	case *map[string]bool:
		fastpathTV.EncMapStringBoolV(*v, e)

	case map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(v, e)
	case *map[float32]interface{}:
		fastpathTV.EncMapFloat32IntfV(*v, e)

	case map[float32]string:
		fastpathTV.EncMapFloat32StringV(v, e)
	case *map[float32]string:
		fastpathTV.EncMapFloat32StringV(*v, e)

	case map[float32]uint:
		fastpathTV.EncMapFloat32UintV(v, e)
	case *map[float32]uint:
		fastpathTV.EncMapFloat32UintV(*v, e)

	case map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(v, e)
	case *map[float32]uint8:
		fastpathTV.EncMapFloat32Uint8V(*v, e)

	case map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(v, e)
	case *map[float32]uint16:
		fastpathTV.EncMapFloat32Uint16V(*v, e)

	case map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(v, e)
	case *map[float32]uint32:
		fastpathTV.EncMapFloat32Uint32V(*v, e)

	case map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(v, e)
	case *map[float32]uint64:
		fastpathTV.EncMapFloat32Uint64V(*v, e)

	case map[float32]uintptr:
		fastpathTV.EncMapFloat32UintptrV(v, e)
	case *map[float32]uintptr:
		fastpathTV.EncMapFloat32UintptrV(*v, e)

	case map[float32]int:
		fastpathTV.EncMapFloat32IntV(v, e)
	case *map[float32]int:
		fastpathTV.EncMapFloat32IntV(*v, e)

	case map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(v, e)
	case *map[float32]int8:
		fastpathTV.EncMapFloat32Int8V(*v, e)

	case map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(v, e)
	case *map[float32]int16:
		fastpathTV.EncMapFloat32Int16V(*v, e)

	case map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(v, e)
	case *map[float32]int32:
		fastpathTV.EncMapFloat32Int32V(*v, e)

	case map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(v, e)
	case *map[float32]int64:
		fastpathTV.EncMapFloat32Int64V(*v, e)

	case map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(v, e)
	case *map[float32]float32:
		fastpathTV.EncMapFloat32Float32V(*v, e)

	case map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(v, e)
	case *map[float32]float64:
		fastpathTV.EncMapFloat32Float64V(*v, e)

	case map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(v, e)
	case *map[float32]bool:
		fastpathTV.EncMapFloat32BoolV(*v, e)

	case map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(v, e)
	case *map[float64]interface{}:
		fastpathTV.EncMapFloat64IntfV(*v, e)

	case map[float64]string:
		fastpathTV.EncMapFloat64StringV(v, e)
	case *map[float64]string:
		fastpathTV.EncMapFloat64StringV(*v, e)

	case map[float64]uint:
		fastpathTV.EncMapFloat64UintV(v, e)
	case *map[float64]uint:
		fastpathTV.EncMapFloat64UintV(*v, e)

	case map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(v, e)
	case *map[float64]uint8:
		fastpathTV.EncMapFloat64Uint8V(*v, e)

	case map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(v, e)
	case *map[float64]uint16:
		fastpathTV.EncMapFloat64Uint16V(*v, e)

	case map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(v, e)
	case *map[float64]uint32:
		fastpathTV.EncMapFloat64Uint32V(*v, e)

	case map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(v, e)
	case *map[float64]uint64:
		fastpathTV.EncMapFloat64Uint64V(*v, e)

	case map[float64]uintptr:
		fastpathTV.EncMapFloat64UintptrV(v, e)
	case *map[float64]uintptr:
		fastpathTV.EncMapFloat64UintptrV(*v, e)

	case map[float64]int:
		fastpathTV.EncMapFloat64IntV(v, e)
	case *map[float64]int:
		fastpathTV.EncMapFloat64IntV(*v, e)

	case map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(v, e)
	case *map[float64]int8:
		fastpathTV.EncMapFloat64Int8V(*v, e)

	case map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(v, e)
	case *map[float64]int16:
		fastpathTV.EncMapFloat64Int16V(*v, e)

	case map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(v, e)
	case *map[float64]int32:
		fastpathTV.EncMapFloat64Int32V(*v, e)

	case map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(v, e)
	case *map[float64]int64:
		fastpathTV.EncMapFloat64Int64V(*v, e)

	case map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(v, e)
	case *map[float64]float32:
		fastpathTV.EncMapFloat64Float32V(*v, e)

	case map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(v, e)
	case *map[float64]float64:
		fastpathTV.EncMapFloat64Float64V(*v, e)

	case map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(v, e)
	case *map[float64]bool:
		fastpathTV.EncMapFloat64BoolV(*v, e)

	case map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(v, e)
	case *map[uint]interface{}:
		fastpathTV.EncMapUintIntfV(*v, e)

	case map[uint]string:
		fastpathTV.EncMapUintStringV(v, e)
	case *map[uint]string:
		fastpathTV.EncMapUintStringV(*v, e)

	case map[uint]uint:
		fastpathTV.EncMapUintUintV(v, e)
	case *map[uint]uint:
		fastpathTV.EncMapUintUintV(*v, e)

	case map[uint]uint8:
		fastpathTV.EncMapUintUint8V(v, e)
	case *map[uint]uint8:
		fastpathTV.EncMapUintUint8V(*v, e)

	case map[uint]uint16:
		fastpathTV.EncMapUintUint16V(v, e)
	case *map[uint]uint16:
		fastpathTV.EncMapUintUint16V(*v, e)

	case map[uint]uint32:
		fastpathTV.EncMapUintUint32V(v, e)
	case *map[uint]uint32:
		fastpathTV.EncMapUintUint32V(*v, e)

	case map[uint]uint64:
		fastpathTV.EncMapUintUint64V(v, e)
	case *map[uint]uint64:
		fastpathTV.EncMapUintUint64V(*v, e)

	case map[uint]uintptr:
		fastpathTV.EncMapUintUintptrV(v, e)
	case *map[uint]uintptr:
		fastpathTV.EncMapUintUintptrV(*v, e)

	case map[uint]int:
		fastpathTV.EncMapUintIntV(v, e)
	case *map[uint]int:
		fastpathTV.EncMapUintIntV(*v, e)

	case map[uint]int8:
		fastpathTV.EncMapUintInt8V(v, e)
	case *map[uint]int8:
		fastpathTV.EncMapUintInt8V(*v, e)

	case map[uint]int16:
		fastpathTV.EncMapUintInt16V(v, e)
	case *map[uint]int16:
		fastpathTV.EncMapUintInt16V(*v, e)

	case map[uint]int32:
		fastpathTV.EncMapUintInt32V(v, e)
	case *map[uint]int32:
		fastpathTV.EncMapUintInt32V(*v, e)

	case map[uint]int64:
		fastpathTV.EncMapUintInt64V(v, e)
	case *map[uint]int64:
		fastpathTV.EncMapUintInt64V(*v, e)

	case map[uint]float32:
		fastpathTV.EncMapUintFloat32V(v, e)
	case *map[uint]float32:
		fastpathTV.EncMapUintFloat32V(*v, e)

	case map[uint]float64:
		fastpathTV.EncMapUintFloat64V(v, e)
	case *map[uint]float64:
		fastpathTV.EncMapUintFloat64V(*v, e)

	case map[uint]bool:
		fastpathTV.EncMapUintBoolV(v, e)
	case *map[uint]bool:
		fastpathTV.EncMapUintBoolV(*v, e)

	case map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(v, e)
	case *map[uint8]interface{}:
		fastpathTV.EncMapUint8IntfV(*v, e)

	case map[uint8]string:
		fastpathTV.EncMapUint8StringV(v, e)
	case *map[uint8]string:
		fastpathTV.EncMapUint8StringV(*v, e)

	case map[uint8]uint:
		fastpathTV.EncMapUint8UintV(v, e)
	case *map[uint8]uint:
		fastpathTV.EncMapUint8UintV(*v, e)

	case map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(v, e)
	case *map[uint8]uint8:
		fastpathTV.EncMapUint8Uint8V(*v, e)

	case map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(v, e)
	case *map[uint8]uint16:
		fastpathTV.EncMapUint8Uint16V(*v, e)

	case map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(v, e)
	case *map[uint8]uint32:
		fastpathTV.EncMapUint8Uint32V(*v, e)

	case map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(v, e)
	case *map[uint8]uint64:
		fastpathTV.EncMapUint8Uint64V(*v, e)

	case map[uint8]uintptr:
		fastpathTV.EncMapUint8UintptrV(v, e)
	case *map[uint8]uintptr:
		fastpathTV.EncMapUint8UintptrV(*v, e)

	case map[uint8]int:
		fastpathTV.EncMapUint8IntV(v, e)
	case *map[uint8]int:
		fastpathTV.EncMapUint8IntV(*v, e)

	case map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(v, e)
	case *map[uint8]int8:
		fastpathTV.EncMapUint8Int8V(*v, e)

	case map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(v, e)
	case *map[uint8]int16:
		fastpathTV.EncMapUint8Int16V(*v, e)

	case map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(v, e)
	case *map[uint8]int32:
		fastpathTV.EncMapUint8Int32V(*v, e)

	case map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(v, e)
	case *map[uint8]int64:
		fastpathTV.EncMapUint8Int64V(*v, e)

	case map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(v, e)
	case *map[uint8]float32:
		fastpathTV.EncMapUint8Float32V(*v, e)

	case map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(v, e)
	case *map[uint8]float64:
		fastpathTV.EncMapUint8Float64V(*v, e)

	case map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(v, e)
	case *map[uint8]bool:
		fastpathTV.EncMapUint8BoolV(*v, e)

	case map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(v, e)
	case *map[uint16]interface{}:
		fastpathTV.EncMapUint16IntfV(*v, e)

	case map[uint16]string:
		fastpathTV.EncMapUint16StringV(v, e)
	case *map[uint16]string:
		fastpathTV.EncMapUint16StringV(*v, e)

	case map[uint16]uint:
		fastpathTV.EncMapUint16UintV(v, e)
	case *map[uint16]uint:
		fastpathTV.EncMapUint16UintV(*v, e)

	case map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(v, e)
	case *map[uint16]uint8:
		fastpathTV.EncMapUint16Uint8V(*v, e)

	case map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(v, e)
	case *map[uint16]uint16:
		fastpathTV.EncMapUint16Uint16V(*v, e)

	case map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(v, e)
	case *map[uint16]uint32:
		fastpathTV.EncMapUint16Uint32V(*v, e)

	case map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(v, e)
	case *map[uint16]uint64:
		fastpathTV.EncMapUint16Uint64V(*v, e)

	case map[uint16]uintptr:
		fastpathTV.EncMapUint16UintptrV(v, e)
	case *map[uint16]uintptr:
		fastpathTV.EncMapUint16UintptrV(*v, e)

	case map[uint16]int:
		fastpathTV.EncMapUint16IntV(v, e)
	case *map[uint16]int:
		fastpathTV.EncMapUint16IntV(*v, e)

	case map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(v, e)
	case *map[uint16]int8:
		fastpathTV.EncMapUint16Int8V(*v, e)

	case map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(v, e)
	case *map[uint16]int16:
		fastpathTV.EncMapUint16Int16V(*v, e)

	case map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(v, e)
	case *map[uint16]int32:
		fastpathTV.EncMapUint16Int32V(*v, e)

	case map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(v, e)
	case *map[uint16]int64:
		fastpathTV.EncMapUint16Int64V(*v, e)

	case map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(v, e)
	case *map[uint16]float32:
		fastpathTV.EncMapUint16Float32V(*v, e)

	case map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(v, e)
	case *map[uint16]float64:
		fastpathTV.EncMapUint16Float64V(*v, e)

	case map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(v, e)
	case *map[uint16]bool:
		fastpathTV.EncMapUint16BoolV(*v, e)

	case map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(v, e)
	case *map[uint32]interface{}:
		fastpathTV.EncMapUint32IntfV(*v, e)

	case map[uint32]string:
		fastpathTV.EncMapUint32StringV(v, e)
	case *map[uint32]string:
		fastpathTV.EncMapUint32StringV(*v, e)

	case map[uint32]uint:
		fastpathTV.EncMapUint32UintV(v, e)
	case *map[uint32]uint:
		fastpathTV.EncMapUint32UintV(*v, e)

	case map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(v, e)
	case *map[uint32]uint8:
		fastpathTV.EncMapUint32Uint8V(*v, e)

	case map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(v, e)
	case *map[uint32]uint16:
		fastpathTV.EncMapUint32Uint16V(*v, e)

	case map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(v, e)
	case *map[uint32]uint32:
		fastpathTV.EncMapUint32Uint32V(*v, e)

	case map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(v, e)
	case *map[uint32]uint64:
		fastpathTV.EncMapUint32Uint64V(*v, e)

	case map[uint32]uintptr:
		fastpathTV.EncMapUint32UintptrV(v, e)
	case *map[uint32]uintptr:
		fastpathTV.EncMapUint32UintptrV(*v, e)

	case map[uint32]int:
		fastpathTV.EncMapUint32IntV(v, e)
	case *map[uint32]int:
		fastpathTV.EncMapUint32IntV(*v, e)

	case map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(v, e)
	case *map[uint32]int8:
		fastpathTV.EncMapUint32Int8V(*v, e)

	case map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(v, e)
	case *map[uint32]int16:
		fastpathTV.EncMapUint32Int16V(*v, e)

	case map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(v, e)
	case *map[uint32]int32:
		fastpathTV.EncMapUint32Int32V(*v, e)

	case map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(v, e)
	case *map[uint32]int64:
		fastpathTV.EncMapUint32Int64V(*v, e)

	case map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(v, e)
	case *map[uint32]float32:
		fastpathTV.EncMapUint32Float32V(*v, e)

	case map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(v, e)
	case *map[uint32]float64:
		fastpathTV.EncMapUint32Float64V(*v, e)

	case map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(v, e)
	case *map[uint32]bool:
		fastpathTV.EncMapUint32BoolV(*v, e)

	case map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(v, e)
	case *map[uint64]interface{}:
		fastpathTV.EncMapUint64IntfV(*v, e)

	case map[uint64]string:
		fastpathTV.EncMapUint64StringV(v, e)
	case *map[uint64]string:
		fastpathTV.EncMapUint64StringV(*v, e)

	case map[uint64]uint:
		fastpathTV.EncMapUint64UintV(v, e)
	case *map[uint64]uint:
		fastpathTV.EncMapUint64UintV(*v, e)

	case map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(v, e)
	case *map[uint64]uint8:
		fastpathTV.EncMapUint64Uint8V(*v, e)

	case map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(v, e)
	case *map[uint64]uint16:
		fastpathTV.EncMapUint64Uint16V(*v, e)

	case map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(v, e)
	case *map[uint64]uint32:
		fastpathTV.EncMapUint64Uint32V(*v, e)

	case map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(v, e)
	case *map[uint64]uint64:
		fastpathTV.EncMapUint64Uint64V(*v, e)

	case map[uint64]uintptr:
		fastpathTV.EncMapUint64UintptrV(v, e)
	case *map[uint64]uintptr:
		fastpathTV.EncMapUint64UintptrV(*v, e)

	case map[uint64]int:
		fastpathTV.EncMapUint64IntV(v, e)
	case *map[uint64]int:
		fastpathTV.EncMapUint64IntV(*v, e)

	case map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(v, e)
	case *map[uint64]int8:
		fastpathTV.EncMapUint64Int8V(*v, e)

	case map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(v, e)
	case *map[uint64]int16:
		fastpathTV.EncMapUint64Int16V(*v, e)

	case map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(v, e)
	case *map[uint64]int32:
		fastpathTV.EncMapUint64Int32V(*v, e)

	case map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(v, e)
	case *map[uint64]int64:
		fastpathTV.EncMapUint64Int64V(*v, e)

	case map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(v, e)
	case *map[uint64]float32:
		fastpathTV.EncMapUint64Float32V(*v, e)

	case map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(v, e)
	case *map[uint64]float64:
		fastpathTV.EncMapUint64Float64V(*v, e)

	case map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(v, e)
	case *map[uint64]bool:
		fastpathTV.EncMapUint64BoolV(*v, e)

	case map[uintptr]interface{}:
		fastpathTV.EncMapUintptrIntfV(v, e)
	case *map[uintptr]interface{}:
		fastpathTV.EncMapUintptrIntfV(*v, e)

	case map[uintptr]string:
		fastpathTV.EncMapUintptrStringV(v, e)
	case *map[uintptr]string:
		fastpathTV.EncMapUintptrStringV(*v, e)

	case map[uintptr]uint:
		fastpathTV.EncMapUintptrUintV(v, e)
	case *map[uintptr]uint:
		fastpathTV.EncMapUintptrUintV(*v, e)

	case map[uintptr]uint8:
		fastpathTV.EncMapUintptrUint8V(v, e)
	case *map[uintptr]uint8:
		fastpathTV.EncMapUintptrUint8V(*v, e)

	case map[uintptr]uint16:
		fastpathTV.EncMapUintptrUint16V(v, e)
	case *map[uintptr]uint16:
		fastpathTV.EncMapUintptrUint16V(*v, e)

	case map[uintptr]uint32:
		fastpathTV.EncMapUintptrUint32V(v, e)
	case *map[uintptr]uint32:
		fastpathTV.EncMapUintptrUint32V(*v, e)

	case map[uintptr]uint64:
		fastpathTV.EncMapUintptrUint64V(v, e)
	case *map[uintptr]uint64:
		fastpathTV.EncMapUintptrUint64V(*v, e)

	case map[uintptr]uintptr:
		fastpathTV.EncMapUintptrUintptrV(v, e)
	case *map[uintptr]uintptr:
		fastpathTV.EncMapUintptrUintptrV(*v, e)

	case map[uintptr]int:
		fastpathTV.EncMapUintptrIntV(v, e)
	case *map[uintptr]int:
		fastpathTV.EncMapUintptrIntV(*v, e)

	case map[uintptr]int8:
		fastpathTV.EncMapUintptrInt8V(v, e)
	case *map[uintptr]int8:
		fastpathTV.EncMapUintptrInt8V(*v, e)

	case map[uintptr]int16:
		fastpathTV.EncMapUintptrInt16V(v, e)
	case *map[uintptr]int16:
		fastpathTV.EncMapUintptrInt16V(*v, e)

	case map[uintptr]int32:
		fastpathTV.EncMapUintptrInt32V(v, e)
	case *map[uintptr]int32:
		fastpathTV.EncMapUintptrInt32V(*v, e)

	case map[uintptr]int64:
		fastpathTV.EncMapUintptrInt64V(v, e)
	case *map[uintptr]int64:
		fastpathTV.EncMapUintptrInt64V(*v, e)

	case map[uintptr]float32:
		fastpathTV.EncMapUintptrFloat32V(v, e)
	case *map[uintptr]float32:
		fastpathTV.EncMapUintptrFloat32V(*v, e)

	case map[uintptr]float64:
		fastpathTV.EncMapUintptrFloat64V(v, e)
	case *map[uintptr]float64:
		fastpathTV.EncMapUintptrFloat64V(*v, e)

	case map[uintptr]bool:
		fastpathTV.EncMapUintptrBoolV(v, e)
	case *map[uintptr]bool:
		fastpathTV.EncMapUintptrBoolV(*v, e)

	case map[int]interface{}:
		fastpathTV.EncMapIntIntfV(v, e)
	case *map[int]interface{}:
		fastpathTV.EncMapIntIntfV(*v, e)

	case map[int]string:
		fastpathTV.EncMapIntStringV(v, e)
	case *map[int]string:
		fastpathTV.EncMapIntStringV(*v, e)

	case map[int]uint:
		fastpathTV.EncMapIntUintV(v, e)
	case *map[int]uint:
		fastpathTV.EncMapIntUintV(*v, e)

	case map[int]uint8:
		fastpathTV.EncMapIntUint8V(v, e)
	case *map[int]uint8:
		fastpathTV.EncMapIntUint8V(*v, e)

	case map[int]uint16:
		fastpathTV.EncMapIntUint16V(v, e)
	case *map[int]uint16:
		fastpathTV.EncMapIntUint16V(*v, e)

	case map[int]uint32:
		fastpathTV.EncMapIntUint32V(v, e)
	case *map[int]uint32:
		fastpathTV.EncMapIntUint32V(*v, e)

	case map[int]uint64:
		fastpathTV.EncMapIntUint64V(v, e)
	case *map[int]uint64:
		fastpathTV.EncMapIntUint64V(*v, e)

	case map[int]uintptr:
		fastpathTV.EncMapIntUintptrV(v, e)
	case *map[int]uintptr:
		fastpathTV.EncMapIntUintptrV(*v, e)

	case map[int]int:
		fastpathTV.EncMapIntIntV(v, e)
	case *map[int]int:
		fastpathTV.EncMapIntIntV(*v, e)

	case map[int]int8:
		fastpathTV.EncMapIntInt8V(v, e)
	case *map[int]int8:
		fastpathTV.EncMapIntInt8V(*v, e)

	case map[int]int16:
		fastpathTV.EncMapIntInt16V(v, e)
	case *map[int]int16:
		fastpathTV.EncMapIntInt16V(*v, e)

	case map[int]int32:
		fastpathTV.EncMapIntInt32V(v, e)
	case *map[int]int32:
		fastpathTV.EncMapIntInt32V(*v, e)

	case map[int]int64:
		fastpathTV.EncMapIntInt64V(v, e)
	case *map[int]int64:
		fastpathTV.EncMapIntInt64V(*v, e)

	case map[int]float32:
		fastpathTV.EncMapIntFloat32V(v, e)
	case *map[int]float32:
		fastpathTV.EncMapIntFloat32V(*v, e)

	case map[int]float64:
		fastpathTV.EncMapIntFloat64V(v, e)
	case *map[int]float64:
		fastpathTV.EncMapIntFloat64V(*v, e)

	case map[int]bool:
		fastpathTV.EncMapIntBoolV(v, e)
	case *map[int]bool:
		fastpathTV.EncMapIntBoolV(*v, e)

	case map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(v, e)
	case *map[int8]interface{}:
		fastpathTV.EncMapInt8IntfV(*v, e)

	case map[int8]string:
		fastpathTV.EncMapInt8StringV(v, e)
	case *map[int8]string:
		fastpathTV.EncMapInt8StringV(*v, e)

	case map[int8]uint:
		fastpathTV.EncMapInt8UintV(v, e)
	case *map[int8]uint:
		fastpathTV.EncMapInt8UintV(*v, e)

	case map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(v, e)
	case *map[int8]uint8:
		fastpathTV.EncMapInt8Uint8V(*v, e)

	case map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(v, e)
	case *map[int8]uint16:
		fastpathTV.EncMapInt8Uint16V(*v, e)

	case map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(v, e)
	case *map[int8]uint32:
		fastpathTV.EncMapInt8Uint32V(*v, e)

	case map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(v, e)
	case *map[int8]uint64:
		fastpathTV.EncMapInt8Uint64V(*v, e)

	case map[int8]uintptr:
		fastpathTV.EncMapInt8UintptrV(v, e)
	case *map[int8]uintptr:
		fastpathTV.EncMapInt8UintptrV(*v, e)

	case map[int8]int:
		fastpathTV.EncMapInt8IntV(v, e)
	case *map[int8]int:
		fastpathTV.EncMapInt8IntV(*v, e)

	case map[int8]int8:
		fastpathTV.EncMapInt8Int8V(v, e)
	case *map[int8]int8:
		fastpathTV.EncMapInt8Int8V(*v, e)

	case map[int8]int16:
		fastpathTV.EncMapInt8Int16V(v, e)
	case *map[int8]int16:
		fastpathTV.EncMapInt8Int16V(*v, e)

	case map[int8]int32:
		fastpathTV.EncMapInt8Int32V(v, e)
	case *map[int8]int32:
		fastpathTV.EncMapInt8Int32V(*v, e)

	case map[int8]int64:
		fastpathTV.EncMapInt8Int64V(v, e)
	case *map[int8]int64:
		fastpathTV.EncMapInt8Int64V(*v, e)

	case map[int8]float32:
		fastpathTV.EncMapInt8Float32V(v, e)
	case *map[int8]float32:
		fastpathTV.EncMapInt8Float32V(*v, e)

	case map[int8]float64:
		fastpathTV.EncMapInt8Float64V(v, e)
	case *map[int8]float64:
		fastpathTV.EncMapInt8Float64V(*v, e)

	case map[int8]bool:
		fastpathTV.EncMapInt8BoolV(v, e)
	case *map[int8]bool:
		fastpathTV.EncMapInt8BoolV(*v, e)

	case map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(v, e)
	case *map[int16]interface{}:
		fastpathTV.EncMapInt16IntfV(*v, e)

	case map[int16]string:
		fastpathTV.EncMapInt16StringV(v, e)
	case *map[int16]string:
		fastpathTV.EncMapInt16StringV(*v, e)

	case map[int16]uint:
		fastpathTV.EncMapInt16UintV(v, e)
	case *map[int16]uint:
		fastpathTV.EncMapInt16UintV(*v, e)

	case map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(v, e)
	case *map[int16]uint8:
		fastpathTV.EncMapInt16Uint8V(*v, e)

	case map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(v, e)
	case *map[int16]uint16:
		fastpathTV.EncMapInt16Uint16V(*v, e)

	case map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(v, e)
	case *map[int16]uint32:
		fastpathTV.EncMapInt16Uint32V(*v, e)

	case map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(v, e)
	case *map[int16]uint64:
		fastpathTV.EncMapInt16Uint64V(*v, e)

	case map[int16]uintptr:
		fastpathTV.EncMapInt16UintptrV(v, e)
	case *map[int16]uintptr:
		fastpathTV.EncMapInt16UintptrV(*v, e)

	case map[int16]int:
		fastpathTV.EncMapInt16IntV(v, e)
	case *map[int16]int:
		fastpathTV.EncMapInt16IntV(*v, e)

	case map[int16]int8:
		fastpathTV.EncMapInt16Int8V(v, e)
	case *map[int16]int8:
		fastpathTV.EncMapInt16Int8V(*v, e)

	case map[int16]int16:
		fastpathTV.EncMapInt16Int16V(v, e)
	case *map[int16]int16:
		fastpathTV.EncMapInt16Int16V(*v, e)

	case map[int16]int32:
		fastpathTV.EncMapInt16Int32V(v, e)
	case *map[int16]int32:
		fastpathTV.EncMapInt16Int32V(*v, e)

	case map[int16]int64:
		fastpathTV.EncMapInt16Int64V(v, e)
	case *map[int16]int64:
		fastpathTV.EncMapInt16Int64V(*v, e)

	case map[int16]float32:
		fastpathTV.EncMapInt16Float32V(v, e)
	case *map[int16]float32:
		fastpathTV.EncMapInt16Float32V(*v, e)

	case map[int16]float64:
		fastpathTV.EncMapInt16Float64V(v, e)
	case *map[int16]float64:
		fastpathTV.EncMapInt16Float64V(*v, e)

	case map[int16]bool:
		fastpathTV.EncMapInt16BoolV(v, e)
	case *map[int16]bool:
		fastpathTV.EncMapInt16BoolV(*v, e)

	case map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(v, e)
	case *map[int32]interface{}:
		fastpathTV.EncMapInt32IntfV(*v, e)

	case map[int32]string:
		fastpathTV.EncMapInt32StringV(v, e)
	case *map[int32]string:
		fastpathTV.EncMapInt32StringV(*v, e)

	case map[int32]uint:
		fastpathTV.EncMapInt32UintV(v, e)
	case *map[int32]uint:
		fastpathTV.EncMapInt32UintV(*v, e)

	case map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(v, e)
	case *map[int32]uint8:
		fastpathTV.EncMapInt32Uint8V(*v, e)

	case map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(v, e)
	case *map[int32]uint16:
		fastpathTV.EncMapInt32Uint16V(*v, e)

	case map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(v, e)
	case *map[int32]uint32:
		fastpathTV.EncMapInt32Uint32V(*v, e)

	case map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(v, e)
	case *map[int32]uint64:
		fastpathTV.EncMapInt32Uint64V(*v, e)

	case map[int32]uintptr:
		fastpathTV.EncMapInt32UintptrV(v, e)
	case *map[int32]uintptr:
		fastpathTV.EncMapInt32UintptrV(*v, e)

	case map[int32]int:
		fastpathTV.EncMapInt32IntV(v, e)
	case *map[int32]int:
		fastpathTV.EncMapInt32IntV(*v, e)

	case map[int32]int8:
		fastpathTV.EncMapInt32Int8V(v, e)
	case *map[int32]int8:
		fastpathTV.EncMapInt32Int8V(*v, e)

	case map[int32]int16:
		fastpathTV.EncMapInt32Int16V(v, e)
	case *map[int32]int16:
		fastpathTV.EncMapInt32Int16V(*v, e)

	case map[int32]int32:
		fastpathTV.EncMapInt32Int32V(v, e)
	case *map[int32]int32:
		fastpathTV.EncMapInt32Int32V(*v, e)

	case map[int32]int64:
		fastpathTV.EncMapInt32Int64V(v, e)
	case *map[int32]int64:
		fastpathTV.EncMapInt32Int64V(*v, e)

	case map[int32]float32:
		fastpathTV.EncMapInt32Float32V(v, e)
	case *map[int32]float32:
		fastpathTV.EncMapInt32Float32V(*v, e)

	case map[int32]float64:
		fastpathTV.EncMapInt32Float64V(v, e)
	case *map[int32]float64:
		fastpathTV.EncMapInt32Float64V(*v, e)

	case map[int32]bool:
		fastpathTV.EncMapInt32BoolV(v, e)
	case *map[int32]bool:
		fastpathTV.EncMapInt32BoolV(*v, e)

	case map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(v, e)
	case *map[int64]interface{}:
		fastpathTV.EncMapInt64IntfV(*v, e)

	case map[int64]string:
		fastpathTV.EncMapInt64StringV(v, e)
	case *map[int64]string:
		fastpathTV.EncMapInt64StringV(*v, e)

	case map[int64]uint:
		fastpathTV.EncMapInt64UintV(v, e)
	case *map[int64]uint:
		fastpathTV.EncMapInt64UintV(*v, e)

	case map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(v, e)
	case *map[int64]uint8:
		fastpathTV.EncMapInt64Uint8V(*v, e)

	case map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(v, e)
	case *map[int64]uint16:
		fastpathTV.EncMapInt64Uint16V(*v, e)

	case map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(v, e)
	case *map[int64]uint32:
		fastpathTV.EncMapInt64Uint32V(*v, e)

	case map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(v, e)
	case *map[int64]uint64:
		fastpathTV.EncMapInt64Uint64V(*v, e)

	case map[int64]uintptr:
		fastpathTV.EncMapInt64UintptrV(v, e)
	case *map[int64]uintptr:
		fastpathTV.EncMapInt64UintptrV(*v, e)

	case map[int64]int:
		fastpathTV.EncMapInt64IntV(v, e)
	case *map[int64]int:
		fastpathTV.EncMapInt64IntV(*v, e)

	case map[int64]int8:
		fastpathTV.EncMapInt64Int8V(v, e)
	case *map[int64]int8:
		fastpathTV.EncMapInt64Int8V(*v, e)

	case map[int64]int16:
		fastpathTV.EncMapInt64Int16V(v, e)
	case *map[int64]int16:
		fastpathTV.EncMapInt64Int16V(*v, e)

	case map[int64]int32:
		fastpathTV.EncMapInt64Int32V(v, e)
	case *map[int64]int32:
		fastpathTV.EncMapInt64Int32V(*v, e)

	case map[int64]int64:
		fastpathTV.EncMapInt64Int64V(v, e)
	case *map[int64]int64:
		fastpathTV.EncMapInt64Int64V(*v, e)

	case map[int64]float32:
		fastpathTV.EncMapInt64Float32V(v, e)
	case *map[int64]float32:
		fastpathTV.EncMapInt64Float32V(*v, e)

	case map[int64]float64:
		fastpathTV.EncMapInt64Float64V(v, e)
	case *map[int64]float64:
		fastpathTV.EncMapInt64Float64V(*v, e)

	case map[int64]bool:
		fastpathTV.EncMapInt64BoolV(v, e)
	case *map[int64]bool:
		fastpathTV.EncMapInt64BoolV(*v, e)

	case map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(v, e)
	case *map[bool]interface{}:
		fastpathTV.EncMapBoolIntfV(*v, e)

	case map[bool]string:
		fastpathTV.EncMapBoolStringV(v, e)
	case *map[bool]string:
		fastpathTV.EncMapBoolStringV(*v, e)

	case map[bool]uint:
		fastpathTV.EncMapBoolUintV(v, e)
	case *map[bool]uint:
		fastpathTV.EncMapBoolUintV(*v, e)

	case map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(v, e)
	case *map[bool]uint8:
		fastpathTV.EncMapBoolUint8V(*v, e)

	case map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(v, e)
	case *map[bool]uint16:
		fastpathTV.EncMapBoolUint16V(*v, e)

	case map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(v, e)
	case *map[bool]uint32:
		fastpathTV.EncMapBoolUint32V(*v, e)

	case map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(v, e)
	case *map[bool]uint64:
		fastpathTV.EncMapBoolUint64V(*v, e)

	case map[bool]uintptr:
		fastpathTV.EncMapBoolUintptrV(v, e)
	case *map[bool]uintptr:
		fastpathTV.EncMapBoolUintptrV(*v, e)

	case map[bool]int:
		fastpathTV.EncMapBoolIntV(v, e)
	case *map[bool]int:
		fastpathTV.EncMapBoolIntV(*v, e)

	case map[bool]int8:
		fastpathTV.EncMapBoolInt8V(v, e)
	case *map[bool]int8:
		fastpathTV.EncMapBoolInt8V(*v, e)

	case map[bool]int16:
		fastpathTV.EncMapBoolInt16V(v, e)
	case *map[bool]int16:
		fastpathTV.EncMapBoolInt16V(*v, e)

	case map[bool]int32:
		fastpathTV.EncMapBoolInt32V(v, e)
	case *map[bool]int32:
		fastpathTV.EncMapBoolInt32V(*v, e)

	case map[bool]int64:
		fastpathTV.EncMapBoolInt64V(v, e)
	case *map[bool]int64:
		fastpathTV.EncMapBoolInt64V(*v, e)

	case map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(v, e)
	case *map[bool]float32:
		fastpathTV.EncMapBoolFloat32V(*v, e)

	case map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(v, e)
	case *map[bool]float64:
		fastpathTV.EncMapBoolFloat64V(*v, e)

	case map[bool]bool:
		fastpathTV.EncMapBoolBoolV(v, e)
	case *map[bool]bool:
		fastpathTV.EncMapBoolBoolV(*v, e)

	default:
		_ = v // TODO: workaround https://github.com/golang/go/issues/12927 (remove after go 1.6 release)
		return false
	}
	return true
}

// -- -- fast path functions

func (e *Encoder) fastpathEncSliceIntfR(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceIntfV(rv2i(rv).([]interface{}), e)
	} else {
		fastpathTV.EncSliceIntfV(rv2i(rv).([]interface{}), e)
	}
}
func (_ fastpathT) EncSliceIntfV(v []interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		e.encode(v2)
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceIntfV(v []interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		e.encode(v2)
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceStringR(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceStringV(rv2i(rv).([]string), e)
	} else {
		fastpathTV.EncSliceStringV(rv2i(rv).([]string), e)
	}
}
func (_ fastpathT) EncSliceStringV(v []string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeString(c_UTF8, v2)
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceStringV(v []string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeString(c_UTF8, v2)
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceFloat32R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceFloat32V(rv2i(rv).([]float32), e)
	} else {
		fastpathTV.EncSliceFloat32V(rv2i(rv).([]float32), e)
	}
}
func (_ fastpathT) EncSliceFloat32V(v []float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeFloat32(v2)
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceFloat32V(v []float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeFloat32(v2)
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceFloat64R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceFloat64V(rv2i(rv).([]float64), e)
	} else {
		fastpathTV.EncSliceFloat64V(rv2i(rv).([]float64), e)
	}
}
func (_ fastpathT) EncSliceFloat64V(v []float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeFloat64(v2)
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceFloat64V(v []float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeFloat64(v2)
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceUintR(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceUintV(rv2i(rv).([]uint), e)
	} else {
		fastpathTV.EncSliceUintV(rv2i(rv).([]uint), e)
	}
}
func (_ fastpathT) EncSliceUintV(v []uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceUintV(v []uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceUint16R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceUint16V(rv2i(rv).([]uint16), e)
	} else {
		fastpathTV.EncSliceUint16V(rv2i(rv).([]uint16), e)
	}
}
func (_ fastpathT) EncSliceUint16V(v []uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceUint16V(v []uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceUint32R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceUint32V(rv2i(rv).([]uint32), e)
	} else {
		fastpathTV.EncSliceUint32V(rv2i(rv).([]uint32), e)
	}
}
func (_ fastpathT) EncSliceUint32V(v []uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceUint32V(v []uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceUint64R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceUint64V(rv2i(rv).([]uint64), e)
	} else {
		fastpathTV.EncSliceUint64V(rv2i(rv).([]uint64), e)
	}
}
func (_ fastpathT) EncSliceUint64V(v []uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceUint64V(v []uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeUint(uint64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceUintptrR(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceUintptrV(rv2i(rv).([]uintptr), e)
	} else {
		fastpathTV.EncSliceUintptrV(rv2i(rv).([]uintptr), e)
	}
}
func (_ fastpathT) EncSliceUintptrV(v []uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		e.encode(v2)
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceUintptrV(v []uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		e.encode(v2)
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceIntR(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceIntV(rv2i(rv).([]int), e)
	} else {
		fastpathTV.EncSliceIntV(rv2i(rv).([]int), e)
	}
}
func (_ fastpathT) EncSliceIntV(v []int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceIntV(v []int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceInt8R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceInt8V(rv2i(rv).([]int8), e)
	} else {
		fastpathTV.EncSliceInt8V(rv2i(rv).([]int8), e)
	}
}
func (_ fastpathT) EncSliceInt8V(v []int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceInt8V(v []int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceInt16R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceInt16V(rv2i(rv).([]int16), e)
	} else {
		fastpathTV.EncSliceInt16V(rv2i(rv).([]int16), e)
	}
}
func (_ fastpathT) EncSliceInt16V(v []int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceInt16V(v []int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceInt32R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceInt32V(rv2i(rv).([]int32), e)
	} else {
		fastpathTV.EncSliceInt32V(rv2i(rv).([]int32), e)
	}
}
func (_ fastpathT) EncSliceInt32V(v []int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceInt32V(v []int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceInt64R(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceInt64V(rv2i(rv).([]int64), e)
	} else {
		fastpathTV.EncSliceInt64V(rv2i(rv).([]int64), e)
	}
}
func (_ fastpathT) EncSliceInt64V(v []int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceInt64V(v []int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeInt(int64(v2))
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncSliceBoolR(f *codecFnInfo, rv reflect.Value) {
	if f.ti.mbs {
		fastpathTV.EncAsMapSliceBoolV(rv2i(rv).([]bool), e)
	} else {
		fastpathTV.EncSliceBoolV(rv2i(rv).([]bool), e)
	}
}
func (_ fastpathT) EncSliceBoolV(v []bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteArrayStart(len(v))
	for _, v2 := range v {
		if esep {
			ee.WriteArrayElem()
		}
		ee.EncodeBool(v2)
	}
	ee.WriteArrayEnd()
}

func (_ fastpathT) EncAsMapSliceBoolV(v []bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	if len(v)%2 == 1 {
		e.errorf("mapBySlice requires even slice length, but got %v", len(v))
		return
	}
	ee.WriteMapStart(len(v) / 2)
	for j, v2 := range v {
		if esep {
			if j%2 == 0 {
				ee.WriteMapElemKey()
			} else {
				ee.WriteMapElemValue()
			}
		}
		ee.EncodeBool(v2)
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfIntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfIntfV(rv2i(rv).(map[interface{}]interface{}), e)
}
func (_ fastpathT) EncMapIntfIntfV(v map[interface{}]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfStringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfStringV(rv2i(rv).(map[interface{}]string), e)
}
func (_ fastpathT) EncMapIntfStringV(v map[interface{}]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfUintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfUintV(rv2i(rv).(map[interface{}]uint), e)
}
func (_ fastpathT) EncMapIntfUintV(v map[interface{}]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfUint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfUint8V(rv2i(rv).(map[interface{}]uint8), e)
}
func (_ fastpathT) EncMapIntfUint8V(v map[interface{}]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfUint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfUint16V(rv2i(rv).(map[interface{}]uint16), e)
}
func (_ fastpathT) EncMapIntfUint16V(v map[interface{}]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfUint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfUint32V(rv2i(rv).(map[interface{}]uint32), e)
}
func (_ fastpathT) EncMapIntfUint32V(v map[interface{}]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfUint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfUint64V(rv2i(rv).(map[interface{}]uint64), e)
}
func (_ fastpathT) EncMapIntfUint64V(v map[interface{}]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfUintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfUintptrV(rv2i(rv).(map[interface{}]uintptr), e)
}
func (_ fastpathT) EncMapIntfUintptrV(v map[interface{}]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfIntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfIntV(rv2i(rv).(map[interface{}]int), e)
}
func (_ fastpathT) EncMapIntfIntV(v map[interface{}]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfInt8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfInt8V(rv2i(rv).(map[interface{}]int8), e)
}
func (_ fastpathT) EncMapIntfInt8V(v map[interface{}]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfInt16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfInt16V(rv2i(rv).(map[interface{}]int16), e)
}
func (_ fastpathT) EncMapIntfInt16V(v map[interface{}]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfInt32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfInt32V(rv2i(rv).(map[interface{}]int32), e)
}
func (_ fastpathT) EncMapIntfInt32V(v map[interface{}]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfInt64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfInt64V(rv2i(rv).(map[interface{}]int64), e)
}
func (_ fastpathT) EncMapIntfInt64V(v map[interface{}]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfFloat32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfFloat32V(rv2i(rv).(map[interface{}]float32), e)
}
func (_ fastpathT) EncMapIntfFloat32V(v map[interface{}]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfFloat64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfFloat64V(rv2i(rv).(map[interface{}]float64), e)
}
func (_ fastpathT) EncMapIntfFloat64V(v map[interface{}]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntfBoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntfBoolV(rv2i(rv).(map[interface{}]bool), e)
}
func (_ fastpathT) EncMapIntfBoolV(v map[interface{}]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		var mksv []byte = make([]byte, 0, len(v)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		v2 := make([]bytesI, len(v))
		var i, l int
		var vp *bytesI
		for k2, _ := range v {
			l = len(mksv)
			e2.MustEncode(k2)
			vp = &v2[i]
			vp.v = mksv[l:]
			vp.i = k2
			i++
		}
		sort.Sort(bytesISlice(v2))
		for j := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.asis(v2[j].v)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[v2[j].i])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringIntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringIntfV(rv2i(rv).(map[string]interface{}), e)
}
func (_ fastpathT) EncMapStringIntfV(v map[string]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[string(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringStringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringStringV(rv2i(rv).(map[string]string), e)
}
func (_ fastpathT) EncMapStringStringV(v map[string]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[string(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringUintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringUintV(rv2i(rv).(map[string]uint), e)
}
func (_ fastpathT) EncMapStringUintV(v map[string]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringUint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringUint8V(rv2i(rv).(map[string]uint8), e)
}
func (_ fastpathT) EncMapStringUint8V(v map[string]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringUint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringUint16V(rv2i(rv).(map[string]uint16), e)
}
func (_ fastpathT) EncMapStringUint16V(v map[string]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringUint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringUint32V(rv2i(rv).(map[string]uint32), e)
}
func (_ fastpathT) EncMapStringUint32V(v map[string]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringUint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringUint64V(rv2i(rv).(map[string]uint64), e)
}
func (_ fastpathT) EncMapStringUint64V(v map[string]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringUintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringUintptrV(rv2i(rv).(map[string]uintptr), e)
}
func (_ fastpathT) EncMapStringUintptrV(v map[string]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[string(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringIntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringIntV(rv2i(rv).(map[string]int), e)
}
func (_ fastpathT) EncMapStringIntV(v map[string]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringInt8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringInt8V(rv2i(rv).(map[string]int8), e)
}
func (_ fastpathT) EncMapStringInt8V(v map[string]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringInt16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringInt16V(rv2i(rv).(map[string]int16), e)
}
func (_ fastpathT) EncMapStringInt16V(v map[string]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringInt32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringInt32V(rv2i(rv).(map[string]int32), e)
}
func (_ fastpathT) EncMapStringInt32V(v map[string]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringInt64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringInt64V(rv2i(rv).(map[string]int64), e)
}
func (_ fastpathT) EncMapStringInt64V(v map[string]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[string(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringFloat32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringFloat32V(rv2i(rv).(map[string]float32), e)
}
func (_ fastpathT) EncMapStringFloat32V(v map[string]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[string(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringFloat64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringFloat64V(rv2i(rv).(map[string]float64), e)
}
func (_ fastpathT) EncMapStringFloat64V(v map[string]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[string(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapStringBoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapStringBoolV(rv2i(rv).(map[string]bool), e)
}
func (_ fastpathT) EncMapStringBoolV(v map[string]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	if e.h.Canonical {
		v2 := make([]string, len(v))
		var i int
		for k, _ := range v {
			v2[i] = string(k)
			i++
		}
		sort.Sort(stringSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[string(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(k2)
			} else {
				ee.EncodeString(c_UTF8, k2)
			}
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32IntfV(rv2i(rv).(map[float32]interface{}), e)
}
func (_ fastpathT) EncMapFloat32IntfV(v map[float32]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[float32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32StringV(rv2i(rv).(map[float32]string), e)
}
func (_ fastpathT) EncMapFloat32StringV(v map[float32]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[float32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32UintV(rv2i(rv).(map[float32]uint), e)
}
func (_ fastpathT) EncMapFloat32UintV(v map[float32]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint8V(rv2i(rv).(map[float32]uint8), e)
}
func (_ fastpathT) EncMapFloat32Uint8V(v map[float32]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint16V(rv2i(rv).(map[float32]uint16), e)
}
func (_ fastpathT) EncMapFloat32Uint16V(v map[float32]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint32V(rv2i(rv).(map[float32]uint32), e)
}
func (_ fastpathT) EncMapFloat32Uint32V(v map[float32]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Uint64V(rv2i(rv).(map[float32]uint64), e)
}
func (_ fastpathT) EncMapFloat32Uint64V(v map[float32]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32UintptrV(rv2i(rv).(map[float32]uintptr), e)
}
func (_ fastpathT) EncMapFloat32UintptrV(v map[float32]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[float32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32IntV(rv2i(rv).(map[float32]int), e)
}
func (_ fastpathT) EncMapFloat32IntV(v map[float32]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Int8V(rv2i(rv).(map[float32]int8), e)
}
func (_ fastpathT) EncMapFloat32Int8V(v map[float32]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Int16V(rv2i(rv).(map[float32]int16), e)
}
func (_ fastpathT) EncMapFloat32Int16V(v map[float32]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Int32V(rv2i(rv).(map[float32]int32), e)
}
func (_ fastpathT) EncMapFloat32Int32V(v map[float32]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Int64V(rv2i(rv).(map[float32]int64), e)
}
func (_ fastpathT) EncMapFloat32Int64V(v map[float32]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Float32V(rv2i(rv).(map[float32]float32), e)
}
func (_ fastpathT) EncMapFloat32Float32V(v map[float32]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[float32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32Float64V(rv2i(rv).(map[float32]float64), e)
}
func (_ fastpathT) EncMapFloat32Float64V(v map[float32]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[float32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat32BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat32BoolV(rv2i(rv).(map[float32]bool), e)
}
func (_ fastpathT) EncMapFloat32BoolV(v map[float32]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[float32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64IntfV(rv2i(rv).(map[float64]interface{}), e)
}
func (_ fastpathT) EncMapFloat64IntfV(v map[float64]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[float64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64StringV(rv2i(rv).(map[float64]string), e)
}
func (_ fastpathT) EncMapFloat64StringV(v map[float64]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[float64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64UintV(rv2i(rv).(map[float64]uint), e)
}
func (_ fastpathT) EncMapFloat64UintV(v map[float64]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint8V(rv2i(rv).(map[float64]uint8), e)
}
func (_ fastpathT) EncMapFloat64Uint8V(v map[float64]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint16V(rv2i(rv).(map[float64]uint16), e)
}
func (_ fastpathT) EncMapFloat64Uint16V(v map[float64]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint32V(rv2i(rv).(map[float64]uint32), e)
}
func (_ fastpathT) EncMapFloat64Uint32V(v map[float64]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Uint64V(rv2i(rv).(map[float64]uint64), e)
}
func (_ fastpathT) EncMapFloat64Uint64V(v map[float64]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64UintptrV(rv2i(rv).(map[float64]uintptr), e)
}
func (_ fastpathT) EncMapFloat64UintptrV(v map[float64]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[float64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64IntV(rv2i(rv).(map[float64]int), e)
}
func (_ fastpathT) EncMapFloat64IntV(v map[float64]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Int8V(rv2i(rv).(map[float64]int8), e)
}
func (_ fastpathT) EncMapFloat64Int8V(v map[float64]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Int16V(rv2i(rv).(map[float64]int16), e)
}
func (_ fastpathT) EncMapFloat64Int16V(v map[float64]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Int32V(rv2i(rv).(map[float64]int32), e)
}
func (_ fastpathT) EncMapFloat64Int32V(v map[float64]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Int64V(rv2i(rv).(map[float64]int64), e)
}
func (_ fastpathT) EncMapFloat64Int64V(v map[float64]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[float64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Float32V(rv2i(rv).(map[float64]float32), e)
}
func (_ fastpathT) EncMapFloat64Float32V(v map[float64]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[float64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64Float64V(rv2i(rv).(map[float64]float64), e)
}
func (_ fastpathT) EncMapFloat64Float64V(v map[float64]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[float64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapFloat64BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapFloat64BoolV(rv2i(rv).(map[float64]bool), e)
}
func (_ fastpathT) EncMapFloat64BoolV(v map[float64]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]float64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = float64(k)
			i++
		}
		sort.Sort(floatSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(float64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[float64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintIntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintIntfV(rv2i(rv).(map[uint]interface{}), e)
}
func (_ fastpathT) EncMapUintIntfV(v map[uint]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintStringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintStringV(rv2i(rv).(map[uint]string), e)
}
func (_ fastpathT) EncMapUintStringV(v map[uint]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[uint(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintUintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintUintV(rv2i(rv).(map[uint]uint), e)
}
func (_ fastpathT) EncMapUintUintV(v map[uint]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintUint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintUint8V(rv2i(rv).(map[uint]uint8), e)
}
func (_ fastpathT) EncMapUintUint8V(v map[uint]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintUint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintUint16V(rv2i(rv).(map[uint]uint16), e)
}
func (_ fastpathT) EncMapUintUint16V(v map[uint]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintUint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintUint32V(rv2i(rv).(map[uint]uint32), e)
}
func (_ fastpathT) EncMapUintUint32V(v map[uint]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintUint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintUint64V(rv2i(rv).(map[uint]uint64), e)
}
func (_ fastpathT) EncMapUintUint64V(v map[uint]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintUintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintUintptrV(rv2i(rv).(map[uint]uintptr), e)
}
func (_ fastpathT) EncMapUintUintptrV(v map[uint]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintIntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintIntV(rv2i(rv).(map[uint]int), e)
}
func (_ fastpathT) EncMapUintIntV(v map[uint]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintInt8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintInt8V(rv2i(rv).(map[uint]int8), e)
}
func (_ fastpathT) EncMapUintInt8V(v map[uint]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintInt16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintInt16V(rv2i(rv).(map[uint]int16), e)
}
func (_ fastpathT) EncMapUintInt16V(v map[uint]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintInt32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintInt32V(rv2i(rv).(map[uint]int32), e)
}
func (_ fastpathT) EncMapUintInt32V(v map[uint]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintInt64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintInt64V(rv2i(rv).(map[uint]int64), e)
}
func (_ fastpathT) EncMapUintInt64V(v map[uint]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintFloat32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintFloat32V(rv2i(rv).(map[uint]float32), e)
}
func (_ fastpathT) EncMapUintFloat32V(v map[uint]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[uint(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintFloat64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintFloat64V(rv2i(rv).(map[uint]float64), e)
}
func (_ fastpathT) EncMapUintFloat64V(v map[uint]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[uint(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintBoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintBoolV(rv2i(rv).(map[uint]bool), e)
}
func (_ fastpathT) EncMapUintBoolV(v map[uint]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[uint(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8IntfV(rv2i(rv).(map[uint8]interface{}), e)
}
func (_ fastpathT) EncMapUint8IntfV(v map[uint8]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8StringV(rv2i(rv).(map[uint8]string), e)
}
func (_ fastpathT) EncMapUint8StringV(v map[uint8]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[uint8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8UintV(rv2i(rv).(map[uint8]uint), e)
}
func (_ fastpathT) EncMapUint8UintV(v map[uint8]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Uint8V(rv2i(rv).(map[uint8]uint8), e)
}
func (_ fastpathT) EncMapUint8Uint8V(v map[uint8]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Uint16V(rv2i(rv).(map[uint8]uint16), e)
}
func (_ fastpathT) EncMapUint8Uint16V(v map[uint8]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Uint32V(rv2i(rv).(map[uint8]uint32), e)
}
func (_ fastpathT) EncMapUint8Uint32V(v map[uint8]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Uint64V(rv2i(rv).(map[uint8]uint64), e)
}
func (_ fastpathT) EncMapUint8Uint64V(v map[uint8]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8UintptrV(rv2i(rv).(map[uint8]uintptr), e)
}
func (_ fastpathT) EncMapUint8UintptrV(v map[uint8]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8IntV(rv2i(rv).(map[uint8]int), e)
}
func (_ fastpathT) EncMapUint8IntV(v map[uint8]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Int8V(rv2i(rv).(map[uint8]int8), e)
}
func (_ fastpathT) EncMapUint8Int8V(v map[uint8]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Int16V(rv2i(rv).(map[uint8]int16), e)
}
func (_ fastpathT) EncMapUint8Int16V(v map[uint8]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Int32V(rv2i(rv).(map[uint8]int32), e)
}
func (_ fastpathT) EncMapUint8Int32V(v map[uint8]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Int64V(rv2i(rv).(map[uint8]int64), e)
}
func (_ fastpathT) EncMapUint8Int64V(v map[uint8]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Float32V(rv2i(rv).(map[uint8]float32), e)
}
func (_ fastpathT) EncMapUint8Float32V(v map[uint8]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[uint8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8Float64V(rv2i(rv).(map[uint8]float64), e)
}
func (_ fastpathT) EncMapUint8Float64V(v map[uint8]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[uint8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint8BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint8BoolV(rv2i(rv).(map[uint8]bool), e)
}
func (_ fastpathT) EncMapUint8BoolV(v map[uint8]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[uint8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16IntfV(rv2i(rv).(map[uint16]interface{}), e)
}
func (_ fastpathT) EncMapUint16IntfV(v map[uint16]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16StringV(rv2i(rv).(map[uint16]string), e)
}
func (_ fastpathT) EncMapUint16StringV(v map[uint16]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[uint16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16UintV(rv2i(rv).(map[uint16]uint), e)
}
func (_ fastpathT) EncMapUint16UintV(v map[uint16]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Uint8V(rv2i(rv).(map[uint16]uint8), e)
}
func (_ fastpathT) EncMapUint16Uint8V(v map[uint16]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Uint16V(rv2i(rv).(map[uint16]uint16), e)
}
func (_ fastpathT) EncMapUint16Uint16V(v map[uint16]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Uint32V(rv2i(rv).(map[uint16]uint32), e)
}
func (_ fastpathT) EncMapUint16Uint32V(v map[uint16]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Uint64V(rv2i(rv).(map[uint16]uint64), e)
}
func (_ fastpathT) EncMapUint16Uint64V(v map[uint16]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16UintptrV(rv2i(rv).(map[uint16]uintptr), e)
}
func (_ fastpathT) EncMapUint16UintptrV(v map[uint16]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16IntV(rv2i(rv).(map[uint16]int), e)
}
func (_ fastpathT) EncMapUint16IntV(v map[uint16]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Int8V(rv2i(rv).(map[uint16]int8), e)
}
func (_ fastpathT) EncMapUint16Int8V(v map[uint16]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Int16V(rv2i(rv).(map[uint16]int16), e)
}
func (_ fastpathT) EncMapUint16Int16V(v map[uint16]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Int32V(rv2i(rv).(map[uint16]int32), e)
}
func (_ fastpathT) EncMapUint16Int32V(v map[uint16]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Int64V(rv2i(rv).(map[uint16]int64), e)
}
func (_ fastpathT) EncMapUint16Int64V(v map[uint16]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Float32V(rv2i(rv).(map[uint16]float32), e)
}
func (_ fastpathT) EncMapUint16Float32V(v map[uint16]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[uint16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16Float64V(rv2i(rv).(map[uint16]float64), e)
}
func (_ fastpathT) EncMapUint16Float64V(v map[uint16]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[uint16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint16BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint16BoolV(rv2i(rv).(map[uint16]bool), e)
}
func (_ fastpathT) EncMapUint16BoolV(v map[uint16]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[uint16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32IntfV(rv2i(rv).(map[uint32]interface{}), e)
}
func (_ fastpathT) EncMapUint32IntfV(v map[uint32]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32StringV(rv2i(rv).(map[uint32]string), e)
}
func (_ fastpathT) EncMapUint32StringV(v map[uint32]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[uint32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32UintV(rv2i(rv).(map[uint32]uint), e)
}
func (_ fastpathT) EncMapUint32UintV(v map[uint32]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Uint8V(rv2i(rv).(map[uint32]uint8), e)
}
func (_ fastpathT) EncMapUint32Uint8V(v map[uint32]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Uint16V(rv2i(rv).(map[uint32]uint16), e)
}
func (_ fastpathT) EncMapUint32Uint16V(v map[uint32]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Uint32V(rv2i(rv).(map[uint32]uint32), e)
}
func (_ fastpathT) EncMapUint32Uint32V(v map[uint32]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Uint64V(rv2i(rv).(map[uint32]uint64), e)
}
func (_ fastpathT) EncMapUint32Uint64V(v map[uint32]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32UintptrV(rv2i(rv).(map[uint32]uintptr), e)
}
func (_ fastpathT) EncMapUint32UintptrV(v map[uint32]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32IntV(rv2i(rv).(map[uint32]int), e)
}
func (_ fastpathT) EncMapUint32IntV(v map[uint32]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Int8V(rv2i(rv).(map[uint32]int8), e)
}
func (_ fastpathT) EncMapUint32Int8V(v map[uint32]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Int16V(rv2i(rv).(map[uint32]int16), e)
}
func (_ fastpathT) EncMapUint32Int16V(v map[uint32]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Int32V(rv2i(rv).(map[uint32]int32), e)
}
func (_ fastpathT) EncMapUint32Int32V(v map[uint32]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Int64V(rv2i(rv).(map[uint32]int64), e)
}
func (_ fastpathT) EncMapUint32Int64V(v map[uint32]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Float32V(rv2i(rv).(map[uint32]float32), e)
}
func (_ fastpathT) EncMapUint32Float32V(v map[uint32]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[uint32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32Float64V(rv2i(rv).(map[uint32]float64), e)
}
func (_ fastpathT) EncMapUint32Float64V(v map[uint32]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[uint32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint32BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint32BoolV(rv2i(rv).(map[uint32]bool), e)
}
func (_ fastpathT) EncMapUint32BoolV(v map[uint32]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[uint32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64IntfV(rv2i(rv).(map[uint64]interface{}), e)
}
func (_ fastpathT) EncMapUint64IntfV(v map[uint64]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64StringV(rv2i(rv).(map[uint64]string), e)
}
func (_ fastpathT) EncMapUint64StringV(v map[uint64]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[uint64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64UintV(rv2i(rv).(map[uint64]uint), e)
}
func (_ fastpathT) EncMapUint64UintV(v map[uint64]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Uint8V(rv2i(rv).(map[uint64]uint8), e)
}
func (_ fastpathT) EncMapUint64Uint8V(v map[uint64]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Uint16V(rv2i(rv).(map[uint64]uint16), e)
}
func (_ fastpathT) EncMapUint64Uint16V(v map[uint64]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Uint32V(rv2i(rv).(map[uint64]uint32), e)
}
func (_ fastpathT) EncMapUint64Uint32V(v map[uint64]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Uint64V(rv2i(rv).(map[uint64]uint64), e)
}
func (_ fastpathT) EncMapUint64Uint64V(v map[uint64]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64UintptrV(rv2i(rv).(map[uint64]uintptr), e)
}
func (_ fastpathT) EncMapUint64UintptrV(v map[uint64]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uint64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64IntV(rv2i(rv).(map[uint64]int), e)
}
func (_ fastpathT) EncMapUint64IntV(v map[uint64]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Int8V(rv2i(rv).(map[uint64]int8), e)
}
func (_ fastpathT) EncMapUint64Int8V(v map[uint64]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Int16V(rv2i(rv).(map[uint64]int16), e)
}
func (_ fastpathT) EncMapUint64Int16V(v map[uint64]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Int32V(rv2i(rv).(map[uint64]int32), e)
}
func (_ fastpathT) EncMapUint64Int32V(v map[uint64]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Int64V(rv2i(rv).(map[uint64]int64), e)
}
func (_ fastpathT) EncMapUint64Int64V(v map[uint64]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uint64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Float32V(rv2i(rv).(map[uint64]float32), e)
}
func (_ fastpathT) EncMapUint64Float32V(v map[uint64]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[uint64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64Float64V(rv2i(rv).(map[uint64]float64), e)
}
func (_ fastpathT) EncMapUint64Float64V(v map[uint64]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[uint64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUint64BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUint64BoolV(rv2i(rv).(map[uint64]bool), e)
}
func (_ fastpathT) EncMapUint64BoolV(v map[uint64]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(uint64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[uint64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(uint64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrIntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrIntfV(rv2i(rv).(map[uintptr]interface{}), e)
}
func (_ fastpathT) EncMapUintptrIntfV(v map[uintptr]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uintptr(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrStringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrStringV(rv2i(rv).(map[uintptr]string), e)
}
func (_ fastpathT) EncMapUintptrStringV(v map[uintptr]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[uintptr(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrUintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrUintV(rv2i(rv).(map[uintptr]uint), e)
}
func (_ fastpathT) EncMapUintptrUintV(v map[uintptr]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrUint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrUint8V(rv2i(rv).(map[uintptr]uint8), e)
}
func (_ fastpathT) EncMapUintptrUint8V(v map[uintptr]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrUint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrUint16V(rv2i(rv).(map[uintptr]uint16), e)
}
func (_ fastpathT) EncMapUintptrUint16V(v map[uintptr]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrUint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrUint32V(rv2i(rv).(map[uintptr]uint32), e)
}
func (_ fastpathT) EncMapUintptrUint32V(v map[uintptr]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrUint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrUint64V(rv2i(rv).(map[uintptr]uint64), e)
}
func (_ fastpathT) EncMapUintptrUint64V(v map[uintptr]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrUintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrUintptrV(rv2i(rv).(map[uintptr]uintptr), e)
}
func (_ fastpathT) EncMapUintptrUintptrV(v map[uintptr]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[uintptr(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrIntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrIntV(rv2i(rv).(map[uintptr]int), e)
}
func (_ fastpathT) EncMapUintptrIntV(v map[uintptr]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrInt8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrInt8V(rv2i(rv).(map[uintptr]int8), e)
}
func (_ fastpathT) EncMapUintptrInt8V(v map[uintptr]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrInt16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrInt16V(rv2i(rv).(map[uintptr]int16), e)
}
func (_ fastpathT) EncMapUintptrInt16V(v map[uintptr]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrInt32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrInt32V(rv2i(rv).(map[uintptr]int32), e)
}
func (_ fastpathT) EncMapUintptrInt32V(v map[uintptr]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrInt64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrInt64V(rv2i(rv).(map[uintptr]int64), e)
}
func (_ fastpathT) EncMapUintptrInt64V(v map[uintptr]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[uintptr(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrFloat32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrFloat32V(rv2i(rv).(map[uintptr]float32), e)
}
func (_ fastpathT) EncMapUintptrFloat32V(v map[uintptr]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[uintptr(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrFloat64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrFloat64V(rv2i(rv).(map[uintptr]float64), e)
}
func (_ fastpathT) EncMapUintptrFloat64V(v map[uintptr]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[uintptr(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapUintptrBoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapUintptrBoolV(rv2i(rv).(map[uintptr]bool), e)
}
func (_ fastpathT) EncMapUintptrBoolV(v map[uintptr]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]uint64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = uint64(k)
			i++
		}
		sort.Sort(uintSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(uintptr(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[uintptr(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			e.encode(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntIntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntIntfV(rv2i(rv).(map[int]interface{}), e)
}
func (_ fastpathT) EncMapIntIntfV(v map[int]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntStringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntStringV(rv2i(rv).(map[int]string), e)
}
func (_ fastpathT) EncMapIntStringV(v map[int]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[int(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntUintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntUintV(rv2i(rv).(map[int]uint), e)
}
func (_ fastpathT) EncMapIntUintV(v map[int]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntUint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntUint8V(rv2i(rv).(map[int]uint8), e)
}
func (_ fastpathT) EncMapIntUint8V(v map[int]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntUint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntUint16V(rv2i(rv).(map[int]uint16), e)
}
func (_ fastpathT) EncMapIntUint16V(v map[int]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntUint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntUint32V(rv2i(rv).(map[int]uint32), e)
}
func (_ fastpathT) EncMapIntUint32V(v map[int]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntUint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntUint64V(rv2i(rv).(map[int]uint64), e)
}
func (_ fastpathT) EncMapIntUint64V(v map[int]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntUintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntUintptrV(rv2i(rv).(map[int]uintptr), e)
}
func (_ fastpathT) EncMapIntUintptrV(v map[int]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntIntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntIntV(rv2i(rv).(map[int]int), e)
}
func (_ fastpathT) EncMapIntIntV(v map[int]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntInt8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntInt8V(rv2i(rv).(map[int]int8), e)
}
func (_ fastpathT) EncMapIntInt8V(v map[int]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntInt16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntInt16V(rv2i(rv).(map[int]int16), e)
}
func (_ fastpathT) EncMapIntInt16V(v map[int]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntInt32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntInt32V(rv2i(rv).(map[int]int32), e)
}
func (_ fastpathT) EncMapIntInt32V(v map[int]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntInt64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntInt64V(rv2i(rv).(map[int]int64), e)
}
func (_ fastpathT) EncMapIntInt64V(v map[int]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntFloat32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntFloat32V(rv2i(rv).(map[int]float32), e)
}
func (_ fastpathT) EncMapIntFloat32V(v map[int]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[int(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntFloat64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntFloat64V(rv2i(rv).(map[int]float64), e)
}
func (_ fastpathT) EncMapIntFloat64V(v map[int]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[int(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapIntBoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapIntBoolV(rv2i(rv).(map[int]bool), e)
}
func (_ fastpathT) EncMapIntBoolV(v map[int]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[int(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8IntfV(rv2i(rv).(map[int8]interface{}), e)
}
func (_ fastpathT) EncMapInt8IntfV(v map[int8]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8StringV(rv2i(rv).(map[int8]string), e)
}
func (_ fastpathT) EncMapInt8StringV(v map[int8]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[int8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8UintV(rv2i(rv).(map[int8]uint), e)
}
func (_ fastpathT) EncMapInt8UintV(v map[int8]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Uint8V(rv2i(rv).(map[int8]uint8), e)
}
func (_ fastpathT) EncMapInt8Uint8V(v map[int8]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Uint16V(rv2i(rv).(map[int8]uint16), e)
}
func (_ fastpathT) EncMapInt8Uint16V(v map[int8]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Uint32V(rv2i(rv).(map[int8]uint32), e)
}
func (_ fastpathT) EncMapInt8Uint32V(v map[int8]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Uint64V(rv2i(rv).(map[int8]uint64), e)
}
func (_ fastpathT) EncMapInt8Uint64V(v map[int8]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8UintptrV(rv2i(rv).(map[int8]uintptr), e)
}
func (_ fastpathT) EncMapInt8UintptrV(v map[int8]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8IntV(rv2i(rv).(map[int8]int), e)
}
func (_ fastpathT) EncMapInt8IntV(v map[int8]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Int8V(rv2i(rv).(map[int8]int8), e)
}
func (_ fastpathT) EncMapInt8Int8V(v map[int8]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Int16V(rv2i(rv).(map[int8]int16), e)
}
func (_ fastpathT) EncMapInt8Int16V(v map[int8]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Int32V(rv2i(rv).(map[int8]int32), e)
}
func (_ fastpathT) EncMapInt8Int32V(v map[int8]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Int64V(rv2i(rv).(map[int8]int64), e)
}
func (_ fastpathT) EncMapInt8Int64V(v map[int8]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int8(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Float32V(rv2i(rv).(map[int8]float32), e)
}
func (_ fastpathT) EncMapInt8Float32V(v map[int8]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[int8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8Float64V(rv2i(rv).(map[int8]float64), e)
}
func (_ fastpathT) EncMapInt8Float64V(v map[int8]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[int8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt8BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt8BoolV(rv2i(rv).(map[int8]bool), e)
}
func (_ fastpathT) EncMapInt8BoolV(v map[int8]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int8(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[int8(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16IntfV(rv2i(rv).(map[int16]interface{}), e)
}
func (_ fastpathT) EncMapInt16IntfV(v map[int16]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16StringV(rv2i(rv).(map[int16]string), e)
}
func (_ fastpathT) EncMapInt16StringV(v map[int16]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[int16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16UintV(rv2i(rv).(map[int16]uint), e)
}
func (_ fastpathT) EncMapInt16UintV(v map[int16]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Uint8V(rv2i(rv).(map[int16]uint8), e)
}
func (_ fastpathT) EncMapInt16Uint8V(v map[int16]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Uint16V(rv2i(rv).(map[int16]uint16), e)
}
func (_ fastpathT) EncMapInt16Uint16V(v map[int16]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Uint32V(rv2i(rv).(map[int16]uint32), e)
}
func (_ fastpathT) EncMapInt16Uint32V(v map[int16]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Uint64V(rv2i(rv).(map[int16]uint64), e)
}
func (_ fastpathT) EncMapInt16Uint64V(v map[int16]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16UintptrV(rv2i(rv).(map[int16]uintptr), e)
}
func (_ fastpathT) EncMapInt16UintptrV(v map[int16]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16IntV(rv2i(rv).(map[int16]int), e)
}
func (_ fastpathT) EncMapInt16IntV(v map[int16]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Int8V(rv2i(rv).(map[int16]int8), e)
}
func (_ fastpathT) EncMapInt16Int8V(v map[int16]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Int16V(rv2i(rv).(map[int16]int16), e)
}
func (_ fastpathT) EncMapInt16Int16V(v map[int16]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Int32V(rv2i(rv).(map[int16]int32), e)
}
func (_ fastpathT) EncMapInt16Int32V(v map[int16]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Int64V(rv2i(rv).(map[int16]int64), e)
}
func (_ fastpathT) EncMapInt16Int64V(v map[int16]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int16(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Float32V(rv2i(rv).(map[int16]float32), e)
}
func (_ fastpathT) EncMapInt16Float32V(v map[int16]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[int16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16Float64V(rv2i(rv).(map[int16]float64), e)
}
func (_ fastpathT) EncMapInt16Float64V(v map[int16]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[int16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt16BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt16BoolV(rv2i(rv).(map[int16]bool), e)
}
func (_ fastpathT) EncMapInt16BoolV(v map[int16]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int16(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[int16(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32IntfV(rv2i(rv).(map[int32]interface{}), e)
}
func (_ fastpathT) EncMapInt32IntfV(v map[int32]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32StringV(rv2i(rv).(map[int32]string), e)
}
func (_ fastpathT) EncMapInt32StringV(v map[int32]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[int32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32UintV(rv2i(rv).(map[int32]uint), e)
}
func (_ fastpathT) EncMapInt32UintV(v map[int32]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Uint8V(rv2i(rv).(map[int32]uint8), e)
}
func (_ fastpathT) EncMapInt32Uint8V(v map[int32]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Uint16V(rv2i(rv).(map[int32]uint16), e)
}
func (_ fastpathT) EncMapInt32Uint16V(v map[int32]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Uint32V(rv2i(rv).(map[int32]uint32), e)
}
func (_ fastpathT) EncMapInt32Uint32V(v map[int32]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Uint64V(rv2i(rv).(map[int32]uint64), e)
}
func (_ fastpathT) EncMapInt32Uint64V(v map[int32]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32UintptrV(rv2i(rv).(map[int32]uintptr), e)
}
func (_ fastpathT) EncMapInt32UintptrV(v map[int32]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32IntV(rv2i(rv).(map[int32]int), e)
}
func (_ fastpathT) EncMapInt32IntV(v map[int32]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Int8V(rv2i(rv).(map[int32]int8), e)
}
func (_ fastpathT) EncMapInt32Int8V(v map[int32]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Int16V(rv2i(rv).(map[int32]int16), e)
}
func (_ fastpathT) EncMapInt32Int16V(v map[int32]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Int32V(rv2i(rv).(map[int32]int32), e)
}
func (_ fastpathT) EncMapInt32Int32V(v map[int32]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Int64V(rv2i(rv).(map[int32]int64), e)
}
func (_ fastpathT) EncMapInt32Int64V(v map[int32]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int32(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Float32V(rv2i(rv).(map[int32]float32), e)
}
func (_ fastpathT) EncMapInt32Float32V(v map[int32]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[int32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32Float64V(rv2i(rv).(map[int32]float64), e)
}
func (_ fastpathT) EncMapInt32Float64V(v map[int32]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[int32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt32BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt32BoolV(rv2i(rv).(map[int32]bool), e)
}
func (_ fastpathT) EncMapInt32BoolV(v map[int32]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int32(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[int32(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64IntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64IntfV(rv2i(rv).(map[int64]interface{}), e)
}
func (_ fastpathT) EncMapInt64IntfV(v map[int64]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64StringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64StringV(rv2i(rv).(map[int64]string), e)
}
func (_ fastpathT) EncMapInt64StringV(v map[int64]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[int64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64UintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64UintV(rv2i(rv).(map[int64]uint), e)
}
func (_ fastpathT) EncMapInt64UintV(v map[int64]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Uint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Uint8V(rv2i(rv).(map[int64]uint8), e)
}
func (_ fastpathT) EncMapInt64Uint8V(v map[int64]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Uint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Uint16V(rv2i(rv).(map[int64]uint16), e)
}
func (_ fastpathT) EncMapInt64Uint16V(v map[int64]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Uint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Uint32V(rv2i(rv).(map[int64]uint32), e)
}
func (_ fastpathT) EncMapInt64Uint32V(v map[int64]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Uint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Uint64V(rv2i(rv).(map[int64]uint64), e)
}
func (_ fastpathT) EncMapInt64Uint64V(v map[int64]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64UintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64UintptrV(rv2i(rv).(map[int64]uintptr), e)
}
func (_ fastpathT) EncMapInt64UintptrV(v map[int64]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[int64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64IntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64IntV(rv2i(rv).(map[int64]int), e)
}
func (_ fastpathT) EncMapInt64IntV(v map[int64]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Int8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Int8V(rv2i(rv).(map[int64]int8), e)
}
func (_ fastpathT) EncMapInt64Int8V(v map[int64]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Int16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Int16V(rv2i(rv).(map[int64]int16), e)
}
func (_ fastpathT) EncMapInt64Int16V(v map[int64]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Int32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Int32V(rv2i(rv).(map[int64]int32), e)
}
func (_ fastpathT) EncMapInt64Int32V(v map[int64]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Int64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Int64V(rv2i(rv).(map[int64]int64), e)
}
func (_ fastpathT) EncMapInt64Int64V(v map[int64]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[int64(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Float32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Float32V(rv2i(rv).(map[int64]float32), e)
}
func (_ fastpathT) EncMapInt64Float32V(v map[int64]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[int64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64Float64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64Float64V(rv2i(rv).(map[int64]float64), e)
}
func (_ fastpathT) EncMapInt64Float64V(v map[int64]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[int64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapInt64BoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapInt64BoolV(rv2i(rv).(map[int64]bool), e)
}
func (_ fastpathT) EncMapInt64BoolV(v map[int64]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]int64, len(v))
		var i int
		for k, _ := range v {
			v2[i] = int64(k)
			i++
		}
		sort.Sort(intSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(int64(k2)))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[int64(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(int64(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolIntfR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolIntfV(rv2i(rv).(map[bool]interface{}), e)
}
func (_ fastpathT) EncMapBoolIntfV(v map[bool]interface{}, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[bool(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolStringR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolStringV(rv2i(rv).(map[bool]string), e)
}
func (_ fastpathT) EncMapBoolStringV(v map[bool]string, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v[bool(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeString(c_UTF8, v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolUintR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolUintV(rv2i(rv).(map[bool]uint), e)
}
func (_ fastpathT) EncMapBoolUintV(v map[bool]uint, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolUint8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolUint8V(rv2i(rv).(map[bool]uint8), e)
}
func (_ fastpathT) EncMapBoolUint8V(v map[bool]uint8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolUint16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolUint16V(rv2i(rv).(map[bool]uint16), e)
}
func (_ fastpathT) EncMapBoolUint16V(v map[bool]uint16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolUint32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolUint32V(rv2i(rv).(map[bool]uint32), e)
}
func (_ fastpathT) EncMapBoolUint32V(v map[bool]uint32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolUint64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolUint64V(rv2i(rv).(map[bool]uint64), e)
}
func (_ fastpathT) EncMapBoolUint64V(v map[bool]uint64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeUint(uint64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolUintptrR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolUintptrV(rv2i(rv).(map[bool]uintptr), e)
}
func (_ fastpathT) EncMapBoolUintptrV(v map[bool]uintptr, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v[bool(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			e.encode(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolIntR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolIntV(rv2i(rv).(map[bool]int), e)
}
func (_ fastpathT) EncMapBoolIntV(v map[bool]int, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolInt8R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolInt8V(rv2i(rv).(map[bool]int8), e)
}
func (_ fastpathT) EncMapBoolInt8V(v map[bool]int8, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolInt16R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolInt16V(rv2i(rv).(map[bool]int16), e)
}
func (_ fastpathT) EncMapBoolInt16V(v map[bool]int16, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolInt32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolInt32V(rv2i(rv).(map[bool]int32), e)
}
func (_ fastpathT) EncMapBoolInt32V(v map[bool]int32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolInt64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolInt64V(rv2i(rv).(map[bool]int64), e)
}
func (_ fastpathT) EncMapBoolInt64V(v map[bool]int64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v[bool(k2)]))
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeInt(int64(v2))
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolFloat32R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolFloat32V(rv2i(rv).(map[bool]float32), e)
}
func (_ fastpathT) EncMapBoolFloat32V(v map[bool]float32, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v[bool(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat32(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolFloat64R(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolFloat64V(rv2i(rv).(map[bool]float64), e)
}
func (_ fastpathT) EncMapBoolFloat64V(v map[bool]float64, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v[bool(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeFloat64(v2)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) fastpathEncMapBoolBoolR(f *codecFnInfo, rv reflect.Value) {
	fastpathTV.EncMapBoolBoolV(rv2i(rv).(map[bool]bool), e)
}
func (_ fastpathT) EncMapBoolBoolV(v map[bool]bool, e *Encoder) {
	ee, esep := e.e, e.hh.hasElemSeparators()
	ee.WriteMapStart(len(v))
	if e.h.Canonical {
		v2 := make([]bool, len(v))
		var i int
		for k, _ := range v {
			v2[i] = bool(k)
			i++
		}
		sort.Sort(boolSlice(v2))
		for _, k2 := range v2 {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(bool(k2))
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v[bool(k2)])
		}
	} else {
		for k2, v2 := range v {
			if esep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(k2)
			if esep {
				ee.WriteMapElemValue()
			}
			ee.EncodeBool(v2)
		}
	}
	ee.WriteMapEnd()
}

// -- decode

// -- -- fast path type switch
func fastpathDecodeTypeSwitch(iv interface{}, d *Decoder) bool {
	switch v := iv.(type) {

	case []interface{}:
		fastpathTV.DecSliceIntfV(v, false, d)
	case *[]interface{}:
		if v2, changed2 := fastpathTV.DecSliceIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]interface{}:
		fastpathTV.DecMapIntfIntfV(v, false, d)
	case *map[interface{}]interface{}:
		if v2, changed2 := fastpathTV.DecMapIntfIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]string:
		fastpathTV.DecMapIntfStringV(v, false, d)
	case *map[interface{}]string:
		if v2, changed2 := fastpathTV.DecMapIntfStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]uint:
		fastpathTV.DecMapIntfUintV(v, false, d)
	case *map[interface{}]uint:
		if v2, changed2 := fastpathTV.DecMapIntfUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]uint8:
		fastpathTV.DecMapIntfUint8V(v, false, d)
	case *map[interface{}]uint8:
		if v2, changed2 := fastpathTV.DecMapIntfUint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]uint16:
		fastpathTV.DecMapIntfUint16V(v, false, d)
	case *map[interface{}]uint16:
		if v2, changed2 := fastpathTV.DecMapIntfUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]uint32:
		fastpathTV.DecMapIntfUint32V(v, false, d)
	case *map[interface{}]uint32:
		if v2, changed2 := fastpathTV.DecMapIntfUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]uint64:
		fastpathTV.DecMapIntfUint64V(v, false, d)
	case *map[interface{}]uint64:
		if v2, changed2 := fastpathTV.DecMapIntfUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]uintptr:
		fastpathTV.DecMapIntfUintptrV(v, false, d)
	case *map[interface{}]uintptr:
		if v2, changed2 := fastpathTV.DecMapIntfUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]int:
		fastpathTV.DecMapIntfIntV(v, false, d)
	case *map[interface{}]int:
		if v2, changed2 := fastpathTV.DecMapIntfIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]int8:
		fastpathTV.DecMapIntfInt8V(v, false, d)
	case *map[interface{}]int8:
		if v2, changed2 := fastpathTV.DecMapIntfInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]int16:
		fastpathTV.DecMapIntfInt16V(v, false, d)
	case *map[interface{}]int16:
		if v2, changed2 := fastpathTV.DecMapIntfInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]int32:
		fastpathTV.DecMapIntfInt32V(v, false, d)
	case *map[interface{}]int32:
		if v2, changed2 := fastpathTV.DecMapIntfInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]int64:
		fastpathTV.DecMapIntfInt64V(v, false, d)
	case *map[interface{}]int64:
		if v2, changed2 := fastpathTV.DecMapIntfInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]float32:
		fastpathTV.DecMapIntfFloat32V(v, false, d)
	case *map[interface{}]float32:
		if v2, changed2 := fastpathTV.DecMapIntfFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]float64:
		fastpathTV.DecMapIntfFloat64V(v, false, d)
	case *map[interface{}]float64:
		if v2, changed2 := fastpathTV.DecMapIntfFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[interface{}]bool:
		fastpathTV.DecMapIntfBoolV(v, false, d)
	case *map[interface{}]bool:
		if v2, changed2 := fastpathTV.DecMapIntfBoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []string:
		fastpathTV.DecSliceStringV(v, false, d)
	case *[]string:
		if v2, changed2 := fastpathTV.DecSliceStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]interface{}:
		fastpathTV.DecMapStringIntfV(v, false, d)
	case *map[string]interface{}:
		if v2, changed2 := fastpathTV.DecMapStringIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]string:
		fastpathTV.DecMapStringStringV(v, false, d)
	case *map[string]string:
		if v2, changed2 := fastpathTV.DecMapStringStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]uint:
		fastpathTV.DecMapStringUintV(v, false, d)
	case *map[string]uint:
		if v2, changed2 := fastpathTV.DecMapStringUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]uint8:
		fastpathTV.DecMapStringUint8V(v, false, d)
	case *map[string]uint8:
		if v2, changed2 := fastpathTV.DecMapStringUint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]uint16:
		fastpathTV.DecMapStringUint16V(v, false, d)
	case *map[string]uint16:
		if v2, changed2 := fastpathTV.DecMapStringUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]uint32:
		fastpathTV.DecMapStringUint32V(v, false, d)
	case *map[string]uint32:
		if v2, changed2 := fastpathTV.DecMapStringUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]uint64:
		fastpathTV.DecMapStringUint64V(v, false, d)
	case *map[string]uint64:
		if v2, changed2 := fastpathTV.DecMapStringUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]uintptr:
		fastpathTV.DecMapStringUintptrV(v, false, d)
	case *map[string]uintptr:
		if v2, changed2 := fastpathTV.DecMapStringUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]int:
		fastpathTV.DecMapStringIntV(v, false, d)
	case *map[string]int:
		if v2, changed2 := fastpathTV.DecMapStringIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]int8:
		fastpathTV.DecMapStringInt8V(v, false, d)
	case *map[string]int8:
		if v2, changed2 := fastpathTV.DecMapStringInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]int16:
		fastpathTV.DecMapStringInt16V(v, false, d)
	case *map[string]int16:
		if v2, changed2 := fastpathTV.DecMapStringInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]int32:
		fastpathTV.DecMapStringInt32V(v, false, d)
	case *map[string]int32:
		if v2, changed2 := fastpathTV.DecMapStringInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]int64:
		fastpathTV.DecMapStringInt64V(v, false, d)
	case *map[string]int64:
		if v2, changed2 := fastpathTV.DecMapStringInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]float32:
		fastpathTV.DecMapStringFloat32V(v, false, d)
	case *map[string]float32:
		if v2, changed2 := fastpathTV.DecMapStringFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]float64:
		fastpathTV.DecMapStringFloat64V(v, false, d)
	case *map[string]float64:
		if v2, changed2 := fastpathTV.DecMapStringFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[string]bool:
		fastpathTV.DecMapStringBoolV(v, false, d)
	case *map[string]bool:
		if v2, changed2 := fastpathTV.DecMapStringBoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []float32:
		fastpathTV.DecSliceFloat32V(v, false, d)
	case *[]float32:
		if v2, changed2 := fastpathTV.DecSliceFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]interface{}:
		fastpathTV.DecMapFloat32IntfV(v, false, d)
	case *map[float32]interface{}:
		if v2, changed2 := fastpathTV.DecMapFloat32IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]string:
		fastpathTV.DecMapFloat32StringV(v, false, d)
	case *map[float32]string:
		if v2, changed2 := fastpathTV.DecMapFloat32StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]uint:
		fastpathTV.DecMapFloat32UintV(v, false, d)
	case *map[float32]uint:
		if v2, changed2 := fastpathTV.DecMapFloat32UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]uint8:
		fastpathTV.DecMapFloat32Uint8V(v, false, d)
	case *map[float32]uint8:
		if v2, changed2 := fastpathTV.DecMapFloat32Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]uint16:
		fastpathTV.DecMapFloat32Uint16V(v, false, d)
	case *map[float32]uint16:
		if v2, changed2 := fastpathTV.DecMapFloat32Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]uint32:
		fastpathTV.DecMapFloat32Uint32V(v, false, d)
	case *map[float32]uint32:
		if v2, changed2 := fastpathTV.DecMapFloat32Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]uint64:
		fastpathTV.DecMapFloat32Uint64V(v, false, d)
	case *map[float32]uint64:
		if v2, changed2 := fastpathTV.DecMapFloat32Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]uintptr:
		fastpathTV.DecMapFloat32UintptrV(v, false, d)
	case *map[float32]uintptr:
		if v2, changed2 := fastpathTV.DecMapFloat32UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]int:
		fastpathTV.DecMapFloat32IntV(v, false, d)
	case *map[float32]int:
		if v2, changed2 := fastpathTV.DecMapFloat32IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]int8:
		fastpathTV.DecMapFloat32Int8V(v, false, d)
	case *map[float32]int8:
		if v2, changed2 := fastpathTV.DecMapFloat32Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]int16:
		fastpathTV.DecMapFloat32Int16V(v, false, d)
	case *map[float32]int16:
		if v2, changed2 := fastpathTV.DecMapFloat32Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]int32:
		fastpathTV.DecMapFloat32Int32V(v, false, d)
	case *map[float32]int32:
		if v2, changed2 := fastpathTV.DecMapFloat32Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]int64:
		fastpathTV.DecMapFloat32Int64V(v, false, d)
	case *map[float32]int64:
		if v2, changed2 := fastpathTV.DecMapFloat32Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]float32:
		fastpathTV.DecMapFloat32Float32V(v, false, d)
	case *map[float32]float32:
		if v2, changed2 := fastpathTV.DecMapFloat32Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]float64:
		fastpathTV.DecMapFloat32Float64V(v, false, d)
	case *map[float32]float64:
		if v2, changed2 := fastpathTV.DecMapFloat32Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float32]bool:
		fastpathTV.DecMapFloat32BoolV(v, false, d)
	case *map[float32]bool:
		if v2, changed2 := fastpathTV.DecMapFloat32BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []float64:
		fastpathTV.DecSliceFloat64V(v, false, d)
	case *[]float64:
		if v2, changed2 := fastpathTV.DecSliceFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]interface{}:
		fastpathTV.DecMapFloat64IntfV(v, false, d)
	case *map[float64]interface{}:
		if v2, changed2 := fastpathTV.DecMapFloat64IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]string:
		fastpathTV.DecMapFloat64StringV(v, false, d)
	case *map[float64]string:
		if v2, changed2 := fastpathTV.DecMapFloat64StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]uint:
		fastpathTV.DecMapFloat64UintV(v, false, d)
	case *map[float64]uint:
		if v2, changed2 := fastpathTV.DecMapFloat64UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]uint8:
		fastpathTV.DecMapFloat64Uint8V(v, false, d)
	case *map[float64]uint8:
		if v2, changed2 := fastpathTV.DecMapFloat64Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]uint16:
		fastpathTV.DecMapFloat64Uint16V(v, false, d)
	case *map[float64]uint16:
		if v2, changed2 := fastpathTV.DecMapFloat64Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]uint32:
		fastpathTV.DecMapFloat64Uint32V(v, false, d)
	case *map[float64]uint32:
		if v2, changed2 := fastpathTV.DecMapFloat64Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]uint64:
		fastpathTV.DecMapFloat64Uint64V(v, false, d)
	case *map[float64]uint64:
		if v2, changed2 := fastpathTV.DecMapFloat64Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]uintptr:
		fastpathTV.DecMapFloat64UintptrV(v, false, d)
	case *map[float64]uintptr:
		if v2, changed2 := fastpathTV.DecMapFloat64UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]int:
		fastpathTV.DecMapFloat64IntV(v, false, d)
	case *map[float64]int:
		if v2, changed2 := fastpathTV.DecMapFloat64IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]int8:
		fastpathTV.DecMapFloat64Int8V(v, false, d)
	case *map[float64]int8:
		if v2, changed2 := fastpathTV.DecMapFloat64Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]int16:
		fastpathTV.DecMapFloat64Int16V(v, false, d)
	case *map[float64]int16:
		if v2, changed2 := fastpathTV.DecMapFloat64Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]int32:
		fastpathTV.DecMapFloat64Int32V(v, false, d)
	case *map[float64]int32:
		if v2, changed2 := fastpathTV.DecMapFloat64Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]int64:
		fastpathTV.DecMapFloat64Int64V(v, false, d)
	case *map[float64]int64:
		if v2, changed2 := fastpathTV.DecMapFloat64Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]float32:
		fastpathTV.DecMapFloat64Float32V(v, false, d)
	case *map[float64]float32:
		if v2, changed2 := fastpathTV.DecMapFloat64Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]float64:
		fastpathTV.DecMapFloat64Float64V(v, false, d)
	case *map[float64]float64:
		if v2, changed2 := fastpathTV.DecMapFloat64Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[float64]bool:
		fastpathTV.DecMapFloat64BoolV(v, false, d)
	case *map[float64]bool:
		if v2, changed2 := fastpathTV.DecMapFloat64BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []uint:
		fastpathTV.DecSliceUintV(v, false, d)
	case *[]uint:
		if v2, changed2 := fastpathTV.DecSliceUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]interface{}:
		fastpathTV.DecMapUintIntfV(v, false, d)
	case *map[uint]interface{}:
		if v2, changed2 := fastpathTV.DecMapUintIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]string:
		fastpathTV.DecMapUintStringV(v, false, d)
	case *map[uint]string:
		if v2, changed2 := fastpathTV.DecMapUintStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]uint:
		fastpathTV.DecMapUintUintV(v, false, d)
	case *map[uint]uint:
		if v2, changed2 := fastpathTV.DecMapUintUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]uint8:
		fastpathTV.DecMapUintUint8V(v, false, d)
	case *map[uint]uint8:
		if v2, changed2 := fastpathTV.DecMapUintUint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]uint16:
		fastpathTV.DecMapUintUint16V(v, false, d)
	case *map[uint]uint16:
		if v2, changed2 := fastpathTV.DecMapUintUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]uint32:
		fastpathTV.DecMapUintUint32V(v, false, d)
	case *map[uint]uint32:
		if v2, changed2 := fastpathTV.DecMapUintUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]uint64:
		fastpathTV.DecMapUintUint64V(v, false, d)
	case *map[uint]uint64:
		if v2, changed2 := fastpathTV.DecMapUintUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]uintptr:
		fastpathTV.DecMapUintUintptrV(v, false, d)
	case *map[uint]uintptr:
		if v2, changed2 := fastpathTV.DecMapUintUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]int:
		fastpathTV.DecMapUintIntV(v, false, d)
	case *map[uint]int:
		if v2, changed2 := fastpathTV.DecMapUintIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]int8:
		fastpathTV.DecMapUintInt8V(v, false, d)
	case *map[uint]int8:
		if v2, changed2 := fastpathTV.DecMapUintInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]int16:
		fastpathTV.DecMapUintInt16V(v, false, d)
	case *map[uint]int16:
		if v2, changed2 := fastpathTV.DecMapUintInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]int32:
		fastpathTV.DecMapUintInt32V(v, false, d)
	case *map[uint]int32:
		if v2, changed2 := fastpathTV.DecMapUintInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]int64:
		fastpathTV.DecMapUintInt64V(v, false, d)
	case *map[uint]int64:
		if v2, changed2 := fastpathTV.DecMapUintInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]float32:
		fastpathTV.DecMapUintFloat32V(v, false, d)
	case *map[uint]float32:
		if v2, changed2 := fastpathTV.DecMapUintFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]float64:
		fastpathTV.DecMapUintFloat64V(v, false, d)
	case *map[uint]float64:
		if v2, changed2 := fastpathTV.DecMapUintFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint]bool:
		fastpathTV.DecMapUintBoolV(v, false, d)
	case *map[uint]bool:
		if v2, changed2 := fastpathTV.DecMapUintBoolV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]interface{}:
		fastpathTV.DecMapUint8IntfV(v, false, d)
	case *map[uint8]interface{}:
		if v2, changed2 := fastpathTV.DecMapUint8IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]string:
		fastpathTV.DecMapUint8StringV(v, false, d)
	case *map[uint8]string:
		if v2, changed2 := fastpathTV.DecMapUint8StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]uint:
		fastpathTV.DecMapUint8UintV(v, false, d)
	case *map[uint8]uint:
		if v2, changed2 := fastpathTV.DecMapUint8UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]uint8:
		fastpathTV.DecMapUint8Uint8V(v, false, d)
	case *map[uint8]uint8:
		if v2, changed2 := fastpathTV.DecMapUint8Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]uint16:
		fastpathTV.DecMapUint8Uint16V(v, false, d)
	case *map[uint8]uint16:
		if v2, changed2 := fastpathTV.DecMapUint8Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]uint32:
		fastpathTV.DecMapUint8Uint32V(v, false, d)
	case *map[uint8]uint32:
		if v2, changed2 := fastpathTV.DecMapUint8Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]uint64:
		fastpathTV.DecMapUint8Uint64V(v, false, d)
	case *map[uint8]uint64:
		if v2, changed2 := fastpathTV.DecMapUint8Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]uintptr:
		fastpathTV.DecMapUint8UintptrV(v, false, d)
	case *map[uint8]uintptr:
		if v2, changed2 := fastpathTV.DecMapUint8UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]int:
		fastpathTV.DecMapUint8IntV(v, false, d)
	case *map[uint8]int:
		if v2, changed2 := fastpathTV.DecMapUint8IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]int8:
		fastpathTV.DecMapUint8Int8V(v, false, d)
	case *map[uint8]int8:
		if v2, changed2 := fastpathTV.DecMapUint8Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]int16:
		fastpathTV.DecMapUint8Int16V(v, false, d)
	case *map[uint8]int16:
		if v2, changed2 := fastpathTV.DecMapUint8Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]int32:
		fastpathTV.DecMapUint8Int32V(v, false, d)
	case *map[uint8]int32:
		if v2, changed2 := fastpathTV.DecMapUint8Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]int64:
		fastpathTV.DecMapUint8Int64V(v, false, d)
	case *map[uint8]int64:
		if v2, changed2 := fastpathTV.DecMapUint8Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]float32:
		fastpathTV.DecMapUint8Float32V(v, false, d)
	case *map[uint8]float32:
		if v2, changed2 := fastpathTV.DecMapUint8Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]float64:
		fastpathTV.DecMapUint8Float64V(v, false, d)
	case *map[uint8]float64:
		if v2, changed2 := fastpathTV.DecMapUint8Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint8]bool:
		fastpathTV.DecMapUint8BoolV(v, false, d)
	case *map[uint8]bool:
		if v2, changed2 := fastpathTV.DecMapUint8BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []uint16:
		fastpathTV.DecSliceUint16V(v, false, d)
	case *[]uint16:
		if v2, changed2 := fastpathTV.DecSliceUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]interface{}:
		fastpathTV.DecMapUint16IntfV(v, false, d)
	case *map[uint16]interface{}:
		if v2, changed2 := fastpathTV.DecMapUint16IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]string:
		fastpathTV.DecMapUint16StringV(v, false, d)
	case *map[uint16]string:
		if v2, changed2 := fastpathTV.DecMapUint16StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]uint:
		fastpathTV.DecMapUint16UintV(v, false, d)
	case *map[uint16]uint:
		if v2, changed2 := fastpathTV.DecMapUint16UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]uint8:
		fastpathTV.DecMapUint16Uint8V(v, false, d)
	case *map[uint16]uint8:
		if v2, changed2 := fastpathTV.DecMapUint16Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]uint16:
		fastpathTV.DecMapUint16Uint16V(v, false, d)
	case *map[uint16]uint16:
		if v2, changed2 := fastpathTV.DecMapUint16Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]uint32:
		fastpathTV.DecMapUint16Uint32V(v, false, d)
	case *map[uint16]uint32:
		if v2, changed2 := fastpathTV.DecMapUint16Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]uint64:
		fastpathTV.DecMapUint16Uint64V(v, false, d)
	case *map[uint16]uint64:
		if v2, changed2 := fastpathTV.DecMapUint16Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]uintptr:
		fastpathTV.DecMapUint16UintptrV(v, false, d)
	case *map[uint16]uintptr:
		if v2, changed2 := fastpathTV.DecMapUint16UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]int:
		fastpathTV.DecMapUint16IntV(v, false, d)
	case *map[uint16]int:
		if v2, changed2 := fastpathTV.DecMapUint16IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]int8:
		fastpathTV.DecMapUint16Int8V(v, false, d)
	case *map[uint16]int8:
		if v2, changed2 := fastpathTV.DecMapUint16Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]int16:
		fastpathTV.DecMapUint16Int16V(v, false, d)
	case *map[uint16]int16:
		if v2, changed2 := fastpathTV.DecMapUint16Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]int32:
		fastpathTV.DecMapUint16Int32V(v, false, d)
	case *map[uint16]int32:
		if v2, changed2 := fastpathTV.DecMapUint16Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]int64:
		fastpathTV.DecMapUint16Int64V(v, false, d)
	case *map[uint16]int64:
		if v2, changed2 := fastpathTV.DecMapUint16Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]float32:
		fastpathTV.DecMapUint16Float32V(v, false, d)
	case *map[uint16]float32:
		if v2, changed2 := fastpathTV.DecMapUint16Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]float64:
		fastpathTV.DecMapUint16Float64V(v, false, d)
	case *map[uint16]float64:
		if v2, changed2 := fastpathTV.DecMapUint16Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint16]bool:
		fastpathTV.DecMapUint16BoolV(v, false, d)
	case *map[uint16]bool:
		if v2, changed2 := fastpathTV.DecMapUint16BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []uint32:
		fastpathTV.DecSliceUint32V(v, false, d)
	case *[]uint32:
		if v2, changed2 := fastpathTV.DecSliceUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]interface{}:
		fastpathTV.DecMapUint32IntfV(v, false, d)
	case *map[uint32]interface{}:
		if v2, changed2 := fastpathTV.DecMapUint32IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]string:
		fastpathTV.DecMapUint32StringV(v, false, d)
	case *map[uint32]string:
		if v2, changed2 := fastpathTV.DecMapUint32StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]uint:
		fastpathTV.DecMapUint32UintV(v, false, d)
	case *map[uint32]uint:
		if v2, changed2 := fastpathTV.DecMapUint32UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]uint8:
		fastpathTV.DecMapUint32Uint8V(v, false, d)
	case *map[uint32]uint8:
		if v2, changed2 := fastpathTV.DecMapUint32Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]uint16:
		fastpathTV.DecMapUint32Uint16V(v, false, d)
	case *map[uint32]uint16:
		if v2, changed2 := fastpathTV.DecMapUint32Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]uint32:
		fastpathTV.DecMapUint32Uint32V(v, false, d)
	case *map[uint32]uint32:
		if v2, changed2 := fastpathTV.DecMapUint32Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]uint64:
		fastpathTV.DecMapUint32Uint64V(v, false, d)
	case *map[uint32]uint64:
		if v2, changed2 := fastpathTV.DecMapUint32Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]uintptr:
		fastpathTV.DecMapUint32UintptrV(v, false, d)
	case *map[uint32]uintptr:
		if v2, changed2 := fastpathTV.DecMapUint32UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]int:
		fastpathTV.DecMapUint32IntV(v, false, d)
	case *map[uint32]int:
		if v2, changed2 := fastpathTV.DecMapUint32IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]int8:
		fastpathTV.DecMapUint32Int8V(v, false, d)
	case *map[uint32]int8:
		if v2, changed2 := fastpathTV.DecMapUint32Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]int16:
		fastpathTV.DecMapUint32Int16V(v, false, d)
	case *map[uint32]int16:
		if v2, changed2 := fastpathTV.DecMapUint32Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]int32:
		fastpathTV.DecMapUint32Int32V(v, false, d)
	case *map[uint32]int32:
		if v2, changed2 := fastpathTV.DecMapUint32Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]int64:
		fastpathTV.DecMapUint32Int64V(v, false, d)
	case *map[uint32]int64:
		if v2, changed2 := fastpathTV.DecMapUint32Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]float32:
		fastpathTV.DecMapUint32Float32V(v, false, d)
	case *map[uint32]float32:
		if v2, changed2 := fastpathTV.DecMapUint32Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]float64:
		fastpathTV.DecMapUint32Float64V(v, false, d)
	case *map[uint32]float64:
		if v2, changed2 := fastpathTV.DecMapUint32Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint32]bool:
		fastpathTV.DecMapUint32BoolV(v, false, d)
	case *map[uint32]bool:
		if v2, changed2 := fastpathTV.DecMapUint32BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []uint64:
		fastpathTV.DecSliceUint64V(v, false, d)
	case *[]uint64:
		if v2, changed2 := fastpathTV.DecSliceUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]interface{}:
		fastpathTV.DecMapUint64IntfV(v, false, d)
	case *map[uint64]interface{}:
		if v2, changed2 := fastpathTV.DecMapUint64IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]string:
		fastpathTV.DecMapUint64StringV(v, false, d)
	case *map[uint64]string:
		if v2, changed2 := fastpathTV.DecMapUint64StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]uint:
		fastpathTV.DecMapUint64UintV(v, false, d)
	case *map[uint64]uint:
		if v2, changed2 := fastpathTV.DecMapUint64UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]uint8:
		fastpathTV.DecMapUint64Uint8V(v, false, d)
	case *map[uint64]uint8:
		if v2, changed2 := fastpathTV.DecMapUint64Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]uint16:
		fastpathTV.DecMapUint64Uint16V(v, false, d)
	case *map[uint64]uint16:
		if v2, changed2 := fastpathTV.DecMapUint64Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]uint32:
		fastpathTV.DecMapUint64Uint32V(v, false, d)
	case *map[uint64]uint32:
		if v2, changed2 := fastpathTV.DecMapUint64Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]uint64:
		fastpathTV.DecMapUint64Uint64V(v, false, d)
	case *map[uint64]uint64:
		if v2, changed2 := fastpathTV.DecMapUint64Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]uintptr:
		fastpathTV.DecMapUint64UintptrV(v, false, d)
	case *map[uint64]uintptr:
		if v2, changed2 := fastpathTV.DecMapUint64UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]int:
		fastpathTV.DecMapUint64IntV(v, false, d)
	case *map[uint64]int:
		if v2, changed2 := fastpathTV.DecMapUint64IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]int8:
		fastpathTV.DecMapUint64Int8V(v, false, d)
	case *map[uint64]int8:
		if v2, changed2 := fastpathTV.DecMapUint64Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]int16:
		fastpathTV.DecMapUint64Int16V(v, false, d)
	case *map[uint64]int16:
		if v2, changed2 := fastpathTV.DecMapUint64Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]int32:
		fastpathTV.DecMapUint64Int32V(v, false, d)
	case *map[uint64]int32:
		if v2, changed2 := fastpathTV.DecMapUint64Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]int64:
		fastpathTV.DecMapUint64Int64V(v, false, d)
	case *map[uint64]int64:
		if v2, changed2 := fastpathTV.DecMapUint64Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]float32:
		fastpathTV.DecMapUint64Float32V(v, false, d)
	case *map[uint64]float32:
		if v2, changed2 := fastpathTV.DecMapUint64Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]float64:
		fastpathTV.DecMapUint64Float64V(v, false, d)
	case *map[uint64]float64:
		if v2, changed2 := fastpathTV.DecMapUint64Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uint64]bool:
		fastpathTV.DecMapUint64BoolV(v, false, d)
	case *map[uint64]bool:
		if v2, changed2 := fastpathTV.DecMapUint64BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []uintptr:
		fastpathTV.DecSliceUintptrV(v, false, d)
	case *[]uintptr:
		if v2, changed2 := fastpathTV.DecSliceUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]interface{}:
		fastpathTV.DecMapUintptrIntfV(v, false, d)
	case *map[uintptr]interface{}:
		if v2, changed2 := fastpathTV.DecMapUintptrIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]string:
		fastpathTV.DecMapUintptrStringV(v, false, d)
	case *map[uintptr]string:
		if v2, changed2 := fastpathTV.DecMapUintptrStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]uint:
		fastpathTV.DecMapUintptrUintV(v, false, d)
	case *map[uintptr]uint:
		if v2, changed2 := fastpathTV.DecMapUintptrUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]uint8:
		fastpathTV.DecMapUintptrUint8V(v, false, d)
	case *map[uintptr]uint8:
		if v2, changed2 := fastpathTV.DecMapUintptrUint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]uint16:
		fastpathTV.DecMapUintptrUint16V(v, false, d)
	case *map[uintptr]uint16:
		if v2, changed2 := fastpathTV.DecMapUintptrUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]uint32:
		fastpathTV.DecMapUintptrUint32V(v, false, d)
	case *map[uintptr]uint32:
		if v2, changed2 := fastpathTV.DecMapUintptrUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]uint64:
		fastpathTV.DecMapUintptrUint64V(v, false, d)
	case *map[uintptr]uint64:
		if v2, changed2 := fastpathTV.DecMapUintptrUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]uintptr:
		fastpathTV.DecMapUintptrUintptrV(v, false, d)
	case *map[uintptr]uintptr:
		if v2, changed2 := fastpathTV.DecMapUintptrUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]int:
		fastpathTV.DecMapUintptrIntV(v, false, d)
	case *map[uintptr]int:
		if v2, changed2 := fastpathTV.DecMapUintptrIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]int8:
		fastpathTV.DecMapUintptrInt8V(v, false, d)
	case *map[uintptr]int8:
		if v2, changed2 := fastpathTV.DecMapUintptrInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]int16:
		fastpathTV.DecMapUintptrInt16V(v, false, d)
	case *map[uintptr]int16:
		if v2, changed2 := fastpathTV.DecMapUintptrInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]int32:
		fastpathTV.DecMapUintptrInt32V(v, false, d)
	case *map[uintptr]int32:
		if v2, changed2 := fastpathTV.DecMapUintptrInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]int64:
		fastpathTV.DecMapUintptrInt64V(v, false, d)
	case *map[uintptr]int64:
		if v2, changed2 := fastpathTV.DecMapUintptrInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]float32:
		fastpathTV.DecMapUintptrFloat32V(v, false, d)
	case *map[uintptr]float32:
		if v2, changed2 := fastpathTV.DecMapUintptrFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]float64:
		fastpathTV.DecMapUintptrFloat64V(v, false, d)
	case *map[uintptr]float64:
		if v2, changed2 := fastpathTV.DecMapUintptrFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[uintptr]bool:
		fastpathTV.DecMapUintptrBoolV(v, false, d)
	case *map[uintptr]bool:
		if v2, changed2 := fastpathTV.DecMapUintptrBoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []int:
		fastpathTV.DecSliceIntV(v, false, d)
	case *[]int:
		if v2, changed2 := fastpathTV.DecSliceIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]interface{}:
		fastpathTV.DecMapIntIntfV(v, false, d)
	case *map[int]interface{}:
		if v2, changed2 := fastpathTV.DecMapIntIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]string:
		fastpathTV.DecMapIntStringV(v, false, d)
	case *map[int]string:
		if v2, changed2 := fastpathTV.DecMapIntStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]uint:
		fastpathTV.DecMapIntUintV(v, false, d)
	case *map[int]uint:
		if v2, changed2 := fastpathTV.DecMapIntUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]uint8:
		fastpathTV.DecMapIntUint8V(v, false, d)
	case *map[int]uint8:
		if v2, changed2 := fastpathTV.DecMapIntUint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]uint16:
		fastpathTV.DecMapIntUint16V(v, false, d)
	case *map[int]uint16:
		if v2, changed2 := fastpathTV.DecMapIntUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]uint32:
		fastpathTV.DecMapIntUint32V(v, false, d)
	case *map[int]uint32:
		if v2, changed2 := fastpathTV.DecMapIntUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]uint64:
		fastpathTV.DecMapIntUint64V(v, false, d)
	case *map[int]uint64:
		if v2, changed2 := fastpathTV.DecMapIntUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]uintptr:
		fastpathTV.DecMapIntUintptrV(v, false, d)
	case *map[int]uintptr:
		if v2, changed2 := fastpathTV.DecMapIntUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]int:
		fastpathTV.DecMapIntIntV(v, false, d)
	case *map[int]int:
		if v2, changed2 := fastpathTV.DecMapIntIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]int8:
		fastpathTV.DecMapIntInt8V(v, false, d)
	case *map[int]int8:
		if v2, changed2 := fastpathTV.DecMapIntInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]int16:
		fastpathTV.DecMapIntInt16V(v, false, d)
	case *map[int]int16:
		if v2, changed2 := fastpathTV.DecMapIntInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]int32:
		fastpathTV.DecMapIntInt32V(v, false, d)
	case *map[int]int32:
		if v2, changed2 := fastpathTV.DecMapIntInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]int64:
		fastpathTV.DecMapIntInt64V(v, false, d)
	case *map[int]int64:
		if v2, changed2 := fastpathTV.DecMapIntInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]float32:
		fastpathTV.DecMapIntFloat32V(v, false, d)
	case *map[int]float32:
		if v2, changed2 := fastpathTV.DecMapIntFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]float64:
		fastpathTV.DecMapIntFloat64V(v, false, d)
	case *map[int]float64:
		if v2, changed2 := fastpathTV.DecMapIntFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int]bool:
		fastpathTV.DecMapIntBoolV(v, false, d)
	case *map[int]bool:
		if v2, changed2 := fastpathTV.DecMapIntBoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []int8:
		fastpathTV.DecSliceInt8V(v, false, d)
	case *[]int8:
		if v2, changed2 := fastpathTV.DecSliceInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]interface{}:
		fastpathTV.DecMapInt8IntfV(v, false, d)
	case *map[int8]interface{}:
		if v2, changed2 := fastpathTV.DecMapInt8IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]string:
		fastpathTV.DecMapInt8StringV(v, false, d)
	case *map[int8]string:
		if v2, changed2 := fastpathTV.DecMapInt8StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]uint:
		fastpathTV.DecMapInt8UintV(v, false, d)
	case *map[int8]uint:
		if v2, changed2 := fastpathTV.DecMapInt8UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]uint8:
		fastpathTV.DecMapInt8Uint8V(v, false, d)
	case *map[int8]uint8:
		if v2, changed2 := fastpathTV.DecMapInt8Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]uint16:
		fastpathTV.DecMapInt8Uint16V(v, false, d)
	case *map[int8]uint16:
		if v2, changed2 := fastpathTV.DecMapInt8Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]uint32:
		fastpathTV.DecMapInt8Uint32V(v, false, d)
	case *map[int8]uint32:
		if v2, changed2 := fastpathTV.DecMapInt8Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]uint64:
		fastpathTV.DecMapInt8Uint64V(v, false, d)
	case *map[int8]uint64:
		if v2, changed2 := fastpathTV.DecMapInt8Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]uintptr:
		fastpathTV.DecMapInt8UintptrV(v, false, d)
	case *map[int8]uintptr:
		if v2, changed2 := fastpathTV.DecMapInt8UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]int:
		fastpathTV.DecMapInt8IntV(v, false, d)
	case *map[int8]int:
		if v2, changed2 := fastpathTV.DecMapInt8IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]int8:
		fastpathTV.DecMapInt8Int8V(v, false, d)
	case *map[int8]int8:
		if v2, changed2 := fastpathTV.DecMapInt8Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]int16:
		fastpathTV.DecMapInt8Int16V(v, false, d)
	case *map[int8]int16:
		if v2, changed2 := fastpathTV.DecMapInt8Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]int32:
		fastpathTV.DecMapInt8Int32V(v, false, d)
	case *map[int8]int32:
		if v2, changed2 := fastpathTV.DecMapInt8Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]int64:
		fastpathTV.DecMapInt8Int64V(v, false, d)
	case *map[int8]int64:
		if v2, changed2 := fastpathTV.DecMapInt8Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]float32:
		fastpathTV.DecMapInt8Float32V(v, false, d)
	case *map[int8]float32:
		if v2, changed2 := fastpathTV.DecMapInt8Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]float64:
		fastpathTV.DecMapInt8Float64V(v, false, d)
	case *map[int8]float64:
		if v2, changed2 := fastpathTV.DecMapInt8Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int8]bool:
		fastpathTV.DecMapInt8BoolV(v, false, d)
	case *map[int8]bool:
		if v2, changed2 := fastpathTV.DecMapInt8BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []int16:
		fastpathTV.DecSliceInt16V(v, false, d)
	case *[]int16:
		if v2, changed2 := fastpathTV.DecSliceInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]interface{}:
		fastpathTV.DecMapInt16IntfV(v, false, d)
	case *map[int16]interface{}:
		if v2, changed2 := fastpathTV.DecMapInt16IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]string:
		fastpathTV.DecMapInt16StringV(v, false, d)
	case *map[int16]string:
		if v2, changed2 := fastpathTV.DecMapInt16StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]uint:
		fastpathTV.DecMapInt16UintV(v, false, d)
	case *map[int16]uint:
		if v2, changed2 := fastpathTV.DecMapInt16UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]uint8:
		fastpathTV.DecMapInt16Uint8V(v, false, d)
	case *map[int16]uint8:
		if v2, changed2 := fastpathTV.DecMapInt16Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]uint16:
		fastpathTV.DecMapInt16Uint16V(v, false, d)
	case *map[int16]uint16:
		if v2, changed2 := fastpathTV.DecMapInt16Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]uint32:
		fastpathTV.DecMapInt16Uint32V(v, false, d)
	case *map[int16]uint32:
		if v2, changed2 := fastpathTV.DecMapInt16Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]uint64:
		fastpathTV.DecMapInt16Uint64V(v, false, d)
	case *map[int16]uint64:
		if v2, changed2 := fastpathTV.DecMapInt16Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]uintptr:
		fastpathTV.DecMapInt16UintptrV(v, false, d)
	case *map[int16]uintptr:
		if v2, changed2 := fastpathTV.DecMapInt16UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]int:
		fastpathTV.DecMapInt16IntV(v, false, d)
	case *map[int16]int:
		if v2, changed2 := fastpathTV.DecMapInt16IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]int8:
		fastpathTV.DecMapInt16Int8V(v, false, d)
	case *map[int16]int8:
		if v2, changed2 := fastpathTV.DecMapInt16Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]int16:
		fastpathTV.DecMapInt16Int16V(v, false, d)
	case *map[int16]int16:
		if v2, changed2 := fastpathTV.DecMapInt16Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]int32:
		fastpathTV.DecMapInt16Int32V(v, false, d)
	case *map[int16]int32:
		if v2, changed2 := fastpathTV.DecMapInt16Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]int64:
		fastpathTV.DecMapInt16Int64V(v, false, d)
	case *map[int16]int64:
		if v2, changed2 := fastpathTV.DecMapInt16Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]float32:
		fastpathTV.DecMapInt16Float32V(v, false, d)
	case *map[int16]float32:
		if v2, changed2 := fastpathTV.DecMapInt16Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]float64:
		fastpathTV.DecMapInt16Float64V(v, false, d)
	case *map[int16]float64:
		if v2, changed2 := fastpathTV.DecMapInt16Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int16]bool:
		fastpathTV.DecMapInt16BoolV(v, false, d)
	case *map[int16]bool:
		if v2, changed2 := fastpathTV.DecMapInt16BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []int32:
		fastpathTV.DecSliceInt32V(v, false, d)
	case *[]int32:
		if v2, changed2 := fastpathTV.DecSliceInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]interface{}:
		fastpathTV.DecMapInt32IntfV(v, false, d)
	case *map[int32]interface{}:
		if v2, changed2 := fastpathTV.DecMapInt32IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]string:
		fastpathTV.DecMapInt32StringV(v, false, d)
	case *map[int32]string:
		if v2, changed2 := fastpathTV.DecMapInt32StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]uint:
		fastpathTV.DecMapInt32UintV(v, false, d)
	case *map[int32]uint:
		if v2, changed2 := fastpathTV.DecMapInt32UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]uint8:
		fastpathTV.DecMapInt32Uint8V(v, false, d)
	case *map[int32]uint8:
		if v2, changed2 := fastpathTV.DecMapInt32Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]uint16:
		fastpathTV.DecMapInt32Uint16V(v, false, d)
	case *map[int32]uint16:
		if v2, changed2 := fastpathTV.DecMapInt32Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]uint32:
		fastpathTV.DecMapInt32Uint32V(v, false, d)
	case *map[int32]uint32:
		if v2, changed2 := fastpathTV.DecMapInt32Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]uint64:
		fastpathTV.DecMapInt32Uint64V(v, false, d)
	case *map[int32]uint64:
		if v2, changed2 := fastpathTV.DecMapInt32Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]uintptr:
		fastpathTV.DecMapInt32UintptrV(v, false, d)
	case *map[int32]uintptr:
		if v2, changed2 := fastpathTV.DecMapInt32UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]int:
		fastpathTV.DecMapInt32IntV(v, false, d)
	case *map[int32]int:
		if v2, changed2 := fastpathTV.DecMapInt32IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]int8:
		fastpathTV.DecMapInt32Int8V(v, false, d)
	case *map[int32]int8:
		if v2, changed2 := fastpathTV.DecMapInt32Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]int16:
		fastpathTV.DecMapInt32Int16V(v, false, d)
	case *map[int32]int16:
		if v2, changed2 := fastpathTV.DecMapInt32Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]int32:
		fastpathTV.DecMapInt32Int32V(v, false, d)
	case *map[int32]int32:
		if v2, changed2 := fastpathTV.DecMapInt32Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]int64:
		fastpathTV.DecMapInt32Int64V(v, false, d)
	case *map[int32]int64:
		if v2, changed2 := fastpathTV.DecMapInt32Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]float32:
		fastpathTV.DecMapInt32Float32V(v, false, d)
	case *map[int32]float32:
		if v2, changed2 := fastpathTV.DecMapInt32Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]float64:
		fastpathTV.DecMapInt32Float64V(v, false, d)
	case *map[int32]float64:
		if v2, changed2 := fastpathTV.DecMapInt32Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int32]bool:
		fastpathTV.DecMapInt32BoolV(v, false, d)
	case *map[int32]bool:
		if v2, changed2 := fastpathTV.DecMapInt32BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []int64:
		fastpathTV.DecSliceInt64V(v, false, d)
	case *[]int64:
		if v2, changed2 := fastpathTV.DecSliceInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]interface{}:
		fastpathTV.DecMapInt64IntfV(v, false, d)
	case *map[int64]interface{}:
		if v2, changed2 := fastpathTV.DecMapInt64IntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]string:
		fastpathTV.DecMapInt64StringV(v, false, d)
	case *map[int64]string:
		if v2, changed2 := fastpathTV.DecMapInt64StringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]uint:
		fastpathTV.DecMapInt64UintV(v, false, d)
	case *map[int64]uint:
		if v2, changed2 := fastpathTV.DecMapInt64UintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]uint8:
		fastpathTV.DecMapInt64Uint8V(v, false, d)
	case *map[int64]uint8:
		if v2, changed2 := fastpathTV.DecMapInt64Uint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]uint16:
		fastpathTV.DecMapInt64Uint16V(v, false, d)
	case *map[int64]uint16:
		if v2, changed2 := fastpathTV.DecMapInt64Uint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]uint32:
		fastpathTV.DecMapInt64Uint32V(v, false, d)
	case *map[int64]uint32:
		if v2, changed2 := fastpathTV.DecMapInt64Uint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]uint64:
		fastpathTV.DecMapInt64Uint64V(v, false, d)
	case *map[int64]uint64:
		if v2, changed2 := fastpathTV.DecMapInt64Uint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]uintptr:
		fastpathTV.DecMapInt64UintptrV(v, false, d)
	case *map[int64]uintptr:
		if v2, changed2 := fastpathTV.DecMapInt64UintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]int:
		fastpathTV.DecMapInt64IntV(v, false, d)
	case *map[int64]int:
		if v2, changed2 := fastpathTV.DecMapInt64IntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]int8:
		fastpathTV.DecMapInt64Int8V(v, false, d)
	case *map[int64]int8:
		if v2, changed2 := fastpathTV.DecMapInt64Int8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]int16:
		fastpathTV.DecMapInt64Int16V(v, false, d)
	case *map[int64]int16:
		if v2, changed2 := fastpathTV.DecMapInt64Int16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]int32:
		fastpathTV.DecMapInt64Int32V(v, false, d)
	case *map[int64]int32:
		if v2, changed2 := fastpathTV.DecMapInt64Int32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]int64:
		fastpathTV.DecMapInt64Int64V(v, false, d)
	case *map[int64]int64:
		if v2, changed2 := fastpathTV.DecMapInt64Int64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]float32:
		fastpathTV.DecMapInt64Float32V(v, false, d)
	case *map[int64]float32:
		if v2, changed2 := fastpathTV.DecMapInt64Float32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]float64:
		fastpathTV.DecMapInt64Float64V(v, false, d)
	case *map[int64]float64:
		if v2, changed2 := fastpathTV.DecMapInt64Float64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[int64]bool:
		fastpathTV.DecMapInt64BoolV(v, false, d)
	case *map[int64]bool:
		if v2, changed2 := fastpathTV.DecMapInt64BoolV(*v, true, d); changed2 {
			*v = v2
		}

	case []bool:
		fastpathTV.DecSliceBoolV(v, false, d)
	case *[]bool:
		if v2, changed2 := fastpathTV.DecSliceBoolV(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]interface{}:
		fastpathTV.DecMapBoolIntfV(v, false, d)
	case *map[bool]interface{}:
		if v2, changed2 := fastpathTV.DecMapBoolIntfV(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]string:
		fastpathTV.DecMapBoolStringV(v, false, d)
	case *map[bool]string:
		if v2, changed2 := fastpathTV.DecMapBoolStringV(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]uint:
		fastpathTV.DecMapBoolUintV(v, false, d)
	case *map[bool]uint:
		if v2, changed2 := fastpathTV.DecMapBoolUintV(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]uint8:
		fastpathTV.DecMapBoolUint8V(v, false, d)
	case *map[bool]uint8:
		if v2, changed2 := fastpathTV.DecMapBoolUint8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]uint16:
		fastpathTV.DecMapBoolUint16V(v, false, d)
	case *map[bool]uint16:
		if v2, changed2 := fastpathTV.DecMapBoolUint16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]uint32:
		fastpathTV.DecMapBoolUint32V(v, false, d)
	case *map[bool]uint32:
		if v2, changed2 := fastpathTV.DecMapBoolUint32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]uint64:
		fastpathTV.DecMapBoolUint64V(v, false, d)
	case *map[bool]uint64:
		if v2, changed2 := fastpathTV.DecMapBoolUint64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]uintptr:
		fastpathTV.DecMapBoolUintptrV(v, false, d)
	case *map[bool]uintptr:
		if v2, changed2 := fastpathTV.DecMapBoolUintptrV(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]int:
		fastpathTV.DecMapBoolIntV(v, false, d)
	case *map[bool]int:
		if v2, changed2 := fastpathTV.DecMapBoolIntV(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]int8:
		fastpathTV.DecMapBoolInt8V(v, false, d)
	case *map[bool]int8:
		if v2, changed2 := fastpathTV.DecMapBoolInt8V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]int16:
		fastpathTV.DecMapBoolInt16V(v, false, d)
	case *map[bool]int16:
		if v2, changed2 := fastpathTV.DecMapBoolInt16V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]int32:
		fastpathTV.DecMapBoolInt32V(v, false, d)
	case *map[bool]int32:
		if v2, changed2 := fastpathTV.DecMapBoolInt32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]int64:
		fastpathTV.DecMapBoolInt64V(v, false, d)
	case *map[bool]int64:
		if v2, changed2 := fastpathTV.DecMapBoolInt64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]float32:
		fastpathTV.DecMapBoolFloat32V(v, false, d)
	case *map[bool]float32:
		if v2, changed2 := fastpathTV.DecMapBoolFloat32V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]float64:
		fastpathTV.DecMapBoolFloat64V(v, false, d)
	case *map[bool]float64:
		if v2, changed2 := fastpathTV.DecMapBoolFloat64V(*v, true, d); changed2 {
			*v = v2
		}

	case map[bool]bool:
		fastpathTV.DecMapBoolBoolV(v, false, d)
	case *map[bool]bool:
		if v2, changed2 := fastpathTV.DecMapBoolBoolV(*v, true, d); changed2 {
			*v = v2
		}

	default:
		_ = v // TODO: workaround https://github.com/golang/go/issues/12927 (remove after go 1.6 release)
		return false
	}
	return true
}

func fastpathDecodeSetZeroTypeSwitch(iv interface{}, d *Decoder) bool {
	switch v := iv.(type) {

	case *[]interface{}:
		*v = nil

	case *map[interface{}]interface{}:
		*v = nil

	case *map[interface{}]string:
		*v = nil

	case *map[interface{}]uint:
		*v = nil

	case *map[interface{}]uint8:
		*v = nil

	case *map[interface{}]uint16:
		*v = nil

	case *map[interface{}]uint32:
		*v = nil

	case *map[interface{}]uint64:
		*v = nil

	case *map[interface{}]uintptr:
		*v = nil

	case *map[interface{}]int:
		*v = nil

	case *map[interface{}]int8:
		*v = nil

	case *map[interface{}]int16:
		*v = nil

	case *map[interface{}]int32:
		*v = nil

	case *map[interface{}]int64:
		*v = nil

	case *map[interface{}]float32:
		*v = nil

	case *map[interface{}]float64:
		*v = nil

	case *map[interface{}]bool:
		*v = nil

	case *[]string:
		*v = nil

	case *map[string]interface{}:
		*v = nil

	case *map[string]string:
		*v = nil

	case *map[string]uint:
		*v = nil

	case *map[string]uint8:
		*v = nil

	case *map[string]uint16:
		*v = nil

	case *map[string]uint32:
		*v = nil

	case *map[string]uint64:
		*v = nil

	case *map[string]uintptr:
		*v = nil

	case *map[string]int:
		*v = nil

	case *map[string]int8:
		*v = nil

	case *map[string]int16:
		*v = nil

	case *map[string]int32:
		*v = nil

	case *map[string]int64:
		*v = nil

	case *map[string]float32:
		*v = nil

	case *map[string]float64:
		*v = nil

	case *map[string]bool:
		*v = nil

	case *[]float32:
		*v = nil

	case *map[float32]interface{}:
		*v = nil

	case *map[float32]string:
		*v = nil

	case *map[float32]uint:
		*v = nil

	case *map[float32]uint8:
		*v = nil

	case *map[float32]uint16:
		*v = nil

	case *map[float32]uint32:
		*v = nil

	case *map[float32]uint64:
		*v = nil

	case *map[float32]uintptr:
		*v = nil

	case *map[float32]int:
		*v = nil

	case *map[float32]int8:
		*v = nil

	case *map[float32]int16:
		*v = nil

	case *map[float32]int32:
		*v = nil

	case *map[float32]int64:
		*v = nil

	case *map[float32]float32:
		*v = nil

	case *map[float32]float64:
		*v = nil

	case *map[float32]bool:
		*v = nil

	case *[]float64:
		*v = nil

	case *map[float64]interface{}:
		*v = nil

	case *map[float64]string:
		*v = nil

	case *map[float64]uint:
		*v = nil

	case *map[float64]uint8:
		*v = nil

	case *map[float64]uint16:
		*v = nil

	case *map[float64]uint32:
		*v = nil

	case *map[float64]uint64:
		*v = nil

	case *map[float64]uintptr:
		*v = nil

	case *map[float64]int:
		*v = nil

	case *map[float64]int8:
		*v = nil

	case *map[float64]int16:
		*v = nil

	case *map[float64]int32:
		*v = nil

	case *map[float64]int64:
		*v = nil

	case *map[float64]float32:
		*v = nil

	case *map[float64]float64:
		*v = nil

	case *map[float64]bool:
		*v = nil

	case *[]uint:
		*v = nil

	case *map[uint]interface{}:
		*v = nil

	case *map[uint]string:
		*v = nil

	case *map[uint]uint:
		*v = nil

	case *map[uint]uint8:
		*v = nil

	case *map[uint]uint16:
		*v = nil

	case *map[uint]uint32:
		*v = nil

	case *map[uint]uint64:
		*v = nil

	case *map[uint]uintptr:
		*v = nil

	case *map[uint]int:
		*v = nil

	case *map[uint]int8:
		*v = nil

	case *map[uint]int16:
		*v = nil

	case *map[uint]int32:
		*v = nil

	case *map[uint]int64:
		*v = nil

	case *map[uint]float32:
		*v = nil

	case *map[uint]float64:
		*v = nil

	case *map[uint]bool:
		*v = nil

	case *map[uint8]interface{}:
		*v = nil

	case *map[uint8]string:
		*v = nil

	case *map[uint8]uint:
		*v = nil

	case *map[uint8]uint8:
		*v = nil

	case *map[uint8]uint16:
		*v = nil

	case *map[uint8]uint32:
		*v = nil

	case *map[uint8]uint64:
		*v = nil

	case *map[uint8]uintptr:
		*v = nil

	case *map[uint8]int:
		*v = nil

	case *map[uint8]int8:
		*v = nil

	case *map[uint8]int16:
		*v = nil

	case *map[uint8]int32:
		*v = nil

	case *map[uint8]int64:
		*v = nil

	case *map[uint8]float32:
		*v = nil

	case *map[uint8]float64:
		*v = nil

	case *map[uint8]bool:
		*v = nil

	case *[]uint16:
		*v = nil

	case *map[uint16]interface{}:
		*v = nil

	case *map[uint16]string:
		*v = nil

	case *map[uint16]uint:
		*v = nil

	case *map[uint16]uint8:
		*v = nil

	case *map[uint16]uint16:
		*v = nil

	case *map[uint16]uint32:
		*v = nil

	case *map[uint16]uint64:
		*v = nil

	case *map[uint16]uintptr:
		*v = nil

	case *map[uint16]int:
		*v = nil

	case *map[uint16]int8:
		*v = nil

	case *map[uint16]int16:
		*v = nil

	case *map[uint16]int32:
		*v = nil

	case *map[uint16]int64:
		*v = nil

	case *map[uint16]float32:
		*v = nil

	case *map[uint16]float64:
		*v = nil

	case *map[uint16]bool:
		*v = nil

	case *[]uint32:
		*v = nil

	case *map[uint32]interface{}:
		*v = nil

	case *map[uint32]string:
		*v = nil

	case *map[uint32]uint:
		*v = nil

	case *map[uint32]uint8:
		*v = nil

	case *map[uint32]uint16:
		*v = nil

	case *map[uint32]uint32:
		*v = nil

	case *map[uint32]uint64:
		*v = nil

	case *map[uint32]uintptr:
		*v = nil

	case *map[uint32]int:
		*v = nil

	case *map[uint32]int8:
		*v = nil

	case *map[uint32]int16:
		*v = nil

	case *map[uint32]int32:
		*v = nil

	case *map[uint32]int64:
		*v = nil

	case *map[uint32]float32:
		*v = nil

	case *map[uint32]float64:
		*v = nil

	case *map[uint32]bool:
		*v = nil

	case *[]uint64:
		*v = nil

	case *map[uint64]interface{}:
		*v = nil

	case *map[uint64]string:
		*v = nil

	case *map[uint64]uint:
		*v = nil

	case *map[uint64]uint8:
		*v = nil

	case *map[uint64]uint16:
		*v = nil

	case *map[uint64]uint32:
		*v = nil

	case *map[uint64]uint64:
		*v = nil

	case *map[uint64]uintptr:
		*v = nil

	case *map[uint64]int:
		*v = nil

	case *map[uint64]int8:
		*v = nil

	case *map[uint64]int16:
		*v = nil

	case *map[uint64]int32:
		*v = nil

	case *map[uint64]int64:
		*v = nil

	case *map[uint64]float32:
		*v = nil

	case *map[uint64]float64:
		*v = nil

	case *map[uint64]bool:
		*v = nil

	case *[]uintptr:
		*v = nil

	case *map[uintptr]interface{}:
		*v = nil

	case *map[uintptr]string:
		*v = nil

	case *map[uintptr]uint:
		*v = nil

	case *map[uintptr]uint8:
		*v = nil

	case *map[uintptr]uint16:
		*v = nil

	case *map[uintptr]uint32:
		*v = nil

	case *map[uintptr]uint64:
		*v = nil

	case *map[uintptr]uintptr:
		*v = nil

	case *map[uintptr]int:
		*v = nil

	case *map[uintptr]int8:
		*v = nil

	case *map[uintptr]int16:
		*v = nil

	case *map[uintptr]int32:
		*v = nil

	case *map[uintptr]int64:
		*v = nil

	case *map[uintptr]float32:
		*v = nil

	case *map[uintptr]float64:
		*v = nil

	case *map[uintptr]bool:
		*v = nil

	case *[]int:
		*v = nil

	case *map[int]interface{}:
		*v = nil

	case *map[int]string:
		*v = nil

	case *map[int]uint:
		*v = nil

	case *map[int]uint8:
		*v = nil

	case *map[int]uint16:
		*v = nil

	case *map[int]uint32:
		*v = nil

	case *map[int]uint64:
		*v = nil

	case *map[int]uintptr:
		*v = nil

	case *map[int]int:
		*v = nil

	case *map[int]int8:
		*v = nil

	case *map[int]int16:
		*v = nil

	case *map[int]int32:
		*v = nil

	case *map[int]int64:
		*v = nil

	case *map[int]float32:
		*v = nil

	case *map[int]float64:
		*v = nil

	case *map[int]bool:
		*v = nil

	case *[]int8:
		*v = nil

	case *map[int8]interface{}:
		*v = nil

	case *map[int8]string:
		*v = nil

	case *map[int8]uint:
		*v = nil

	case *map[int8]uint8:
		*v = nil

	case *map[int8]uint16:
		*v = nil

	case *map[int8]uint32:
		*v = nil

	case *map[int8]uint64:
		*v = nil

	case *map[int8]uintptr:
		*v = nil

	case *map[int8]int:
		*v = nil

	case *map[int8]int8:
		*v = nil

	case *map[int8]int16:
		*v = nil

	case *map[int8]int32:
		*v = nil

	case *map[int8]int64:
		*v = nil

	case *map[int8]float32:
		*v = nil

	case *map[int8]float64:
		*v = nil

	case *map[int8]bool:
		*v = nil

	case *[]int16:
		*v = nil

	case *map[int16]interface{}:
		*v = nil

	case *map[int16]string:
		*v = nil

	case *map[int16]uint:
		*v = nil

	case *map[int16]uint8:
		*v = nil

	case *map[int16]uint16:
		*v = nil

	case *map[int16]uint32:
		*v = nil

	case *map[int16]uint64:
		*v = nil

	case *map[int16]uintptr:
		*v = nil

	case *map[int16]int:
		*v = nil

	case *map[int16]int8:
		*v = nil

	case *map[int16]int16:
		*v = nil

	case *map[int16]int32:
		*v = nil

	case *map[int16]int64:
		*v = nil

	case *map[int16]float32:
		*v = nil

	case *map[int16]float64:
		*v = nil

	case *map[int16]bool:
		*v = nil

	case *[]int32:
		*v = nil

	case *map[int32]interface{}:
		*v = nil

	case *map[int32]string:
		*v = nil

	case *map[int32]uint:
		*v = nil

	case *map[int32]uint8:
		*v = nil

	case *map[int32]uint16:
		*v = nil

	case *map[int32]uint32:
		*v = nil

	case *map[int32]uint64:
		*v = nil

	case *map[int32]uintptr:
		*v = nil

	case *map[int32]int:
		*v = nil

	case *map[int32]int8:
		*v = nil

	case *map[int32]int16:
		*v = nil

	case *map[int32]int32:
		*v = nil

	case *map[int32]int64:
		*v = nil

	case *map[int32]float32:
		*v = nil

	case *map[int32]float64:
		*v = nil

	case *map[int32]bool:
		*v = nil

	case *[]int64:
		*v = nil

	case *map[int64]interface{}:
		*v = nil

	case *map[int64]string:
		*v = nil

	case *map[int64]uint:
		*v = nil

	case *map[int64]uint8:
		*v = nil

	case *map[int64]uint16:
		*v = nil

	case *map[int64]uint32:
		*v = nil

	case *map[int64]uint64:
		*v = nil

	case *map[int64]uintptr:
		*v = nil

	case *map[int64]int:
		*v = nil

	case *map[int64]int8:
		*v = nil

	case *map[int64]int16:
		*v = nil

	case *map[int64]int32:
		*v = nil

	case *map[int64]int64:
		*v = nil

	case *map[int64]float32:
		*v = nil

	case *map[int64]float64:
		*v = nil

	case *map[int64]bool:
		*v = nil

	case *[]bool:
		*v = nil

	case *map[bool]interface{}:
		*v = nil

	case *map[bool]string:
		*v = nil

	case *map[bool]uint:
		*v = nil

	case *map[bool]uint8:
		*v = nil

	case *map[bool]uint16:
		*v = nil

	case *map[bool]uint32:
		*v = nil

	case *map[bool]uint64:
		*v = nil

	case *map[bool]uintptr:
		*v = nil

	case *map[bool]int:
		*v = nil

	case *map[bool]int8:
		*v = nil

	case *map[bool]int16:
		*v = nil

	case *map[bool]int32:
		*v = nil

	case *map[bool]int64:
		*v = nil

	case *map[bool]float32:
		*v = nil

	case *map[bool]float64:
		*v = nil

	case *map[bool]bool:
		*v = nil

	default:
		_ = v // TODO: workaround https://github.com/golang/go/issues/12927 (remove after go 1.6 release)
		return false
	}
	return true
}

// -- -- fast path functions

func (d *Decoder) fastpathDecSliceIntfR(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]interface{})
		if v, changed := fastpathTV.DecSliceIntfV(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceIntfV(rv2i(rv).([]interface{}), !array, d)
	}
}

func (f fastpathT) DecSliceIntfX(vp *[]interface{}, d *Decoder) {
	if v, changed := f.DecSliceIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceIntfV(v []interface{}, canChange bool, d *Decoder) (_ []interface{}, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []interface{}{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 16)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]interface{}, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 16)
			} else {
				xlen = 8
			}
			v = make([]interface{}, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, nil)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			d.decode(&v[j])
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]interface{}, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceStringR(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]string)
		if v, changed := fastpathTV.DecSliceStringV(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceStringV(rv2i(rv).([]string), !array, d)
	}
}

func (f fastpathT) DecSliceStringX(vp *[]string, d *Decoder) {
	if v, changed := f.DecSliceStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceStringV(v []string, canChange bool, d *Decoder) (_ []string, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []string{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 16)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]string, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 16)
			} else {
				xlen = 8
			}
			v = make([]string, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, "")
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = dd.DecodeString()
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]string, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceFloat32R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]float32)
		if v, changed := fastpathTV.DecSliceFloat32V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceFloat32V(rv2i(rv).([]float32), !array, d)
	}
}

func (f fastpathT) DecSliceFloat32X(vp *[]float32, d *Decoder) {
	if v, changed := f.DecSliceFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceFloat32V(v []float32, canChange bool, d *Decoder) (_ []float32, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []float32{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 4)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]float32, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 4)
			} else {
				xlen = 8
			}
			v = make([]float32, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = float32(dd.DecodeFloat(true))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]float32, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceFloat64R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]float64)
		if v, changed := fastpathTV.DecSliceFloat64V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceFloat64V(rv2i(rv).([]float64), !array, d)
	}
}

func (f fastpathT) DecSliceFloat64X(vp *[]float64, d *Decoder) {
	if v, changed := f.DecSliceFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceFloat64V(v []float64, canChange bool, d *Decoder) (_ []float64, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []float64{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]float64, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			} else {
				xlen = 8
			}
			v = make([]float64, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = dd.DecodeFloat(false)
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]float64, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceUintR(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]uint)
		if v, changed := fastpathTV.DecSliceUintV(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceUintV(rv2i(rv).([]uint), !array, d)
	}
}

func (f fastpathT) DecSliceUintX(vp *[]uint, d *Decoder) {
	if v, changed := f.DecSliceUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUintV(v []uint, canChange bool, d *Decoder) (_ []uint, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []uint{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]uint, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			} else {
				xlen = 8
			}
			v = make([]uint, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = uint(dd.DecodeUint(uintBitsize))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]uint, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceUint16R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]uint16)
		if v, changed := fastpathTV.DecSliceUint16V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceUint16V(rv2i(rv).([]uint16), !array, d)
	}
}

func (f fastpathT) DecSliceUint16X(vp *[]uint16, d *Decoder) {
	if v, changed := f.DecSliceUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUint16V(v []uint16, canChange bool, d *Decoder) (_ []uint16, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []uint16{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 2)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]uint16, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 2)
			} else {
				xlen = 8
			}
			v = make([]uint16, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = uint16(dd.DecodeUint(16))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]uint16, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceUint32R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]uint32)
		if v, changed := fastpathTV.DecSliceUint32V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceUint32V(rv2i(rv).([]uint32), !array, d)
	}
}

func (f fastpathT) DecSliceUint32X(vp *[]uint32, d *Decoder) {
	if v, changed := f.DecSliceUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUint32V(v []uint32, canChange bool, d *Decoder) (_ []uint32, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []uint32{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 4)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]uint32, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 4)
			} else {
				xlen = 8
			}
			v = make([]uint32, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = uint32(dd.DecodeUint(32))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]uint32, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceUint64R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]uint64)
		if v, changed := fastpathTV.DecSliceUint64V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceUint64V(rv2i(rv).([]uint64), !array, d)
	}
}

func (f fastpathT) DecSliceUint64X(vp *[]uint64, d *Decoder) {
	if v, changed := f.DecSliceUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUint64V(v []uint64, canChange bool, d *Decoder) (_ []uint64, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []uint64{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]uint64, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			} else {
				xlen = 8
			}
			v = make([]uint64, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = dd.DecodeUint(64)
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]uint64, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceUintptrR(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]uintptr)
		if v, changed := fastpathTV.DecSliceUintptrV(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceUintptrV(rv2i(rv).([]uintptr), !array, d)
	}
}

func (f fastpathT) DecSliceUintptrX(vp *[]uintptr, d *Decoder) {
	if v, changed := f.DecSliceUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceUintptrV(v []uintptr, canChange bool, d *Decoder) (_ []uintptr, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []uintptr{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]uintptr, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			} else {
				xlen = 8
			}
			v = make([]uintptr, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = uintptr(dd.DecodeUint(uintBitsize))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]uintptr, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceIntR(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]int)
		if v, changed := fastpathTV.DecSliceIntV(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceIntV(rv2i(rv).([]int), !array, d)
	}
}

func (f fastpathT) DecSliceIntX(vp *[]int, d *Decoder) {
	if v, changed := f.DecSliceIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceIntV(v []int, canChange bool, d *Decoder) (_ []int, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []int{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]int, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			} else {
				xlen = 8
			}
			v = make([]int, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = int(dd.DecodeInt(intBitsize))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]int, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceInt8R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]int8)
		if v, changed := fastpathTV.DecSliceInt8V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceInt8V(rv2i(rv).([]int8), !array, d)
	}
}

func (f fastpathT) DecSliceInt8X(vp *[]int8, d *Decoder) {
	if v, changed := f.DecSliceInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt8V(v []int8, canChange bool, d *Decoder) (_ []int8, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []int8{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 1)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]int8, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 1)
			} else {
				xlen = 8
			}
			v = make([]int8, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = int8(dd.DecodeInt(8))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]int8, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceInt16R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]int16)
		if v, changed := fastpathTV.DecSliceInt16V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceInt16V(rv2i(rv).([]int16), !array, d)
	}
}

func (f fastpathT) DecSliceInt16X(vp *[]int16, d *Decoder) {
	if v, changed := f.DecSliceInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt16V(v []int16, canChange bool, d *Decoder) (_ []int16, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []int16{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 2)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]int16, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 2)
			} else {
				xlen = 8
			}
			v = make([]int16, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = int16(dd.DecodeInt(16))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]int16, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceInt32R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]int32)
		if v, changed := fastpathTV.DecSliceInt32V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceInt32V(rv2i(rv).([]int32), !array, d)
	}
}

func (f fastpathT) DecSliceInt32X(vp *[]int32, d *Decoder) {
	if v, changed := f.DecSliceInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt32V(v []int32, canChange bool, d *Decoder) (_ []int32, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []int32{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 4)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]int32, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 4)
			} else {
				xlen = 8
			}
			v = make([]int32, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = int32(dd.DecodeInt(32))
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]int32, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceInt64R(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]int64)
		if v, changed := fastpathTV.DecSliceInt64V(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceInt64V(rv2i(rv).([]int64), !array, d)
	}
}

func (f fastpathT) DecSliceInt64X(vp *[]int64, d *Decoder) {
	if v, changed := f.DecSliceInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceInt64V(v []int64, canChange bool, d *Decoder) (_ []int64, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []int64{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]int64, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 8)
			} else {
				xlen = 8
			}
			v = make([]int64, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, 0)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = dd.DecodeInt(64)
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]int64, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecSliceBoolR(f *codecFnInfo, rv reflect.Value) {
	if array := f.seq == seqTypeArray; !array && rv.Kind() == reflect.Ptr {
		var vp = rv2i(rv).(*[]bool)
		if v, changed := fastpathTV.DecSliceBoolV(*vp, !array, d); changed {
			*vp = v
		}
	} else {
		fastpathTV.DecSliceBoolV(rv2i(rv).([]bool), !array, d)
	}
}

func (f fastpathT) DecSliceBoolX(vp *[]bool, d *Decoder) {
	if v, changed := f.DecSliceBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecSliceBoolV(v []bool, canChange bool, d *Decoder) (_ []bool, changed bool) {
	dd := d.d

	slh, containerLenS := d.decSliceHelperStart()
	if containerLenS == 0 {
		if canChange {
			if v == nil {
				v = []bool{}
			} else if len(v) != 0 {
				v = v[:0]
			}
			changed = true
		}
		slh.End()
		return v, changed
	}

	hasLen := containerLenS > 0
	var xlen int
	if hasLen && canChange {
		if containerLenS > cap(v) {
			xlen = decInferLen(containerLenS, d.h.MaxInitLen, 1)
			if xlen <= cap(v) {
				v = v[:xlen]
			} else {
				v = make([]bool, xlen)
			}
			changed = true
		} else if containerLenS != len(v) {
			v = v[:containerLenS]
			changed = true
		}
	}
	j := 0
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && len(v) == 0 {
			if hasLen {
				xlen = decInferLen(containerLenS, d.h.MaxInitLen, 1)
			} else {
				xlen = 8
			}
			v = make([]bool, xlen)
			changed = true
		}
		// if indefinite, etc, then expand the slice if necessary
		var decodeIntoBlank bool
		if j >= len(v) {
			if canChange {
				v = append(v, false)
				changed = true
			} else {
				d.arrayCannotExpand(len(v), j+1)
				decodeIntoBlank = true
			}
		}
		slh.ElemContainerState(j)
		if decodeIntoBlank {
			d.swallow()
		} else {
			v[j] = dd.DecodeBool()
		}
	}
	if canChange {
		if j < len(v) {
			v = v[:j]
			changed = true
		} else if j == 0 && v == nil {
			v = make([]bool, 0)
			changed = true
		}
	}
	slh.End()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfIntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]interface{})
		if v, changed := fastpathTV.DecMapIntfIntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfIntfV(rv2i(rv).(map[interface{}]interface{}), false, d)
}
func (f fastpathT) DecMapIntfIntfX(vp *map[interface{}]interface{}, d *Decoder) {
	if v, changed := f.DecMapIntfIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfIntfV(v map[interface{}]interface{}, canChange bool,
	d *Decoder) (_ map[interface{}]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[interface{}]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk interface{}
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfStringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]string)
		if v, changed := fastpathTV.DecMapIntfStringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfStringV(rv2i(rv).(map[interface{}]string), false, d)
}
func (f fastpathT) DecMapIntfStringX(vp *map[interface{}]string, d *Decoder) {
	if v, changed := f.DecMapIntfStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfStringV(v map[interface{}]string, canChange bool,
	d *Decoder) (_ map[interface{}]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[interface{}]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfUintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]uint)
		if v, changed := fastpathTV.DecMapIntfUintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfUintV(rv2i(rv).(map[interface{}]uint), false, d)
}
func (f fastpathT) DecMapIntfUintX(vp *map[interface{}]uint, d *Decoder) {
	if v, changed := f.DecMapIntfUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUintV(v map[interface{}]uint, canChange bool,
	d *Decoder) (_ map[interface{}]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfUint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]uint8)
		if v, changed := fastpathTV.DecMapIntfUint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfUint8V(rv2i(rv).(map[interface{}]uint8), false, d)
}
func (f fastpathT) DecMapIntfUint8X(vp *map[interface{}]uint8, d *Decoder) {
	if v, changed := f.DecMapIntfUint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint8V(v map[interface{}]uint8, canChange bool,
	d *Decoder) (_ map[interface{}]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[interface{}]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfUint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]uint16)
		if v, changed := fastpathTV.DecMapIntfUint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfUint16V(rv2i(rv).(map[interface{}]uint16), false, d)
}
func (f fastpathT) DecMapIntfUint16X(vp *map[interface{}]uint16, d *Decoder) {
	if v, changed := f.DecMapIntfUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint16V(v map[interface{}]uint16, canChange bool,
	d *Decoder) (_ map[interface{}]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[interface{}]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfUint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]uint32)
		if v, changed := fastpathTV.DecMapIntfUint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfUint32V(rv2i(rv).(map[interface{}]uint32), false, d)
}
func (f fastpathT) DecMapIntfUint32X(vp *map[interface{}]uint32, d *Decoder) {
	if v, changed := f.DecMapIntfUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint32V(v map[interface{}]uint32, canChange bool,
	d *Decoder) (_ map[interface{}]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[interface{}]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfUint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]uint64)
		if v, changed := fastpathTV.DecMapIntfUint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfUint64V(rv2i(rv).(map[interface{}]uint64), false, d)
}
func (f fastpathT) DecMapIntfUint64X(vp *map[interface{}]uint64, d *Decoder) {
	if v, changed := f.DecMapIntfUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUint64V(v map[interface{}]uint64, canChange bool,
	d *Decoder) (_ map[interface{}]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfUintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]uintptr)
		if v, changed := fastpathTV.DecMapIntfUintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfUintptrV(rv2i(rv).(map[interface{}]uintptr), false, d)
}
func (f fastpathT) DecMapIntfUintptrX(vp *map[interface{}]uintptr, d *Decoder) {
	if v, changed := f.DecMapIntfUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfUintptrV(v map[interface{}]uintptr, canChange bool,
	d *Decoder) (_ map[interface{}]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfIntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]int)
		if v, changed := fastpathTV.DecMapIntfIntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfIntV(rv2i(rv).(map[interface{}]int), false, d)
}
func (f fastpathT) DecMapIntfIntX(vp *map[interface{}]int, d *Decoder) {
	if v, changed := f.DecMapIntfIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfIntV(v map[interface{}]int, canChange bool,
	d *Decoder) (_ map[interface{}]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfInt8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]int8)
		if v, changed := fastpathTV.DecMapIntfInt8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfInt8V(rv2i(rv).(map[interface{}]int8), false, d)
}
func (f fastpathT) DecMapIntfInt8X(vp *map[interface{}]int8, d *Decoder) {
	if v, changed := f.DecMapIntfInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt8V(v map[interface{}]int8, canChange bool,
	d *Decoder) (_ map[interface{}]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[interface{}]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfInt16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]int16)
		if v, changed := fastpathTV.DecMapIntfInt16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfInt16V(rv2i(rv).(map[interface{}]int16), false, d)
}
func (f fastpathT) DecMapIntfInt16X(vp *map[interface{}]int16, d *Decoder) {
	if v, changed := f.DecMapIntfInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt16V(v map[interface{}]int16, canChange bool,
	d *Decoder) (_ map[interface{}]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[interface{}]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfInt32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]int32)
		if v, changed := fastpathTV.DecMapIntfInt32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfInt32V(rv2i(rv).(map[interface{}]int32), false, d)
}
func (f fastpathT) DecMapIntfInt32X(vp *map[interface{}]int32, d *Decoder) {
	if v, changed := f.DecMapIntfInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt32V(v map[interface{}]int32, canChange bool,
	d *Decoder) (_ map[interface{}]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[interface{}]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfInt64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]int64)
		if v, changed := fastpathTV.DecMapIntfInt64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfInt64V(rv2i(rv).(map[interface{}]int64), false, d)
}
func (f fastpathT) DecMapIntfInt64X(vp *map[interface{}]int64, d *Decoder) {
	if v, changed := f.DecMapIntfInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfInt64V(v map[interface{}]int64, canChange bool,
	d *Decoder) (_ map[interface{}]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfFloat32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]float32)
		if v, changed := fastpathTV.DecMapIntfFloat32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfFloat32V(rv2i(rv).(map[interface{}]float32), false, d)
}
func (f fastpathT) DecMapIntfFloat32X(vp *map[interface{}]float32, d *Decoder) {
	if v, changed := f.DecMapIntfFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfFloat32V(v map[interface{}]float32, canChange bool,
	d *Decoder) (_ map[interface{}]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[interface{}]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfFloat64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]float64)
		if v, changed := fastpathTV.DecMapIntfFloat64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfFloat64V(rv2i(rv).(map[interface{}]float64), false, d)
}
func (f fastpathT) DecMapIntfFloat64X(vp *map[interface{}]float64, d *Decoder) {
	if v, changed := f.DecMapIntfFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfFloat64V(v map[interface{}]float64, canChange bool,
	d *Decoder) (_ map[interface{}]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[interface{}]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntfBoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[interface{}]bool)
		if v, changed := fastpathTV.DecMapIntfBoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntfBoolV(rv2i(rv).(map[interface{}]bool), false, d)
}
func (f fastpathT) DecMapIntfBoolX(vp *map[interface{}]bool, d *Decoder) {
	if v, changed := f.DecMapIntfBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntfBoolV(v map[interface{}]bool, canChange bool,
	d *Decoder) (_ map[interface{}]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[interface{}]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk interface{}
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = nil
		d.decode(&mk)
		if bv, bok := mk.([]byte); bok {
			mk = d.string(bv)
		}
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringIntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]interface{})
		if v, changed := fastpathTV.DecMapStringIntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringIntfV(rv2i(rv).(map[string]interface{}), false, d)
}
func (f fastpathT) DecMapStringIntfX(vp *map[string]interface{}, d *Decoder) {
	if v, changed := f.DecMapStringIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringIntfV(v map[string]interface{}, canChange bool,
	d *Decoder) (_ map[string]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[string]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk string
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringStringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]string)
		if v, changed := fastpathTV.DecMapStringStringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringStringV(rv2i(rv).(map[string]string), false, d)
}
func (f fastpathT) DecMapStringStringX(vp *map[string]string, d *Decoder) {
	if v, changed := f.DecMapStringStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringStringV(v map[string]string, canChange bool,
	d *Decoder) (_ map[string]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 32)
		v = make(map[string]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringUintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]uint)
		if v, changed := fastpathTV.DecMapStringUintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringUintV(rv2i(rv).(map[string]uint), false, d)
}
func (f fastpathT) DecMapStringUintX(vp *map[string]uint, d *Decoder) {
	if v, changed := f.DecMapStringUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUintV(v map[string]uint, canChange bool,
	d *Decoder) (_ map[string]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringUint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]uint8)
		if v, changed := fastpathTV.DecMapStringUint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringUint8V(rv2i(rv).(map[string]uint8), false, d)
}
func (f fastpathT) DecMapStringUint8X(vp *map[string]uint8, d *Decoder) {
	if v, changed := f.DecMapStringUint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint8V(v map[string]uint8, canChange bool,
	d *Decoder) (_ map[string]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[string]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringUint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]uint16)
		if v, changed := fastpathTV.DecMapStringUint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringUint16V(rv2i(rv).(map[string]uint16), false, d)
}
func (f fastpathT) DecMapStringUint16X(vp *map[string]uint16, d *Decoder) {
	if v, changed := f.DecMapStringUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint16V(v map[string]uint16, canChange bool,
	d *Decoder) (_ map[string]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[string]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringUint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]uint32)
		if v, changed := fastpathTV.DecMapStringUint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringUint32V(rv2i(rv).(map[string]uint32), false, d)
}
func (f fastpathT) DecMapStringUint32X(vp *map[string]uint32, d *Decoder) {
	if v, changed := f.DecMapStringUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint32V(v map[string]uint32, canChange bool,
	d *Decoder) (_ map[string]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[string]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringUint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]uint64)
		if v, changed := fastpathTV.DecMapStringUint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringUint64V(rv2i(rv).(map[string]uint64), false, d)
}
func (f fastpathT) DecMapStringUint64X(vp *map[string]uint64, d *Decoder) {
	if v, changed := f.DecMapStringUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUint64V(v map[string]uint64, canChange bool,
	d *Decoder) (_ map[string]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringUintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]uintptr)
		if v, changed := fastpathTV.DecMapStringUintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringUintptrV(rv2i(rv).(map[string]uintptr), false, d)
}
func (f fastpathT) DecMapStringUintptrX(vp *map[string]uintptr, d *Decoder) {
	if v, changed := f.DecMapStringUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringUintptrV(v map[string]uintptr, canChange bool,
	d *Decoder) (_ map[string]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringIntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]int)
		if v, changed := fastpathTV.DecMapStringIntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringIntV(rv2i(rv).(map[string]int), false, d)
}
func (f fastpathT) DecMapStringIntX(vp *map[string]int, d *Decoder) {
	if v, changed := f.DecMapStringIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringIntV(v map[string]int, canChange bool,
	d *Decoder) (_ map[string]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringInt8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]int8)
		if v, changed := fastpathTV.DecMapStringInt8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringInt8V(rv2i(rv).(map[string]int8), false, d)
}
func (f fastpathT) DecMapStringInt8X(vp *map[string]int8, d *Decoder) {
	if v, changed := f.DecMapStringInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt8V(v map[string]int8, canChange bool,
	d *Decoder) (_ map[string]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[string]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringInt16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]int16)
		if v, changed := fastpathTV.DecMapStringInt16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringInt16V(rv2i(rv).(map[string]int16), false, d)
}
func (f fastpathT) DecMapStringInt16X(vp *map[string]int16, d *Decoder) {
	if v, changed := f.DecMapStringInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt16V(v map[string]int16, canChange bool,
	d *Decoder) (_ map[string]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[string]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringInt32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]int32)
		if v, changed := fastpathTV.DecMapStringInt32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringInt32V(rv2i(rv).(map[string]int32), false, d)
}
func (f fastpathT) DecMapStringInt32X(vp *map[string]int32, d *Decoder) {
	if v, changed := f.DecMapStringInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt32V(v map[string]int32, canChange bool,
	d *Decoder) (_ map[string]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[string]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringInt64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]int64)
		if v, changed := fastpathTV.DecMapStringInt64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringInt64V(rv2i(rv).(map[string]int64), false, d)
}
func (f fastpathT) DecMapStringInt64X(vp *map[string]int64, d *Decoder) {
	if v, changed := f.DecMapStringInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringInt64V(v map[string]int64, canChange bool,
	d *Decoder) (_ map[string]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringFloat32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]float32)
		if v, changed := fastpathTV.DecMapStringFloat32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringFloat32V(rv2i(rv).(map[string]float32), false, d)
}
func (f fastpathT) DecMapStringFloat32X(vp *map[string]float32, d *Decoder) {
	if v, changed := f.DecMapStringFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringFloat32V(v map[string]float32, canChange bool,
	d *Decoder) (_ map[string]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[string]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringFloat64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]float64)
		if v, changed := fastpathTV.DecMapStringFloat64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringFloat64V(rv2i(rv).(map[string]float64), false, d)
}
func (f fastpathT) DecMapStringFloat64X(vp *map[string]float64, d *Decoder) {
	if v, changed := f.DecMapStringFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringFloat64V(v map[string]float64, canChange bool,
	d *Decoder) (_ map[string]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[string]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapStringBoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[string]bool)
		if v, changed := fastpathTV.DecMapStringBoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapStringBoolV(rv2i(rv).(map[string]bool), false, d)
}
func (f fastpathT) DecMapStringBoolX(vp *map[string]bool, d *Decoder) {
	if v, changed := f.DecMapStringBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapStringBoolV(v map[string]bool, canChange bool,
	d *Decoder) (_ map[string]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[string]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk string
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeString()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]interface{})
		if v, changed := fastpathTV.DecMapFloat32IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32IntfV(rv2i(rv).(map[float32]interface{}), false, d)
}
func (f fastpathT) DecMapFloat32IntfX(vp *map[float32]interface{}, d *Decoder) {
	if v, changed := f.DecMapFloat32IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32IntfV(v map[float32]interface{}, canChange bool,
	d *Decoder) (_ map[float32]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[float32]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk float32
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]string)
		if v, changed := fastpathTV.DecMapFloat32StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32StringV(rv2i(rv).(map[float32]string), false, d)
}
func (f fastpathT) DecMapFloat32StringX(vp *map[float32]string, d *Decoder) {
	if v, changed := f.DecMapFloat32StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32StringV(v map[float32]string, canChange bool,
	d *Decoder) (_ map[float32]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[float32]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]uint)
		if v, changed := fastpathTV.DecMapFloat32UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32UintV(rv2i(rv).(map[float32]uint), false, d)
}
func (f fastpathT) DecMapFloat32UintX(vp *map[float32]uint, d *Decoder) {
	if v, changed := f.DecMapFloat32UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32UintV(v map[float32]uint, canChange bool,
	d *Decoder) (_ map[float32]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]uint8)
		if v, changed := fastpathTV.DecMapFloat32Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Uint8V(rv2i(rv).(map[float32]uint8), false, d)
}
func (f fastpathT) DecMapFloat32Uint8X(vp *map[float32]uint8, d *Decoder) {
	if v, changed := f.DecMapFloat32Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint8V(v map[float32]uint8, canChange bool,
	d *Decoder) (_ map[float32]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[float32]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]uint16)
		if v, changed := fastpathTV.DecMapFloat32Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Uint16V(rv2i(rv).(map[float32]uint16), false, d)
}
func (f fastpathT) DecMapFloat32Uint16X(vp *map[float32]uint16, d *Decoder) {
	if v, changed := f.DecMapFloat32Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint16V(v map[float32]uint16, canChange bool,
	d *Decoder) (_ map[float32]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[float32]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]uint32)
		if v, changed := fastpathTV.DecMapFloat32Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Uint32V(rv2i(rv).(map[float32]uint32), false, d)
}
func (f fastpathT) DecMapFloat32Uint32X(vp *map[float32]uint32, d *Decoder) {
	if v, changed := f.DecMapFloat32Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint32V(v map[float32]uint32, canChange bool,
	d *Decoder) (_ map[float32]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[float32]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]uint64)
		if v, changed := fastpathTV.DecMapFloat32Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Uint64V(rv2i(rv).(map[float32]uint64), false, d)
}
func (f fastpathT) DecMapFloat32Uint64X(vp *map[float32]uint64, d *Decoder) {
	if v, changed := f.DecMapFloat32Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Uint64V(v map[float32]uint64, canChange bool,
	d *Decoder) (_ map[float32]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]uintptr)
		if v, changed := fastpathTV.DecMapFloat32UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32UintptrV(rv2i(rv).(map[float32]uintptr), false, d)
}
func (f fastpathT) DecMapFloat32UintptrX(vp *map[float32]uintptr, d *Decoder) {
	if v, changed := f.DecMapFloat32UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32UintptrV(v map[float32]uintptr, canChange bool,
	d *Decoder) (_ map[float32]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]int)
		if v, changed := fastpathTV.DecMapFloat32IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32IntV(rv2i(rv).(map[float32]int), false, d)
}
func (f fastpathT) DecMapFloat32IntX(vp *map[float32]int, d *Decoder) {
	if v, changed := f.DecMapFloat32IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32IntV(v map[float32]int, canChange bool,
	d *Decoder) (_ map[float32]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]int8)
		if v, changed := fastpathTV.DecMapFloat32Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Int8V(rv2i(rv).(map[float32]int8), false, d)
}
func (f fastpathT) DecMapFloat32Int8X(vp *map[float32]int8, d *Decoder) {
	if v, changed := f.DecMapFloat32Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int8V(v map[float32]int8, canChange bool,
	d *Decoder) (_ map[float32]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[float32]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]int16)
		if v, changed := fastpathTV.DecMapFloat32Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Int16V(rv2i(rv).(map[float32]int16), false, d)
}
func (f fastpathT) DecMapFloat32Int16X(vp *map[float32]int16, d *Decoder) {
	if v, changed := f.DecMapFloat32Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int16V(v map[float32]int16, canChange bool,
	d *Decoder) (_ map[float32]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[float32]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]int32)
		if v, changed := fastpathTV.DecMapFloat32Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Int32V(rv2i(rv).(map[float32]int32), false, d)
}
func (f fastpathT) DecMapFloat32Int32X(vp *map[float32]int32, d *Decoder) {
	if v, changed := f.DecMapFloat32Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int32V(v map[float32]int32, canChange bool,
	d *Decoder) (_ map[float32]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[float32]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]int64)
		if v, changed := fastpathTV.DecMapFloat32Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Int64V(rv2i(rv).(map[float32]int64), false, d)
}
func (f fastpathT) DecMapFloat32Int64X(vp *map[float32]int64, d *Decoder) {
	if v, changed := f.DecMapFloat32Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Int64V(v map[float32]int64, canChange bool,
	d *Decoder) (_ map[float32]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]float32)
		if v, changed := fastpathTV.DecMapFloat32Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Float32V(rv2i(rv).(map[float32]float32), false, d)
}
func (f fastpathT) DecMapFloat32Float32X(vp *map[float32]float32, d *Decoder) {
	if v, changed := f.DecMapFloat32Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Float32V(v map[float32]float32, canChange bool,
	d *Decoder) (_ map[float32]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[float32]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]float64)
		if v, changed := fastpathTV.DecMapFloat32Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32Float64V(rv2i(rv).(map[float32]float64), false, d)
}
func (f fastpathT) DecMapFloat32Float64X(vp *map[float32]float64, d *Decoder) {
	if v, changed := f.DecMapFloat32Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32Float64V(v map[float32]float64, canChange bool,
	d *Decoder) (_ map[float32]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float32]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat32BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float32]bool)
		if v, changed := fastpathTV.DecMapFloat32BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat32BoolV(rv2i(rv).(map[float32]bool), false, d)
}
func (f fastpathT) DecMapFloat32BoolX(vp *map[float32]bool, d *Decoder) {
	if v, changed := f.DecMapFloat32BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat32BoolV(v map[float32]bool, canChange bool,
	d *Decoder) (_ map[float32]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[float32]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float32
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = float32(dd.DecodeFloat(true))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]interface{})
		if v, changed := fastpathTV.DecMapFloat64IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64IntfV(rv2i(rv).(map[float64]interface{}), false, d)
}
func (f fastpathT) DecMapFloat64IntfX(vp *map[float64]interface{}, d *Decoder) {
	if v, changed := f.DecMapFloat64IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64IntfV(v map[float64]interface{}, canChange bool,
	d *Decoder) (_ map[float64]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[float64]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk float64
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]string)
		if v, changed := fastpathTV.DecMapFloat64StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64StringV(rv2i(rv).(map[float64]string), false, d)
}
func (f fastpathT) DecMapFloat64StringX(vp *map[float64]string, d *Decoder) {
	if v, changed := f.DecMapFloat64StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64StringV(v map[float64]string, canChange bool,
	d *Decoder) (_ map[float64]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[float64]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]uint)
		if v, changed := fastpathTV.DecMapFloat64UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64UintV(rv2i(rv).(map[float64]uint), false, d)
}
func (f fastpathT) DecMapFloat64UintX(vp *map[float64]uint, d *Decoder) {
	if v, changed := f.DecMapFloat64UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64UintV(v map[float64]uint, canChange bool,
	d *Decoder) (_ map[float64]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]uint8)
		if v, changed := fastpathTV.DecMapFloat64Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Uint8V(rv2i(rv).(map[float64]uint8), false, d)
}
func (f fastpathT) DecMapFloat64Uint8X(vp *map[float64]uint8, d *Decoder) {
	if v, changed := f.DecMapFloat64Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint8V(v map[float64]uint8, canChange bool,
	d *Decoder) (_ map[float64]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[float64]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]uint16)
		if v, changed := fastpathTV.DecMapFloat64Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Uint16V(rv2i(rv).(map[float64]uint16), false, d)
}
func (f fastpathT) DecMapFloat64Uint16X(vp *map[float64]uint16, d *Decoder) {
	if v, changed := f.DecMapFloat64Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint16V(v map[float64]uint16, canChange bool,
	d *Decoder) (_ map[float64]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[float64]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]uint32)
		if v, changed := fastpathTV.DecMapFloat64Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Uint32V(rv2i(rv).(map[float64]uint32), false, d)
}
func (f fastpathT) DecMapFloat64Uint32X(vp *map[float64]uint32, d *Decoder) {
	if v, changed := f.DecMapFloat64Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint32V(v map[float64]uint32, canChange bool,
	d *Decoder) (_ map[float64]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float64]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]uint64)
		if v, changed := fastpathTV.DecMapFloat64Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Uint64V(rv2i(rv).(map[float64]uint64), false, d)
}
func (f fastpathT) DecMapFloat64Uint64X(vp *map[float64]uint64, d *Decoder) {
	if v, changed := f.DecMapFloat64Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Uint64V(v map[float64]uint64, canChange bool,
	d *Decoder) (_ map[float64]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]uintptr)
		if v, changed := fastpathTV.DecMapFloat64UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64UintptrV(rv2i(rv).(map[float64]uintptr), false, d)
}
func (f fastpathT) DecMapFloat64UintptrX(vp *map[float64]uintptr, d *Decoder) {
	if v, changed := f.DecMapFloat64UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64UintptrV(v map[float64]uintptr, canChange bool,
	d *Decoder) (_ map[float64]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]int)
		if v, changed := fastpathTV.DecMapFloat64IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64IntV(rv2i(rv).(map[float64]int), false, d)
}
func (f fastpathT) DecMapFloat64IntX(vp *map[float64]int, d *Decoder) {
	if v, changed := f.DecMapFloat64IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64IntV(v map[float64]int, canChange bool,
	d *Decoder) (_ map[float64]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]int8)
		if v, changed := fastpathTV.DecMapFloat64Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Int8V(rv2i(rv).(map[float64]int8), false, d)
}
func (f fastpathT) DecMapFloat64Int8X(vp *map[float64]int8, d *Decoder) {
	if v, changed := f.DecMapFloat64Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int8V(v map[float64]int8, canChange bool,
	d *Decoder) (_ map[float64]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[float64]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]int16)
		if v, changed := fastpathTV.DecMapFloat64Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Int16V(rv2i(rv).(map[float64]int16), false, d)
}
func (f fastpathT) DecMapFloat64Int16X(vp *map[float64]int16, d *Decoder) {
	if v, changed := f.DecMapFloat64Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int16V(v map[float64]int16, canChange bool,
	d *Decoder) (_ map[float64]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[float64]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]int32)
		if v, changed := fastpathTV.DecMapFloat64Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Int32V(rv2i(rv).(map[float64]int32), false, d)
}
func (f fastpathT) DecMapFloat64Int32X(vp *map[float64]int32, d *Decoder) {
	if v, changed := f.DecMapFloat64Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int32V(v map[float64]int32, canChange bool,
	d *Decoder) (_ map[float64]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float64]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]int64)
		if v, changed := fastpathTV.DecMapFloat64Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Int64V(rv2i(rv).(map[float64]int64), false, d)
}
func (f fastpathT) DecMapFloat64Int64X(vp *map[float64]int64, d *Decoder) {
	if v, changed := f.DecMapFloat64Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Int64V(v map[float64]int64, canChange bool,
	d *Decoder) (_ map[float64]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]float32)
		if v, changed := fastpathTV.DecMapFloat64Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Float32V(rv2i(rv).(map[float64]float32), false, d)
}
func (f fastpathT) DecMapFloat64Float32X(vp *map[float64]float32, d *Decoder) {
	if v, changed := f.DecMapFloat64Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Float32V(v map[float64]float32, canChange bool,
	d *Decoder) (_ map[float64]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[float64]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]float64)
		if v, changed := fastpathTV.DecMapFloat64Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64Float64V(rv2i(rv).(map[float64]float64), false, d)
}
func (f fastpathT) DecMapFloat64Float64X(vp *map[float64]float64, d *Decoder) {
	if v, changed := f.DecMapFloat64Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64Float64V(v map[float64]float64, canChange bool,
	d *Decoder) (_ map[float64]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[float64]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapFloat64BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[float64]bool)
		if v, changed := fastpathTV.DecMapFloat64BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapFloat64BoolV(rv2i(rv).(map[float64]bool), false, d)
}
func (f fastpathT) DecMapFloat64BoolX(vp *map[float64]bool, d *Decoder) {
	if v, changed := f.DecMapFloat64BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapFloat64BoolV(v map[float64]bool, canChange bool,
	d *Decoder) (_ map[float64]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[float64]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk float64
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeFloat(false)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintIntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]interface{})
		if v, changed := fastpathTV.DecMapUintIntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintIntfV(rv2i(rv).(map[uint]interface{}), false, d)
}
func (f fastpathT) DecMapUintIntfX(vp *map[uint]interface{}, d *Decoder) {
	if v, changed := f.DecMapUintIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintIntfV(v map[uint]interface{}, canChange bool,
	d *Decoder) (_ map[uint]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk uint
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintStringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]string)
		if v, changed := fastpathTV.DecMapUintStringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintStringV(rv2i(rv).(map[uint]string), false, d)
}
func (f fastpathT) DecMapUintStringX(vp *map[uint]string, d *Decoder) {
	if v, changed := f.DecMapUintStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintStringV(v map[uint]string, canChange bool,
	d *Decoder) (_ map[uint]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintUintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]uint)
		if v, changed := fastpathTV.DecMapUintUintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintUintV(rv2i(rv).(map[uint]uint), false, d)
}
func (f fastpathT) DecMapUintUintX(vp *map[uint]uint, d *Decoder) {
	if v, changed := f.DecMapUintUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUintV(v map[uint]uint, canChange bool,
	d *Decoder) (_ map[uint]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintUint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]uint8)
		if v, changed := fastpathTV.DecMapUintUint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintUint8V(rv2i(rv).(map[uint]uint8), false, d)
}
func (f fastpathT) DecMapUintUint8X(vp *map[uint]uint8, d *Decoder) {
	if v, changed := f.DecMapUintUint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint8V(v map[uint]uint8, canChange bool,
	d *Decoder) (_ map[uint]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintUint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]uint16)
		if v, changed := fastpathTV.DecMapUintUint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintUint16V(rv2i(rv).(map[uint]uint16), false, d)
}
func (f fastpathT) DecMapUintUint16X(vp *map[uint]uint16, d *Decoder) {
	if v, changed := f.DecMapUintUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint16V(v map[uint]uint16, canChange bool,
	d *Decoder) (_ map[uint]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintUint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]uint32)
		if v, changed := fastpathTV.DecMapUintUint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintUint32V(rv2i(rv).(map[uint]uint32), false, d)
}
func (f fastpathT) DecMapUintUint32X(vp *map[uint]uint32, d *Decoder) {
	if v, changed := f.DecMapUintUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint32V(v map[uint]uint32, canChange bool,
	d *Decoder) (_ map[uint]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintUint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]uint64)
		if v, changed := fastpathTV.DecMapUintUint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintUint64V(rv2i(rv).(map[uint]uint64), false, d)
}
func (f fastpathT) DecMapUintUint64X(vp *map[uint]uint64, d *Decoder) {
	if v, changed := f.DecMapUintUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUint64V(v map[uint]uint64, canChange bool,
	d *Decoder) (_ map[uint]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintUintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]uintptr)
		if v, changed := fastpathTV.DecMapUintUintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintUintptrV(rv2i(rv).(map[uint]uintptr), false, d)
}
func (f fastpathT) DecMapUintUintptrX(vp *map[uint]uintptr, d *Decoder) {
	if v, changed := f.DecMapUintUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintUintptrV(v map[uint]uintptr, canChange bool,
	d *Decoder) (_ map[uint]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintIntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]int)
		if v, changed := fastpathTV.DecMapUintIntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintIntV(rv2i(rv).(map[uint]int), false, d)
}
func (f fastpathT) DecMapUintIntX(vp *map[uint]int, d *Decoder) {
	if v, changed := f.DecMapUintIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintIntV(v map[uint]int, canChange bool,
	d *Decoder) (_ map[uint]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintInt8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]int8)
		if v, changed := fastpathTV.DecMapUintInt8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintInt8V(rv2i(rv).(map[uint]int8), false, d)
}
func (f fastpathT) DecMapUintInt8X(vp *map[uint]int8, d *Decoder) {
	if v, changed := f.DecMapUintInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt8V(v map[uint]int8, canChange bool,
	d *Decoder) (_ map[uint]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintInt16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]int16)
		if v, changed := fastpathTV.DecMapUintInt16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintInt16V(rv2i(rv).(map[uint]int16), false, d)
}
func (f fastpathT) DecMapUintInt16X(vp *map[uint]int16, d *Decoder) {
	if v, changed := f.DecMapUintInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt16V(v map[uint]int16, canChange bool,
	d *Decoder) (_ map[uint]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintInt32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]int32)
		if v, changed := fastpathTV.DecMapUintInt32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintInt32V(rv2i(rv).(map[uint]int32), false, d)
}
func (f fastpathT) DecMapUintInt32X(vp *map[uint]int32, d *Decoder) {
	if v, changed := f.DecMapUintInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt32V(v map[uint]int32, canChange bool,
	d *Decoder) (_ map[uint]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintInt64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]int64)
		if v, changed := fastpathTV.DecMapUintInt64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintInt64V(rv2i(rv).(map[uint]int64), false, d)
}
func (f fastpathT) DecMapUintInt64X(vp *map[uint]int64, d *Decoder) {
	if v, changed := f.DecMapUintInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintInt64V(v map[uint]int64, canChange bool,
	d *Decoder) (_ map[uint]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintFloat32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]float32)
		if v, changed := fastpathTV.DecMapUintFloat32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintFloat32V(rv2i(rv).(map[uint]float32), false, d)
}
func (f fastpathT) DecMapUintFloat32X(vp *map[uint]float32, d *Decoder) {
	if v, changed := f.DecMapUintFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintFloat32V(v map[uint]float32, canChange bool,
	d *Decoder) (_ map[uint]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintFloat64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]float64)
		if v, changed := fastpathTV.DecMapUintFloat64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintFloat64V(rv2i(rv).(map[uint]float64), false, d)
}
func (f fastpathT) DecMapUintFloat64X(vp *map[uint]float64, d *Decoder) {
	if v, changed := f.DecMapUintFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintFloat64V(v map[uint]float64, canChange bool,
	d *Decoder) (_ map[uint]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintBoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint]bool)
		if v, changed := fastpathTV.DecMapUintBoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintBoolV(rv2i(rv).(map[uint]bool), false, d)
}
func (f fastpathT) DecMapUintBoolX(vp *map[uint]bool, d *Decoder) {
	if v, changed := f.DecMapUintBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintBoolV(v map[uint]bool, canChange bool,
	d *Decoder) (_ map[uint]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]interface{})
		if v, changed := fastpathTV.DecMapUint8IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8IntfV(rv2i(rv).(map[uint8]interface{}), false, d)
}
func (f fastpathT) DecMapUint8IntfX(vp *map[uint8]interface{}, d *Decoder) {
	if v, changed := f.DecMapUint8IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8IntfV(v map[uint8]interface{}, canChange bool,
	d *Decoder) (_ map[uint8]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[uint8]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk uint8
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]string)
		if v, changed := fastpathTV.DecMapUint8StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8StringV(rv2i(rv).(map[uint8]string), false, d)
}
func (f fastpathT) DecMapUint8StringX(vp *map[uint8]string, d *Decoder) {
	if v, changed := f.DecMapUint8StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8StringV(v map[uint8]string, canChange bool,
	d *Decoder) (_ map[uint8]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[uint8]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]uint)
		if v, changed := fastpathTV.DecMapUint8UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8UintV(rv2i(rv).(map[uint8]uint), false, d)
}
func (f fastpathT) DecMapUint8UintX(vp *map[uint8]uint, d *Decoder) {
	if v, changed := f.DecMapUint8UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8UintV(v map[uint8]uint, canChange bool,
	d *Decoder) (_ map[uint8]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]uint8)
		if v, changed := fastpathTV.DecMapUint8Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Uint8V(rv2i(rv).(map[uint8]uint8), false, d)
}
func (f fastpathT) DecMapUint8Uint8X(vp *map[uint8]uint8, d *Decoder) {
	if v, changed := f.DecMapUint8Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint8V(v map[uint8]uint8, canChange bool,
	d *Decoder) (_ map[uint8]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[uint8]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]uint16)
		if v, changed := fastpathTV.DecMapUint8Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Uint16V(rv2i(rv).(map[uint8]uint16), false, d)
}
func (f fastpathT) DecMapUint8Uint16X(vp *map[uint8]uint16, d *Decoder) {
	if v, changed := f.DecMapUint8Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint16V(v map[uint8]uint16, canChange bool,
	d *Decoder) (_ map[uint8]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint8]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]uint32)
		if v, changed := fastpathTV.DecMapUint8Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Uint32V(rv2i(rv).(map[uint8]uint32), false, d)
}
func (f fastpathT) DecMapUint8Uint32X(vp *map[uint8]uint32, d *Decoder) {
	if v, changed := f.DecMapUint8Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint32V(v map[uint8]uint32, canChange bool,
	d *Decoder) (_ map[uint8]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint8]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]uint64)
		if v, changed := fastpathTV.DecMapUint8Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Uint64V(rv2i(rv).(map[uint8]uint64), false, d)
}
func (f fastpathT) DecMapUint8Uint64X(vp *map[uint8]uint64, d *Decoder) {
	if v, changed := f.DecMapUint8Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Uint64V(v map[uint8]uint64, canChange bool,
	d *Decoder) (_ map[uint8]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]uintptr)
		if v, changed := fastpathTV.DecMapUint8UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8UintptrV(rv2i(rv).(map[uint8]uintptr), false, d)
}
func (f fastpathT) DecMapUint8UintptrX(vp *map[uint8]uintptr, d *Decoder) {
	if v, changed := f.DecMapUint8UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8UintptrV(v map[uint8]uintptr, canChange bool,
	d *Decoder) (_ map[uint8]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]int)
		if v, changed := fastpathTV.DecMapUint8IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8IntV(rv2i(rv).(map[uint8]int), false, d)
}
func (f fastpathT) DecMapUint8IntX(vp *map[uint8]int, d *Decoder) {
	if v, changed := f.DecMapUint8IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8IntV(v map[uint8]int, canChange bool,
	d *Decoder) (_ map[uint8]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]int8)
		if v, changed := fastpathTV.DecMapUint8Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Int8V(rv2i(rv).(map[uint8]int8), false, d)
}
func (f fastpathT) DecMapUint8Int8X(vp *map[uint8]int8, d *Decoder) {
	if v, changed := f.DecMapUint8Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int8V(v map[uint8]int8, canChange bool,
	d *Decoder) (_ map[uint8]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[uint8]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]int16)
		if v, changed := fastpathTV.DecMapUint8Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Int16V(rv2i(rv).(map[uint8]int16), false, d)
}
func (f fastpathT) DecMapUint8Int16X(vp *map[uint8]int16, d *Decoder) {
	if v, changed := f.DecMapUint8Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int16V(v map[uint8]int16, canChange bool,
	d *Decoder) (_ map[uint8]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint8]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]int32)
		if v, changed := fastpathTV.DecMapUint8Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Int32V(rv2i(rv).(map[uint8]int32), false, d)
}
func (f fastpathT) DecMapUint8Int32X(vp *map[uint8]int32, d *Decoder) {
	if v, changed := f.DecMapUint8Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int32V(v map[uint8]int32, canChange bool,
	d *Decoder) (_ map[uint8]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint8]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]int64)
		if v, changed := fastpathTV.DecMapUint8Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Int64V(rv2i(rv).(map[uint8]int64), false, d)
}
func (f fastpathT) DecMapUint8Int64X(vp *map[uint8]int64, d *Decoder) {
	if v, changed := f.DecMapUint8Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Int64V(v map[uint8]int64, canChange bool,
	d *Decoder) (_ map[uint8]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]float32)
		if v, changed := fastpathTV.DecMapUint8Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Float32V(rv2i(rv).(map[uint8]float32), false, d)
}
func (f fastpathT) DecMapUint8Float32X(vp *map[uint8]float32, d *Decoder) {
	if v, changed := f.DecMapUint8Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Float32V(v map[uint8]float32, canChange bool,
	d *Decoder) (_ map[uint8]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint8]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]float64)
		if v, changed := fastpathTV.DecMapUint8Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8Float64V(rv2i(rv).(map[uint8]float64), false, d)
}
func (f fastpathT) DecMapUint8Float64X(vp *map[uint8]float64, d *Decoder) {
	if v, changed := f.DecMapUint8Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8Float64V(v map[uint8]float64, canChange bool,
	d *Decoder) (_ map[uint8]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint8]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint8BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint8]bool)
		if v, changed := fastpathTV.DecMapUint8BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint8BoolV(rv2i(rv).(map[uint8]bool), false, d)
}
func (f fastpathT) DecMapUint8BoolX(vp *map[uint8]bool, d *Decoder) {
	if v, changed := f.DecMapUint8BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint8BoolV(v map[uint8]bool, canChange bool,
	d *Decoder) (_ map[uint8]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[uint8]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint8
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint8(dd.DecodeUint(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]interface{})
		if v, changed := fastpathTV.DecMapUint16IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16IntfV(rv2i(rv).(map[uint16]interface{}), false, d)
}
func (f fastpathT) DecMapUint16IntfX(vp *map[uint16]interface{}, d *Decoder) {
	if v, changed := f.DecMapUint16IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16IntfV(v map[uint16]interface{}, canChange bool,
	d *Decoder) (_ map[uint16]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[uint16]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk uint16
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]string)
		if v, changed := fastpathTV.DecMapUint16StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16StringV(rv2i(rv).(map[uint16]string), false, d)
}
func (f fastpathT) DecMapUint16StringX(vp *map[uint16]string, d *Decoder) {
	if v, changed := f.DecMapUint16StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16StringV(v map[uint16]string, canChange bool,
	d *Decoder) (_ map[uint16]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[uint16]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]uint)
		if v, changed := fastpathTV.DecMapUint16UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16UintV(rv2i(rv).(map[uint16]uint), false, d)
}
func (f fastpathT) DecMapUint16UintX(vp *map[uint16]uint, d *Decoder) {
	if v, changed := f.DecMapUint16UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16UintV(v map[uint16]uint, canChange bool,
	d *Decoder) (_ map[uint16]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]uint8)
		if v, changed := fastpathTV.DecMapUint16Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Uint8V(rv2i(rv).(map[uint16]uint8), false, d)
}
func (f fastpathT) DecMapUint16Uint8X(vp *map[uint16]uint8, d *Decoder) {
	if v, changed := f.DecMapUint16Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint8V(v map[uint16]uint8, canChange bool,
	d *Decoder) (_ map[uint16]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint16]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]uint16)
		if v, changed := fastpathTV.DecMapUint16Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Uint16V(rv2i(rv).(map[uint16]uint16), false, d)
}
func (f fastpathT) DecMapUint16Uint16X(vp *map[uint16]uint16, d *Decoder) {
	if v, changed := f.DecMapUint16Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint16V(v map[uint16]uint16, canChange bool,
	d *Decoder) (_ map[uint16]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[uint16]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]uint32)
		if v, changed := fastpathTV.DecMapUint16Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Uint32V(rv2i(rv).(map[uint16]uint32), false, d)
}
func (f fastpathT) DecMapUint16Uint32X(vp *map[uint16]uint32, d *Decoder) {
	if v, changed := f.DecMapUint16Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint32V(v map[uint16]uint32, canChange bool,
	d *Decoder) (_ map[uint16]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint16]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]uint64)
		if v, changed := fastpathTV.DecMapUint16Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Uint64V(rv2i(rv).(map[uint16]uint64), false, d)
}
func (f fastpathT) DecMapUint16Uint64X(vp *map[uint16]uint64, d *Decoder) {
	if v, changed := f.DecMapUint16Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Uint64V(v map[uint16]uint64, canChange bool,
	d *Decoder) (_ map[uint16]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]uintptr)
		if v, changed := fastpathTV.DecMapUint16UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16UintptrV(rv2i(rv).(map[uint16]uintptr), false, d)
}
func (f fastpathT) DecMapUint16UintptrX(vp *map[uint16]uintptr, d *Decoder) {
	if v, changed := f.DecMapUint16UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16UintptrV(v map[uint16]uintptr, canChange bool,
	d *Decoder) (_ map[uint16]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]int)
		if v, changed := fastpathTV.DecMapUint16IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16IntV(rv2i(rv).(map[uint16]int), false, d)
}
func (f fastpathT) DecMapUint16IntX(vp *map[uint16]int, d *Decoder) {
	if v, changed := f.DecMapUint16IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16IntV(v map[uint16]int, canChange bool,
	d *Decoder) (_ map[uint16]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]int8)
		if v, changed := fastpathTV.DecMapUint16Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Int8V(rv2i(rv).(map[uint16]int8), false, d)
}
func (f fastpathT) DecMapUint16Int8X(vp *map[uint16]int8, d *Decoder) {
	if v, changed := f.DecMapUint16Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int8V(v map[uint16]int8, canChange bool,
	d *Decoder) (_ map[uint16]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint16]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]int16)
		if v, changed := fastpathTV.DecMapUint16Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Int16V(rv2i(rv).(map[uint16]int16), false, d)
}
func (f fastpathT) DecMapUint16Int16X(vp *map[uint16]int16, d *Decoder) {
	if v, changed := f.DecMapUint16Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int16V(v map[uint16]int16, canChange bool,
	d *Decoder) (_ map[uint16]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[uint16]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]int32)
		if v, changed := fastpathTV.DecMapUint16Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Int32V(rv2i(rv).(map[uint16]int32), false, d)
}
func (f fastpathT) DecMapUint16Int32X(vp *map[uint16]int32, d *Decoder) {
	if v, changed := f.DecMapUint16Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int32V(v map[uint16]int32, canChange bool,
	d *Decoder) (_ map[uint16]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint16]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]int64)
		if v, changed := fastpathTV.DecMapUint16Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Int64V(rv2i(rv).(map[uint16]int64), false, d)
}
func (f fastpathT) DecMapUint16Int64X(vp *map[uint16]int64, d *Decoder) {
	if v, changed := f.DecMapUint16Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Int64V(v map[uint16]int64, canChange bool,
	d *Decoder) (_ map[uint16]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]float32)
		if v, changed := fastpathTV.DecMapUint16Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Float32V(rv2i(rv).(map[uint16]float32), false, d)
}
func (f fastpathT) DecMapUint16Float32X(vp *map[uint16]float32, d *Decoder) {
	if v, changed := f.DecMapUint16Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Float32V(v map[uint16]float32, canChange bool,
	d *Decoder) (_ map[uint16]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint16]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]float64)
		if v, changed := fastpathTV.DecMapUint16Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16Float64V(rv2i(rv).(map[uint16]float64), false, d)
}
func (f fastpathT) DecMapUint16Float64X(vp *map[uint16]float64, d *Decoder) {
	if v, changed := f.DecMapUint16Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16Float64V(v map[uint16]float64, canChange bool,
	d *Decoder) (_ map[uint16]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint16]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint16BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint16]bool)
		if v, changed := fastpathTV.DecMapUint16BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint16BoolV(rv2i(rv).(map[uint16]bool), false, d)
}
func (f fastpathT) DecMapUint16BoolX(vp *map[uint16]bool, d *Decoder) {
	if v, changed := f.DecMapUint16BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint16BoolV(v map[uint16]bool, canChange bool,
	d *Decoder) (_ map[uint16]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[uint16]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint16
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint16(dd.DecodeUint(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]interface{})
		if v, changed := fastpathTV.DecMapUint32IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32IntfV(rv2i(rv).(map[uint32]interface{}), false, d)
}
func (f fastpathT) DecMapUint32IntfX(vp *map[uint32]interface{}, d *Decoder) {
	if v, changed := f.DecMapUint32IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32IntfV(v map[uint32]interface{}, canChange bool,
	d *Decoder) (_ map[uint32]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[uint32]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk uint32
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]string)
		if v, changed := fastpathTV.DecMapUint32StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32StringV(rv2i(rv).(map[uint32]string), false, d)
}
func (f fastpathT) DecMapUint32StringX(vp *map[uint32]string, d *Decoder) {
	if v, changed := f.DecMapUint32StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32StringV(v map[uint32]string, canChange bool,
	d *Decoder) (_ map[uint32]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[uint32]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]uint)
		if v, changed := fastpathTV.DecMapUint32UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32UintV(rv2i(rv).(map[uint32]uint), false, d)
}
func (f fastpathT) DecMapUint32UintX(vp *map[uint32]uint, d *Decoder) {
	if v, changed := f.DecMapUint32UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32UintV(v map[uint32]uint, canChange bool,
	d *Decoder) (_ map[uint32]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]uint8)
		if v, changed := fastpathTV.DecMapUint32Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Uint8V(rv2i(rv).(map[uint32]uint8), false, d)
}
func (f fastpathT) DecMapUint32Uint8X(vp *map[uint32]uint8, d *Decoder) {
	if v, changed := f.DecMapUint32Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint8V(v map[uint32]uint8, canChange bool,
	d *Decoder) (_ map[uint32]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint32]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]uint16)
		if v, changed := fastpathTV.DecMapUint32Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Uint16V(rv2i(rv).(map[uint32]uint16), false, d)
}
func (f fastpathT) DecMapUint32Uint16X(vp *map[uint32]uint16, d *Decoder) {
	if v, changed := f.DecMapUint32Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint16V(v map[uint32]uint16, canChange bool,
	d *Decoder) (_ map[uint32]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint32]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]uint32)
		if v, changed := fastpathTV.DecMapUint32Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Uint32V(rv2i(rv).(map[uint32]uint32), false, d)
}
func (f fastpathT) DecMapUint32Uint32X(vp *map[uint32]uint32, d *Decoder) {
	if v, changed := f.DecMapUint32Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint32V(v map[uint32]uint32, canChange bool,
	d *Decoder) (_ map[uint32]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[uint32]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]uint64)
		if v, changed := fastpathTV.DecMapUint32Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Uint64V(rv2i(rv).(map[uint32]uint64), false, d)
}
func (f fastpathT) DecMapUint32Uint64X(vp *map[uint32]uint64, d *Decoder) {
	if v, changed := f.DecMapUint32Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Uint64V(v map[uint32]uint64, canChange bool,
	d *Decoder) (_ map[uint32]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]uintptr)
		if v, changed := fastpathTV.DecMapUint32UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32UintptrV(rv2i(rv).(map[uint32]uintptr), false, d)
}
func (f fastpathT) DecMapUint32UintptrX(vp *map[uint32]uintptr, d *Decoder) {
	if v, changed := f.DecMapUint32UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32UintptrV(v map[uint32]uintptr, canChange bool,
	d *Decoder) (_ map[uint32]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]int)
		if v, changed := fastpathTV.DecMapUint32IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32IntV(rv2i(rv).(map[uint32]int), false, d)
}
func (f fastpathT) DecMapUint32IntX(vp *map[uint32]int, d *Decoder) {
	if v, changed := f.DecMapUint32IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32IntV(v map[uint32]int, canChange bool,
	d *Decoder) (_ map[uint32]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]int8)
		if v, changed := fastpathTV.DecMapUint32Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Int8V(rv2i(rv).(map[uint32]int8), false, d)
}
func (f fastpathT) DecMapUint32Int8X(vp *map[uint32]int8, d *Decoder) {
	if v, changed := f.DecMapUint32Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int8V(v map[uint32]int8, canChange bool,
	d *Decoder) (_ map[uint32]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint32]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]int16)
		if v, changed := fastpathTV.DecMapUint32Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Int16V(rv2i(rv).(map[uint32]int16), false, d)
}
func (f fastpathT) DecMapUint32Int16X(vp *map[uint32]int16, d *Decoder) {
	if v, changed := f.DecMapUint32Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int16V(v map[uint32]int16, canChange bool,
	d *Decoder) (_ map[uint32]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[uint32]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]int32)
		if v, changed := fastpathTV.DecMapUint32Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Int32V(rv2i(rv).(map[uint32]int32), false, d)
}
func (f fastpathT) DecMapUint32Int32X(vp *map[uint32]int32, d *Decoder) {
	if v, changed := f.DecMapUint32Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int32V(v map[uint32]int32, canChange bool,
	d *Decoder) (_ map[uint32]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[uint32]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]int64)
		if v, changed := fastpathTV.DecMapUint32Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Int64V(rv2i(rv).(map[uint32]int64), false, d)
}
func (f fastpathT) DecMapUint32Int64X(vp *map[uint32]int64, d *Decoder) {
	if v, changed := f.DecMapUint32Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Int64V(v map[uint32]int64, canChange bool,
	d *Decoder) (_ map[uint32]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]float32)
		if v, changed := fastpathTV.DecMapUint32Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Float32V(rv2i(rv).(map[uint32]float32), false, d)
}
func (f fastpathT) DecMapUint32Float32X(vp *map[uint32]float32, d *Decoder) {
	if v, changed := f.DecMapUint32Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Float32V(v map[uint32]float32, canChange bool,
	d *Decoder) (_ map[uint32]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[uint32]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]float64)
		if v, changed := fastpathTV.DecMapUint32Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32Float64V(rv2i(rv).(map[uint32]float64), false, d)
}
func (f fastpathT) DecMapUint32Float64X(vp *map[uint32]float64, d *Decoder) {
	if v, changed := f.DecMapUint32Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32Float64V(v map[uint32]float64, canChange bool,
	d *Decoder) (_ map[uint32]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint32]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint32BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint32]bool)
		if v, changed := fastpathTV.DecMapUint32BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint32BoolV(rv2i(rv).(map[uint32]bool), false, d)
}
func (f fastpathT) DecMapUint32BoolX(vp *map[uint32]bool, d *Decoder) {
	if v, changed := f.DecMapUint32BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint32BoolV(v map[uint32]bool, canChange bool,
	d *Decoder) (_ map[uint32]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[uint32]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint32
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uint32(dd.DecodeUint(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]interface{})
		if v, changed := fastpathTV.DecMapUint64IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64IntfV(rv2i(rv).(map[uint64]interface{}), false, d)
}
func (f fastpathT) DecMapUint64IntfX(vp *map[uint64]interface{}, d *Decoder) {
	if v, changed := f.DecMapUint64IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64IntfV(v map[uint64]interface{}, canChange bool,
	d *Decoder) (_ map[uint64]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint64]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk uint64
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]string)
		if v, changed := fastpathTV.DecMapUint64StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64StringV(rv2i(rv).(map[uint64]string), false, d)
}
func (f fastpathT) DecMapUint64StringX(vp *map[uint64]string, d *Decoder) {
	if v, changed := f.DecMapUint64StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64StringV(v map[uint64]string, canChange bool,
	d *Decoder) (_ map[uint64]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uint64]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]uint)
		if v, changed := fastpathTV.DecMapUint64UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64UintV(rv2i(rv).(map[uint64]uint), false, d)
}
func (f fastpathT) DecMapUint64UintX(vp *map[uint64]uint, d *Decoder) {
	if v, changed := f.DecMapUint64UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64UintV(v map[uint64]uint, canChange bool,
	d *Decoder) (_ map[uint64]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]uint8)
		if v, changed := fastpathTV.DecMapUint64Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Uint8V(rv2i(rv).(map[uint64]uint8), false, d)
}
func (f fastpathT) DecMapUint64Uint8X(vp *map[uint64]uint8, d *Decoder) {
	if v, changed := f.DecMapUint64Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint8V(v map[uint64]uint8, canChange bool,
	d *Decoder) (_ map[uint64]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint64]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]uint16)
		if v, changed := fastpathTV.DecMapUint64Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Uint16V(rv2i(rv).(map[uint64]uint16), false, d)
}
func (f fastpathT) DecMapUint64Uint16X(vp *map[uint64]uint16, d *Decoder) {
	if v, changed := f.DecMapUint64Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint16V(v map[uint64]uint16, canChange bool,
	d *Decoder) (_ map[uint64]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint64]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]uint32)
		if v, changed := fastpathTV.DecMapUint64Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Uint32V(rv2i(rv).(map[uint64]uint32), false, d)
}
func (f fastpathT) DecMapUint64Uint32X(vp *map[uint64]uint32, d *Decoder) {
	if v, changed := f.DecMapUint64Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint32V(v map[uint64]uint32, canChange bool,
	d *Decoder) (_ map[uint64]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint64]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]uint64)
		if v, changed := fastpathTV.DecMapUint64Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Uint64V(rv2i(rv).(map[uint64]uint64), false, d)
}
func (f fastpathT) DecMapUint64Uint64X(vp *map[uint64]uint64, d *Decoder) {
	if v, changed := f.DecMapUint64Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Uint64V(v map[uint64]uint64, canChange bool,
	d *Decoder) (_ map[uint64]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]uintptr)
		if v, changed := fastpathTV.DecMapUint64UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64UintptrV(rv2i(rv).(map[uint64]uintptr), false, d)
}
func (f fastpathT) DecMapUint64UintptrX(vp *map[uint64]uintptr, d *Decoder) {
	if v, changed := f.DecMapUint64UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64UintptrV(v map[uint64]uintptr, canChange bool,
	d *Decoder) (_ map[uint64]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]int)
		if v, changed := fastpathTV.DecMapUint64IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64IntV(rv2i(rv).(map[uint64]int), false, d)
}
func (f fastpathT) DecMapUint64IntX(vp *map[uint64]int, d *Decoder) {
	if v, changed := f.DecMapUint64IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64IntV(v map[uint64]int, canChange bool,
	d *Decoder) (_ map[uint64]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]int8)
		if v, changed := fastpathTV.DecMapUint64Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Int8V(rv2i(rv).(map[uint64]int8), false, d)
}
func (f fastpathT) DecMapUint64Int8X(vp *map[uint64]int8, d *Decoder) {
	if v, changed := f.DecMapUint64Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int8V(v map[uint64]int8, canChange bool,
	d *Decoder) (_ map[uint64]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint64]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]int16)
		if v, changed := fastpathTV.DecMapUint64Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Int16V(rv2i(rv).(map[uint64]int16), false, d)
}
func (f fastpathT) DecMapUint64Int16X(vp *map[uint64]int16, d *Decoder) {
	if v, changed := f.DecMapUint64Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int16V(v map[uint64]int16, canChange bool,
	d *Decoder) (_ map[uint64]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uint64]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]int32)
		if v, changed := fastpathTV.DecMapUint64Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Int32V(rv2i(rv).(map[uint64]int32), false, d)
}
func (f fastpathT) DecMapUint64Int32X(vp *map[uint64]int32, d *Decoder) {
	if v, changed := f.DecMapUint64Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int32V(v map[uint64]int32, canChange bool,
	d *Decoder) (_ map[uint64]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint64]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]int64)
		if v, changed := fastpathTV.DecMapUint64Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Int64V(rv2i(rv).(map[uint64]int64), false, d)
}
func (f fastpathT) DecMapUint64Int64X(vp *map[uint64]int64, d *Decoder) {
	if v, changed := f.DecMapUint64Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Int64V(v map[uint64]int64, canChange bool,
	d *Decoder) (_ map[uint64]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]float32)
		if v, changed := fastpathTV.DecMapUint64Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Float32V(rv2i(rv).(map[uint64]float32), false, d)
}
func (f fastpathT) DecMapUint64Float32X(vp *map[uint64]float32, d *Decoder) {
	if v, changed := f.DecMapUint64Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Float32V(v map[uint64]float32, canChange bool,
	d *Decoder) (_ map[uint64]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uint64]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]float64)
		if v, changed := fastpathTV.DecMapUint64Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64Float64V(rv2i(rv).(map[uint64]float64), false, d)
}
func (f fastpathT) DecMapUint64Float64X(vp *map[uint64]float64, d *Decoder) {
	if v, changed := f.DecMapUint64Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64Float64V(v map[uint64]float64, canChange bool,
	d *Decoder) (_ map[uint64]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uint64]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUint64BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uint64]bool)
		if v, changed := fastpathTV.DecMapUint64BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUint64BoolV(rv2i(rv).(map[uint64]bool), false, d)
}
func (f fastpathT) DecMapUint64BoolX(vp *map[uint64]bool, d *Decoder) {
	if v, changed := f.DecMapUint64BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUint64BoolV(v map[uint64]bool, canChange bool,
	d *Decoder) (_ map[uint64]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uint64]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uint64
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeUint(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrIntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]interface{})
		if v, changed := fastpathTV.DecMapUintptrIntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrIntfV(rv2i(rv).(map[uintptr]interface{}), false, d)
}
func (f fastpathT) DecMapUintptrIntfX(vp *map[uintptr]interface{}, d *Decoder) {
	if v, changed := f.DecMapUintptrIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrIntfV(v map[uintptr]interface{}, canChange bool,
	d *Decoder) (_ map[uintptr]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uintptr]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk uintptr
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrStringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]string)
		if v, changed := fastpathTV.DecMapUintptrStringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrStringV(rv2i(rv).(map[uintptr]string), false, d)
}
func (f fastpathT) DecMapUintptrStringX(vp *map[uintptr]string, d *Decoder) {
	if v, changed := f.DecMapUintptrStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrStringV(v map[uintptr]string, canChange bool,
	d *Decoder) (_ map[uintptr]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[uintptr]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrUintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]uint)
		if v, changed := fastpathTV.DecMapUintptrUintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrUintV(rv2i(rv).(map[uintptr]uint), false, d)
}
func (f fastpathT) DecMapUintptrUintX(vp *map[uintptr]uint, d *Decoder) {
	if v, changed := f.DecMapUintptrUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrUintV(v map[uintptr]uint, canChange bool,
	d *Decoder) (_ map[uintptr]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uintptr]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrUint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]uint8)
		if v, changed := fastpathTV.DecMapUintptrUint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrUint8V(rv2i(rv).(map[uintptr]uint8), false, d)
}
func (f fastpathT) DecMapUintptrUint8X(vp *map[uintptr]uint8, d *Decoder) {
	if v, changed := f.DecMapUintptrUint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrUint8V(v map[uintptr]uint8, canChange bool,
	d *Decoder) (_ map[uintptr]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uintptr]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrUint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]uint16)
		if v, changed := fastpathTV.DecMapUintptrUint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrUint16V(rv2i(rv).(map[uintptr]uint16), false, d)
}
func (f fastpathT) DecMapUintptrUint16X(vp *map[uintptr]uint16, d *Decoder) {
	if v, changed := f.DecMapUintptrUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrUint16V(v map[uintptr]uint16, canChange bool,
	d *Decoder) (_ map[uintptr]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uintptr]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrUint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]uint32)
		if v, changed := fastpathTV.DecMapUintptrUint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrUint32V(rv2i(rv).(map[uintptr]uint32), false, d)
}
func (f fastpathT) DecMapUintptrUint32X(vp *map[uintptr]uint32, d *Decoder) {
	if v, changed := f.DecMapUintptrUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrUint32V(v map[uintptr]uint32, canChange bool,
	d *Decoder) (_ map[uintptr]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uintptr]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrUint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]uint64)
		if v, changed := fastpathTV.DecMapUintptrUint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrUint64V(rv2i(rv).(map[uintptr]uint64), false, d)
}
func (f fastpathT) DecMapUintptrUint64X(vp *map[uintptr]uint64, d *Decoder) {
	if v, changed := f.DecMapUintptrUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrUint64V(v map[uintptr]uint64, canChange bool,
	d *Decoder) (_ map[uintptr]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uintptr]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrUintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]uintptr)
		if v, changed := fastpathTV.DecMapUintptrUintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrUintptrV(rv2i(rv).(map[uintptr]uintptr), false, d)
}
func (f fastpathT) DecMapUintptrUintptrX(vp *map[uintptr]uintptr, d *Decoder) {
	if v, changed := f.DecMapUintptrUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrUintptrV(v map[uintptr]uintptr, canChange bool,
	d *Decoder) (_ map[uintptr]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uintptr]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrIntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]int)
		if v, changed := fastpathTV.DecMapUintptrIntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrIntV(rv2i(rv).(map[uintptr]int), false, d)
}
func (f fastpathT) DecMapUintptrIntX(vp *map[uintptr]int, d *Decoder) {
	if v, changed := f.DecMapUintptrIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrIntV(v map[uintptr]int, canChange bool,
	d *Decoder) (_ map[uintptr]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uintptr]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrInt8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]int8)
		if v, changed := fastpathTV.DecMapUintptrInt8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrInt8V(rv2i(rv).(map[uintptr]int8), false, d)
}
func (f fastpathT) DecMapUintptrInt8X(vp *map[uintptr]int8, d *Decoder) {
	if v, changed := f.DecMapUintptrInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrInt8V(v map[uintptr]int8, canChange bool,
	d *Decoder) (_ map[uintptr]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uintptr]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrInt16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]int16)
		if v, changed := fastpathTV.DecMapUintptrInt16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrInt16V(rv2i(rv).(map[uintptr]int16), false, d)
}
func (f fastpathT) DecMapUintptrInt16X(vp *map[uintptr]int16, d *Decoder) {
	if v, changed := f.DecMapUintptrInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrInt16V(v map[uintptr]int16, canChange bool,
	d *Decoder) (_ map[uintptr]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[uintptr]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrInt32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]int32)
		if v, changed := fastpathTV.DecMapUintptrInt32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrInt32V(rv2i(rv).(map[uintptr]int32), false, d)
}
func (f fastpathT) DecMapUintptrInt32X(vp *map[uintptr]int32, d *Decoder) {
	if v, changed := f.DecMapUintptrInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrInt32V(v map[uintptr]int32, canChange bool,
	d *Decoder) (_ map[uintptr]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uintptr]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrInt64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]int64)
		if v, changed := fastpathTV.DecMapUintptrInt64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrInt64V(rv2i(rv).(map[uintptr]int64), false, d)
}
func (f fastpathT) DecMapUintptrInt64X(vp *map[uintptr]int64, d *Decoder) {
	if v, changed := f.DecMapUintptrInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrInt64V(v map[uintptr]int64, canChange bool,
	d *Decoder) (_ map[uintptr]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uintptr]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrFloat32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]float32)
		if v, changed := fastpathTV.DecMapUintptrFloat32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrFloat32V(rv2i(rv).(map[uintptr]float32), false, d)
}
func (f fastpathT) DecMapUintptrFloat32X(vp *map[uintptr]float32, d *Decoder) {
	if v, changed := f.DecMapUintptrFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrFloat32V(v map[uintptr]float32, canChange bool,
	d *Decoder) (_ map[uintptr]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[uintptr]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrFloat64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]float64)
		if v, changed := fastpathTV.DecMapUintptrFloat64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrFloat64V(rv2i(rv).(map[uintptr]float64), false, d)
}
func (f fastpathT) DecMapUintptrFloat64X(vp *map[uintptr]float64, d *Decoder) {
	if v, changed := f.DecMapUintptrFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrFloat64V(v map[uintptr]float64, canChange bool,
	d *Decoder) (_ map[uintptr]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[uintptr]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapUintptrBoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[uintptr]bool)
		if v, changed := fastpathTV.DecMapUintptrBoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapUintptrBoolV(rv2i(rv).(map[uintptr]bool), false, d)
}
func (f fastpathT) DecMapUintptrBoolX(vp *map[uintptr]bool, d *Decoder) {
	if v, changed := f.DecMapUintptrBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapUintptrBoolV(v map[uintptr]bool, canChange bool,
	d *Decoder) (_ map[uintptr]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[uintptr]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk uintptr
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = uintptr(dd.DecodeUint(uintBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntIntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]interface{})
		if v, changed := fastpathTV.DecMapIntIntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntIntfV(rv2i(rv).(map[int]interface{}), false, d)
}
func (f fastpathT) DecMapIntIntfX(vp *map[int]interface{}, d *Decoder) {
	if v, changed := f.DecMapIntIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntIntfV(v map[int]interface{}, canChange bool,
	d *Decoder) (_ map[int]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk int
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntStringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]string)
		if v, changed := fastpathTV.DecMapIntStringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntStringV(rv2i(rv).(map[int]string), false, d)
}
func (f fastpathT) DecMapIntStringX(vp *map[int]string, d *Decoder) {
	if v, changed := f.DecMapIntStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntStringV(v map[int]string, canChange bool,
	d *Decoder) (_ map[int]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntUintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]uint)
		if v, changed := fastpathTV.DecMapIntUintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntUintV(rv2i(rv).(map[int]uint), false, d)
}
func (f fastpathT) DecMapIntUintX(vp *map[int]uint, d *Decoder) {
	if v, changed := f.DecMapIntUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUintV(v map[int]uint, canChange bool,
	d *Decoder) (_ map[int]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntUint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]uint8)
		if v, changed := fastpathTV.DecMapIntUint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntUint8V(rv2i(rv).(map[int]uint8), false, d)
}
func (f fastpathT) DecMapIntUint8X(vp *map[int]uint8, d *Decoder) {
	if v, changed := f.DecMapIntUint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint8V(v map[int]uint8, canChange bool,
	d *Decoder) (_ map[int]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntUint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]uint16)
		if v, changed := fastpathTV.DecMapIntUint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntUint16V(rv2i(rv).(map[int]uint16), false, d)
}
func (f fastpathT) DecMapIntUint16X(vp *map[int]uint16, d *Decoder) {
	if v, changed := f.DecMapIntUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint16V(v map[int]uint16, canChange bool,
	d *Decoder) (_ map[int]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntUint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]uint32)
		if v, changed := fastpathTV.DecMapIntUint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntUint32V(rv2i(rv).(map[int]uint32), false, d)
}
func (f fastpathT) DecMapIntUint32X(vp *map[int]uint32, d *Decoder) {
	if v, changed := f.DecMapIntUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint32V(v map[int]uint32, canChange bool,
	d *Decoder) (_ map[int]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntUint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]uint64)
		if v, changed := fastpathTV.DecMapIntUint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntUint64V(rv2i(rv).(map[int]uint64), false, d)
}
func (f fastpathT) DecMapIntUint64X(vp *map[int]uint64, d *Decoder) {
	if v, changed := f.DecMapIntUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUint64V(v map[int]uint64, canChange bool,
	d *Decoder) (_ map[int]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntUintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]uintptr)
		if v, changed := fastpathTV.DecMapIntUintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntUintptrV(rv2i(rv).(map[int]uintptr), false, d)
}
func (f fastpathT) DecMapIntUintptrX(vp *map[int]uintptr, d *Decoder) {
	if v, changed := f.DecMapIntUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntUintptrV(v map[int]uintptr, canChange bool,
	d *Decoder) (_ map[int]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntIntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]int)
		if v, changed := fastpathTV.DecMapIntIntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntIntV(rv2i(rv).(map[int]int), false, d)
}
func (f fastpathT) DecMapIntIntX(vp *map[int]int, d *Decoder) {
	if v, changed := f.DecMapIntIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntIntV(v map[int]int, canChange bool,
	d *Decoder) (_ map[int]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntInt8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]int8)
		if v, changed := fastpathTV.DecMapIntInt8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntInt8V(rv2i(rv).(map[int]int8), false, d)
}
func (f fastpathT) DecMapIntInt8X(vp *map[int]int8, d *Decoder) {
	if v, changed := f.DecMapIntInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt8V(v map[int]int8, canChange bool,
	d *Decoder) (_ map[int]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntInt16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]int16)
		if v, changed := fastpathTV.DecMapIntInt16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntInt16V(rv2i(rv).(map[int]int16), false, d)
}
func (f fastpathT) DecMapIntInt16X(vp *map[int]int16, d *Decoder) {
	if v, changed := f.DecMapIntInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt16V(v map[int]int16, canChange bool,
	d *Decoder) (_ map[int]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntInt32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]int32)
		if v, changed := fastpathTV.DecMapIntInt32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntInt32V(rv2i(rv).(map[int]int32), false, d)
}
func (f fastpathT) DecMapIntInt32X(vp *map[int]int32, d *Decoder) {
	if v, changed := f.DecMapIntInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt32V(v map[int]int32, canChange bool,
	d *Decoder) (_ map[int]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntInt64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]int64)
		if v, changed := fastpathTV.DecMapIntInt64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntInt64V(rv2i(rv).(map[int]int64), false, d)
}
func (f fastpathT) DecMapIntInt64X(vp *map[int]int64, d *Decoder) {
	if v, changed := f.DecMapIntInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntInt64V(v map[int]int64, canChange bool,
	d *Decoder) (_ map[int]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntFloat32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]float32)
		if v, changed := fastpathTV.DecMapIntFloat32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntFloat32V(rv2i(rv).(map[int]float32), false, d)
}
func (f fastpathT) DecMapIntFloat32X(vp *map[int]float32, d *Decoder) {
	if v, changed := f.DecMapIntFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntFloat32V(v map[int]float32, canChange bool,
	d *Decoder) (_ map[int]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntFloat64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]float64)
		if v, changed := fastpathTV.DecMapIntFloat64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntFloat64V(rv2i(rv).(map[int]float64), false, d)
}
func (f fastpathT) DecMapIntFloat64X(vp *map[int]float64, d *Decoder) {
	if v, changed := f.DecMapIntFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntFloat64V(v map[int]float64, canChange bool,
	d *Decoder) (_ map[int]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapIntBoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int]bool)
		if v, changed := fastpathTV.DecMapIntBoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapIntBoolV(rv2i(rv).(map[int]bool), false, d)
}
func (f fastpathT) DecMapIntBoolX(vp *map[int]bool, d *Decoder) {
	if v, changed := f.DecMapIntBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapIntBoolV(v map[int]bool, canChange bool,
	d *Decoder) (_ map[int]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int(dd.DecodeInt(intBitsize))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]interface{})
		if v, changed := fastpathTV.DecMapInt8IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8IntfV(rv2i(rv).(map[int8]interface{}), false, d)
}
func (f fastpathT) DecMapInt8IntfX(vp *map[int8]interface{}, d *Decoder) {
	if v, changed := f.DecMapInt8IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8IntfV(v map[int8]interface{}, canChange bool,
	d *Decoder) (_ map[int8]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[int8]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk int8
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]string)
		if v, changed := fastpathTV.DecMapInt8StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8StringV(rv2i(rv).(map[int8]string), false, d)
}
func (f fastpathT) DecMapInt8StringX(vp *map[int8]string, d *Decoder) {
	if v, changed := f.DecMapInt8StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8StringV(v map[int8]string, canChange bool,
	d *Decoder) (_ map[int8]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[int8]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]uint)
		if v, changed := fastpathTV.DecMapInt8UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8UintV(rv2i(rv).(map[int8]uint), false, d)
}
func (f fastpathT) DecMapInt8UintX(vp *map[int8]uint, d *Decoder) {
	if v, changed := f.DecMapInt8UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8UintV(v map[int8]uint, canChange bool,
	d *Decoder) (_ map[int8]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]uint8)
		if v, changed := fastpathTV.DecMapInt8Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Uint8V(rv2i(rv).(map[int8]uint8), false, d)
}
func (f fastpathT) DecMapInt8Uint8X(vp *map[int8]uint8, d *Decoder) {
	if v, changed := f.DecMapInt8Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint8V(v map[int8]uint8, canChange bool,
	d *Decoder) (_ map[int8]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[int8]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]uint16)
		if v, changed := fastpathTV.DecMapInt8Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Uint16V(rv2i(rv).(map[int8]uint16), false, d)
}
func (f fastpathT) DecMapInt8Uint16X(vp *map[int8]uint16, d *Decoder) {
	if v, changed := f.DecMapInt8Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint16V(v map[int8]uint16, canChange bool,
	d *Decoder) (_ map[int8]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int8]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]uint32)
		if v, changed := fastpathTV.DecMapInt8Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Uint32V(rv2i(rv).(map[int8]uint32), false, d)
}
func (f fastpathT) DecMapInt8Uint32X(vp *map[int8]uint32, d *Decoder) {
	if v, changed := f.DecMapInt8Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint32V(v map[int8]uint32, canChange bool,
	d *Decoder) (_ map[int8]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int8]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]uint64)
		if v, changed := fastpathTV.DecMapInt8Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Uint64V(rv2i(rv).(map[int8]uint64), false, d)
}
func (f fastpathT) DecMapInt8Uint64X(vp *map[int8]uint64, d *Decoder) {
	if v, changed := f.DecMapInt8Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Uint64V(v map[int8]uint64, canChange bool,
	d *Decoder) (_ map[int8]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]uintptr)
		if v, changed := fastpathTV.DecMapInt8UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8UintptrV(rv2i(rv).(map[int8]uintptr), false, d)
}
func (f fastpathT) DecMapInt8UintptrX(vp *map[int8]uintptr, d *Decoder) {
	if v, changed := f.DecMapInt8UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8UintptrV(v map[int8]uintptr, canChange bool,
	d *Decoder) (_ map[int8]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]int)
		if v, changed := fastpathTV.DecMapInt8IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8IntV(rv2i(rv).(map[int8]int), false, d)
}
func (f fastpathT) DecMapInt8IntX(vp *map[int8]int, d *Decoder) {
	if v, changed := f.DecMapInt8IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8IntV(v map[int8]int, canChange bool,
	d *Decoder) (_ map[int8]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]int8)
		if v, changed := fastpathTV.DecMapInt8Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Int8V(rv2i(rv).(map[int8]int8), false, d)
}
func (f fastpathT) DecMapInt8Int8X(vp *map[int8]int8, d *Decoder) {
	if v, changed := f.DecMapInt8Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int8V(v map[int8]int8, canChange bool,
	d *Decoder) (_ map[int8]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[int8]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]int16)
		if v, changed := fastpathTV.DecMapInt8Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Int16V(rv2i(rv).(map[int8]int16), false, d)
}
func (f fastpathT) DecMapInt8Int16X(vp *map[int8]int16, d *Decoder) {
	if v, changed := f.DecMapInt8Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int16V(v map[int8]int16, canChange bool,
	d *Decoder) (_ map[int8]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int8]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]int32)
		if v, changed := fastpathTV.DecMapInt8Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Int32V(rv2i(rv).(map[int8]int32), false, d)
}
func (f fastpathT) DecMapInt8Int32X(vp *map[int8]int32, d *Decoder) {
	if v, changed := f.DecMapInt8Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int32V(v map[int8]int32, canChange bool,
	d *Decoder) (_ map[int8]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int8]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]int64)
		if v, changed := fastpathTV.DecMapInt8Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Int64V(rv2i(rv).(map[int8]int64), false, d)
}
func (f fastpathT) DecMapInt8Int64X(vp *map[int8]int64, d *Decoder) {
	if v, changed := f.DecMapInt8Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Int64V(v map[int8]int64, canChange bool,
	d *Decoder) (_ map[int8]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]float32)
		if v, changed := fastpathTV.DecMapInt8Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Float32V(rv2i(rv).(map[int8]float32), false, d)
}
func (f fastpathT) DecMapInt8Float32X(vp *map[int8]float32, d *Decoder) {
	if v, changed := f.DecMapInt8Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Float32V(v map[int8]float32, canChange bool,
	d *Decoder) (_ map[int8]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int8]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]float64)
		if v, changed := fastpathTV.DecMapInt8Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8Float64V(rv2i(rv).(map[int8]float64), false, d)
}
func (f fastpathT) DecMapInt8Float64X(vp *map[int8]float64, d *Decoder) {
	if v, changed := f.DecMapInt8Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8Float64V(v map[int8]float64, canChange bool,
	d *Decoder) (_ map[int8]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int8]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt8BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int8]bool)
		if v, changed := fastpathTV.DecMapInt8BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt8BoolV(rv2i(rv).(map[int8]bool), false, d)
}
func (f fastpathT) DecMapInt8BoolX(vp *map[int8]bool, d *Decoder) {
	if v, changed := f.DecMapInt8BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt8BoolV(v map[int8]bool, canChange bool,
	d *Decoder) (_ map[int8]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[int8]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int8
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int8(dd.DecodeInt(8))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]interface{})
		if v, changed := fastpathTV.DecMapInt16IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16IntfV(rv2i(rv).(map[int16]interface{}), false, d)
}
func (f fastpathT) DecMapInt16IntfX(vp *map[int16]interface{}, d *Decoder) {
	if v, changed := f.DecMapInt16IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16IntfV(v map[int16]interface{}, canChange bool,
	d *Decoder) (_ map[int16]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[int16]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk int16
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]string)
		if v, changed := fastpathTV.DecMapInt16StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16StringV(rv2i(rv).(map[int16]string), false, d)
}
func (f fastpathT) DecMapInt16StringX(vp *map[int16]string, d *Decoder) {
	if v, changed := f.DecMapInt16StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16StringV(v map[int16]string, canChange bool,
	d *Decoder) (_ map[int16]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 18)
		v = make(map[int16]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]uint)
		if v, changed := fastpathTV.DecMapInt16UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16UintV(rv2i(rv).(map[int16]uint), false, d)
}
func (f fastpathT) DecMapInt16UintX(vp *map[int16]uint, d *Decoder) {
	if v, changed := f.DecMapInt16UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16UintV(v map[int16]uint, canChange bool,
	d *Decoder) (_ map[int16]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]uint8)
		if v, changed := fastpathTV.DecMapInt16Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Uint8V(rv2i(rv).(map[int16]uint8), false, d)
}
func (f fastpathT) DecMapInt16Uint8X(vp *map[int16]uint8, d *Decoder) {
	if v, changed := f.DecMapInt16Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint8V(v map[int16]uint8, canChange bool,
	d *Decoder) (_ map[int16]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int16]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]uint16)
		if v, changed := fastpathTV.DecMapInt16Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Uint16V(rv2i(rv).(map[int16]uint16), false, d)
}
func (f fastpathT) DecMapInt16Uint16X(vp *map[int16]uint16, d *Decoder) {
	if v, changed := f.DecMapInt16Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint16V(v map[int16]uint16, canChange bool,
	d *Decoder) (_ map[int16]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[int16]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]uint32)
		if v, changed := fastpathTV.DecMapInt16Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Uint32V(rv2i(rv).(map[int16]uint32), false, d)
}
func (f fastpathT) DecMapInt16Uint32X(vp *map[int16]uint32, d *Decoder) {
	if v, changed := f.DecMapInt16Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint32V(v map[int16]uint32, canChange bool,
	d *Decoder) (_ map[int16]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int16]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]uint64)
		if v, changed := fastpathTV.DecMapInt16Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Uint64V(rv2i(rv).(map[int16]uint64), false, d)
}
func (f fastpathT) DecMapInt16Uint64X(vp *map[int16]uint64, d *Decoder) {
	if v, changed := f.DecMapInt16Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Uint64V(v map[int16]uint64, canChange bool,
	d *Decoder) (_ map[int16]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]uintptr)
		if v, changed := fastpathTV.DecMapInt16UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16UintptrV(rv2i(rv).(map[int16]uintptr), false, d)
}
func (f fastpathT) DecMapInt16UintptrX(vp *map[int16]uintptr, d *Decoder) {
	if v, changed := f.DecMapInt16UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16UintptrV(v map[int16]uintptr, canChange bool,
	d *Decoder) (_ map[int16]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]int)
		if v, changed := fastpathTV.DecMapInt16IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16IntV(rv2i(rv).(map[int16]int), false, d)
}
func (f fastpathT) DecMapInt16IntX(vp *map[int16]int, d *Decoder) {
	if v, changed := f.DecMapInt16IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16IntV(v map[int16]int, canChange bool,
	d *Decoder) (_ map[int16]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]int8)
		if v, changed := fastpathTV.DecMapInt16Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Int8V(rv2i(rv).(map[int16]int8), false, d)
}
func (f fastpathT) DecMapInt16Int8X(vp *map[int16]int8, d *Decoder) {
	if v, changed := f.DecMapInt16Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int8V(v map[int16]int8, canChange bool,
	d *Decoder) (_ map[int16]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int16]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]int16)
		if v, changed := fastpathTV.DecMapInt16Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Int16V(rv2i(rv).(map[int16]int16), false, d)
}
func (f fastpathT) DecMapInt16Int16X(vp *map[int16]int16, d *Decoder) {
	if v, changed := f.DecMapInt16Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int16V(v map[int16]int16, canChange bool,
	d *Decoder) (_ map[int16]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 4)
		v = make(map[int16]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]int32)
		if v, changed := fastpathTV.DecMapInt16Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Int32V(rv2i(rv).(map[int16]int32), false, d)
}
func (f fastpathT) DecMapInt16Int32X(vp *map[int16]int32, d *Decoder) {
	if v, changed := f.DecMapInt16Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int32V(v map[int16]int32, canChange bool,
	d *Decoder) (_ map[int16]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int16]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]int64)
		if v, changed := fastpathTV.DecMapInt16Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Int64V(rv2i(rv).(map[int16]int64), false, d)
}
func (f fastpathT) DecMapInt16Int64X(vp *map[int16]int64, d *Decoder) {
	if v, changed := f.DecMapInt16Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Int64V(v map[int16]int64, canChange bool,
	d *Decoder) (_ map[int16]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]float32)
		if v, changed := fastpathTV.DecMapInt16Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Float32V(rv2i(rv).(map[int16]float32), false, d)
}
func (f fastpathT) DecMapInt16Float32X(vp *map[int16]float32, d *Decoder) {
	if v, changed := f.DecMapInt16Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Float32V(v map[int16]float32, canChange bool,
	d *Decoder) (_ map[int16]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int16]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]float64)
		if v, changed := fastpathTV.DecMapInt16Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16Float64V(rv2i(rv).(map[int16]float64), false, d)
}
func (f fastpathT) DecMapInt16Float64X(vp *map[int16]float64, d *Decoder) {
	if v, changed := f.DecMapInt16Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16Float64V(v map[int16]float64, canChange bool,
	d *Decoder) (_ map[int16]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int16]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt16BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int16]bool)
		if v, changed := fastpathTV.DecMapInt16BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt16BoolV(rv2i(rv).(map[int16]bool), false, d)
}
func (f fastpathT) DecMapInt16BoolX(vp *map[int16]bool, d *Decoder) {
	if v, changed := f.DecMapInt16BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt16BoolV(v map[int16]bool, canChange bool,
	d *Decoder) (_ map[int16]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[int16]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int16
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int16(dd.DecodeInt(16))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]interface{})
		if v, changed := fastpathTV.DecMapInt32IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32IntfV(rv2i(rv).(map[int32]interface{}), false, d)
}
func (f fastpathT) DecMapInt32IntfX(vp *map[int32]interface{}, d *Decoder) {
	if v, changed := f.DecMapInt32IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32IntfV(v map[int32]interface{}, canChange bool,
	d *Decoder) (_ map[int32]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[int32]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk int32
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]string)
		if v, changed := fastpathTV.DecMapInt32StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32StringV(rv2i(rv).(map[int32]string), false, d)
}
func (f fastpathT) DecMapInt32StringX(vp *map[int32]string, d *Decoder) {
	if v, changed := f.DecMapInt32StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32StringV(v map[int32]string, canChange bool,
	d *Decoder) (_ map[int32]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 20)
		v = make(map[int32]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]uint)
		if v, changed := fastpathTV.DecMapInt32UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32UintV(rv2i(rv).(map[int32]uint), false, d)
}
func (f fastpathT) DecMapInt32UintX(vp *map[int32]uint, d *Decoder) {
	if v, changed := f.DecMapInt32UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32UintV(v map[int32]uint, canChange bool,
	d *Decoder) (_ map[int32]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]uint8)
		if v, changed := fastpathTV.DecMapInt32Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Uint8V(rv2i(rv).(map[int32]uint8), false, d)
}
func (f fastpathT) DecMapInt32Uint8X(vp *map[int32]uint8, d *Decoder) {
	if v, changed := f.DecMapInt32Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint8V(v map[int32]uint8, canChange bool,
	d *Decoder) (_ map[int32]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int32]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]uint16)
		if v, changed := fastpathTV.DecMapInt32Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Uint16V(rv2i(rv).(map[int32]uint16), false, d)
}
func (f fastpathT) DecMapInt32Uint16X(vp *map[int32]uint16, d *Decoder) {
	if v, changed := f.DecMapInt32Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint16V(v map[int32]uint16, canChange bool,
	d *Decoder) (_ map[int32]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int32]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]uint32)
		if v, changed := fastpathTV.DecMapInt32Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Uint32V(rv2i(rv).(map[int32]uint32), false, d)
}
func (f fastpathT) DecMapInt32Uint32X(vp *map[int32]uint32, d *Decoder) {
	if v, changed := f.DecMapInt32Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint32V(v map[int32]uint32, canChange bool,
	d *Decoder) (_ map[int32]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[int32]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]uint64)
		if v, changed := fastpathTV.DecMapInt32Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Uint64V(rv2i(rv).(map[int32]uint64), false, d)
}
func (f fastpathT) DecMapInt32Uint64X(vp *map[int32]uint64, d *Decoder) {
	if v, changed := f.DecMapInt32Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Uint64V(v map[int32]uint64, canChange bool,
	d *Decoder) (_ map[int32]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]uintptr)
		if v, changed := fastpathTV.DecMapInt32UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32UintptrV(rv2i(rv).(map[int32]uintptr), false, d)
}
func (f fastpathT) DecMapInt32UintptrX(vp *map[int32]uintptr, d *Decoder) {
	if v, changed := f.DecMapInt32UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32UintptrV(v map[int32]uintptr, canChange bool,
	d *Decoder) (_ map[int32]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]int)
		if v, changed := fastpathTV.DecMapInt32IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32IntV(rv2i(rv).(map[int32]int), false, d)
}
func (f fastpathT) DecMapInt32IntX(vp *map[int32]int, d *Decoder) {
	if v, changed := f.DecMapInt32IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32IntV(v map[int32]int, canChange bool,
	d *Decoder) (_ map[int32]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]int8)
		if v, changed := fastpathTV.DecMapInt32Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Int8V(rv2i(rv).(map[int32]int8), false, d)
}
func (f fastpathT) DecMapInt32Int8X(vp *map[int32]int8, d *Decoder) {
	if v, changed := f.DecMapInt32Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int8V(v map[int32]int8, canChange bool,
	d *Decoder) (_ map[int32]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int32]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]int16)
		if v, changed := fastpathTV.DecMapInt32Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Int16V(rv2i(rv).(map[int32]int16), false, d)
}
func (f fastpathT) DecMapInt32Int16X(vp *map[int32]int16, d *Decoder) {
	if v, changed := f.DecMapInt32Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int16V(v map[int32]int16, canChange bool,
	d *Decoder) (_ map[int32]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 6)
		v = make(map[int32]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]int32)
		if v, changed := fastpathTV.DecMapInt32Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Int32V(rv2i(rv).(map[int32]int32), false, d)
}
func (f fastpathT) DecMapInt32Int32X(vp *map[int32]int32, d *Decoder) {
	if v, changed := f.DecMapInt32Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int32V(v map[int32]int32, canChange bool,
	d *Decoder) (_ map[int32]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[int32]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]int64)
		if v, changed := fastpathTV.DecMapInt32Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Int64V(rv2i(rv).(map[int32]int64), false, d)
}
func (f fastpathT) DecMapInt32Int64X(vp *map[int32]int64, d *Decoder) {
	if v, changed := f.DecMapInt32Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Int64V(v map[int32]int64, canChange bool,
	d *Decoder) (_ map[int32]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]float32)
		if v, changed := fastpathTV.DecMapInt32Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Float32V(rv2i(rv).(map[int32]float32), false, d)
}
func (f fastpathT) DecMapInt32Float32X(vp *map[int32]float32, d *Decoder) {
	if v, changed := f.DecMapInt32Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Float32V(v map[int32]float32, canChange bool,
	d *Decoder) (_ map[int32]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 8)
		v = make(map[int32]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]float64)
		if v, changed := fastpathTV.DecMapInt32Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32Float64V(rv2i(rv).(map[int32]float64), false, d)
}
func (f fastpathT) DecMapInt32Float64X(vp *map[int32]float64, d *Decoder) {
	if v, changed := f.DecMapInt32Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32Float64V(v map[int32]float64, canChange bool,
	d *Decoder) (_ map[int32]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int32]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt32BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int32]bool)
		if v, changed := fastpathTV.DecMapInt32BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt32BoolV(rv2i(rv).(map[int32]bool), false, d)
}
func (f fastpathT) DecMapInt32BoolX(vp *map[int32]bool, d *Decoder) {
	if v, changed := f.DecMapInt32BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt32BoolV(v map[int32]bool, canChange bool,
	d *Decoder) (_ map[int32]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[int32]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int32
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = int32(dd.DecodeInt(32))
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64IntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]interface{})
		if v, changed := fastpathTV.DecMapInt64IntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64IntfV(rv2i(rv).(map[int64]interface{}), false, d)
}
func (f fastpathT) DecMapInt64IntfX(vp *map[int64]interface{}, d *Decoder) {
	if v, changed := f.DecMapInt64IntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64IntfV(v map[int64]interface{}, canChange bool,
	d *Decoder) (_ map[int64]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int64]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk int64
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64StringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]string)
		if v, changed := fastpathTV.DecMapInt64StringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64StringV(rv2i(rv).(map[int64]string), false, d)
}
func (f fastpathT) DecMapInt64StringX(vp *map[int64]string, d *Decoder) {
	if v, changed := f.DecMapInt64StringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64StringV(v map[int64]string, canChange bool,
	d *Decoder) (_ map[int64]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 24)
		v = make(map[int64]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64UintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]uint)
		if v, changed := fastpathTV.DecMapInt64UintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64UintV(rv2i(rv).(map[int64]uint), false, d)
}
func (f fastpathT) DecMapInt64UintX(vp *map[int64]uint, d *Decoder) {
	if v, changed := f.DecMapInt64UintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64UintV(v map[int64]uint, canChange bool,
	d *Decoder) (_ map[int64]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Uint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]uint8)
		if v, changed := fastpathTV.DecMapInt64Uint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Uint8V(rv2i(rv).(map[int64]uint8), false, d)
}
func (f fastpathT) DecMapInt64Uint8X(vp *map[int64]uint8, d *Decoder) {
	if v, changed := f.DecMapInt64Uint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint8V(v map[int64]uint8, canChange bool,
	d *Decoder) (_ map[int64]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int64]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Uint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]uint16)
		if v, changed := fastpathTV.DecMapInt64Uint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Uint16V(rv2i(rv).(map[int64]uint16), false, d)
}
func (f fastpathT) DecMapInt64Uint16X(vp *map[int64]uint16, d *Decoder) {
	if v, changed := f.DecMapInt64Uint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint16V(v map[int64]uint16, canChange bool,
	d *Decoder) (_ map[int64]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int64]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Uint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]uint32)
		if v, changed := fastpathTV.DecMapInt64Uint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Uint32V(rv2i(rv).(map[int64]uint32), false, d)
}
func (f fastpathT) DecMapInt64Uint32X(vp *map[int64]uint32, d *Decoder) {
	if v, changed := f.DecMapInt64Uint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint32V(v map[int64]uint32, canChange bool,
	d *Decoder) (_ map[int64]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int64]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Uint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]uint64)
		if v, changed := fastpathTV.DecMapInt64Uint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Uint64V(rv2i(rv).(map[int64]uint64), false, d)
}
func (f fastpathT) DecMapInt64Uint64X(vp *map[int64]uint64, d *Decoder) {
	if v, changed := f.DecMapInt64Uint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Uint64V(v map[int64]uint64, canChange bool,
	d *Decoder) (_ map[int64]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64UintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]uintptr)
		if v, changed := fastpathTV.DecMapInt64UintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64UintptrV(rv2i(rv).(map[int64]uintptr), false, d)
}
func (f fastpathT) DecMapInt64UintptrX(vp *map[int64]uintptr, d *Decoder) {
	if v, changed := f.DecMapInt64UintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64UintptrV(v map[int64]uintptr, canChange bool,
	d *Decoder) (_ map[int64]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64IntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]int)
		if v, changed := fastpathTV.DecMapInt64IntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64IntV(rv2i(rv).(map[int64]int), false, d)
}
func (f fastpathT) DecMapInt64IntX(vp *map[int64]int, d *Decoder) {
	if v, changed := f.DecMapInt64IntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64IntV(v map[int64]int, canChange bool,
	d *Decoder) (_ map[int64]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Int8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]int8)
		if v, changed := fastpathTV.DecMapInt64Int8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Int8V(rv2i(rv).(map[int64]int8), false, d)
}
func (f fastpathT) DecMapInt64Int8X(vp *map[int64]int8, d *Decoder) {
	if v, changed := f.DecMapInt64Int8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int8V(v map[int64]int8, canChange bool,
	d *Decoder) (_ map[int64]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int64]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Int16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]int16)
		if v, changed := fastpathTV.DecMapInt64Int16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Int16V(rv2i(rv).(map[int64]int16), false, d)
}
func (f fastpathT) DecMapInt64Int16X(vp *map[int64]int16, d *Decoder) {
	if v, changed := f.DecMapInt64Int16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int16V(v map[int64]int16, canChange bool,
	d *Decoder) (_ map[int64]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 10)
		v = make(map[int64]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Int32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]int32)
		if v, changed := fastpathTV.DecMapInt64Int32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Int32V(rv2i(rv).(map[int64]int32), false, d)
}
func (f fastpathT) DecMapInt64Int32X(vp *map[int64]int32, d *Decoder) {
	if v, changed := f.DecMapInt64Int32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int32V(v map[int64]int32, canChange bool,
	d *Decoder) (_ map[int64]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int64]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Int64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]int64)
		if v, changed := fastpathTV.DecMapInt64Int64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Int64V(rv2i(rv).(map[int64]int64), false, d)
}
func (f fastpathT) DecMapInt64Int64X(vp *map[int64]int64, d *Decoder) {
	if v, changed := f.DecMapInt64Int64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Int64V(v map[int64]int64, canChange bool,
	d *Decoder) (_ map[int64]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Float32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]float32)
		if v, changed := fastpathTV.DecMapInt64Float32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Float32V(rv2i(rv).(map[int64]float32), false, d)
}
func (f fastpathT) DecMapInt64Float32X(vp *map[int64]float32, d *Decoder) {
	if v, changed := f.DecMapInt64Float32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Float32V(v map[int64]float32, canChange bool,
	d *Decoder) (_ map[int64]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 12)
		v = make(map[int64]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64Float64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]float64)
		if v, changed := fastpathTV.DecMapInt64Float64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64Float64V(rv2i(rv).(map[int64]float64), false, d)
}
func (f fastpathT) DecMapInt64Float64X(vp *map[int64]float64, d *Decoder) {
	if v, changed := f.DecMapInt64Float64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64Float64V(v map[int64]float64, canChange bool,
	d *Decoder) (_ map[int64]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 16)
		v = make(map[int64]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapInt64BoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[int64]bool)
		if v, changed := fastpathTV.DecMapInt64BoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapInt64BoolV(rv2i(rv).(map[int64]bool), false, d)
}
func (f fastpathT) DecMapInt64BoolX(vp *map[int64]bool, d *Decoder) {
	if v, changed := f.DecMapInt64BoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapInt64BoolV(v map[int64]bool, canChange bool,
	d *Decoder) (_ map[int64]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[int64]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk int64
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeInt(64)
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolIntfR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]interface{})
		if v, changed := fastpathTV.DecMapBoolIntfV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolIntfV(rv2i(rv).(map[bool]interface{}), false, d)
}
func (f fastpathT) DecMapBoolIntfX(vp *map[bool]interface{}, d *Decoder) {
	if v, changed := f.DecMapBoolIntfV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolIntfV(v map[bool]interface{}, canChange bool,
	d *Decoder) (_ map[bool]interface{}, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[bool]interface{}, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}
	mapGet := !d.h.MapValueReset && !d.h.InterfaceReset
	var mk bool
	var mv interface{}
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = nil
			}
			continue
		}
		if mapGet {
			mv = v[mk]
		} else {
			mv = nil
		}
		d.decode(&mv)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolStringR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]string)
		if v, changed := fastpathTV.DecMapBoolStringV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolStringV(rv2i(rv).(map[bool]string), false, d)
}
func (f fastpathT) DecMapBoolStringX(vp *map[bool]string, d *Decoder) {
	if v, changed := f.DecMapBoolStringV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolStringV(v map[bool]string, canChange bool,
	d *Decoder) (_ map[bool]string, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 17)
		v = make(map[bool]string, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv string
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = ""
			}
			continue
		}
		mv = dd.DecodeString()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolUintR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]uint)
		if v, changed := fastpathTV.DecMapBoolUintV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolUintV(rv2i(rv).(map[bool]uint), false, d)
}
func (f fastpathT) DecMapBoolUintX(vp *map[bool]uint, d *Decoder) {
	if v, changed := f.DecMapBoolUintV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUintV(v map[bool]uint, canChange bool,
	d *Decoder) (_ map[bool]uint, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]uint, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv uint
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolUint8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]uint8)
		if v, changed := fastpathTV.DecMapBoolUint8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolUint8V(rv2i(rv).(map[bool]uint8), false, d)
}
func (f fastpathT) DecMapBoolUint8X(vp *map[bool]uint8, d *Decoder) {
	if v, changed := f.DecMapBoolUint8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint8V(v map[bool]uint8, canChange bool,
	d *Decoder) (_ map[bool]uint8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[bool]uint8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv uint8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint8(dd.DecodeUint(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolUint16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]uint16)
		if v, changed := fastpathTV.DecMapBoolUint16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolUint16V(rv2i(rv).(map[bool]uint16), false, d)
}
func (f fastpathT) DecMapBoolUint16X(vp *map[bool]uint16, d *Decoder) {
	if v, changed := f.DecMapBoolUint16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint16V(v map[bool]uint16, canChange bool,
	d *Decoder) (_ map[bool]uint16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[bool]uint16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv uint16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint16(dd.DecodeUint(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolUint32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]uint32)
		if v, changed := fastpathTV.DecMapBoolUint32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolUint32V(rv2i(rv).(map[bool]uint32), false, d)
}
func (f fastpathT) DecMapBoolUint32X(vp *map[bool]uint32, d *Decoder) {
	if v, changed := f.DecMapBoolUint32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint32V(v map[bool]uint32, canChange bool,
	d *Decoder) (_ map[bool]uint32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[bool]uint32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv uint32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uint32(dd.DecodeUint(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolUint64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]uint64)
		if v, changed := fastpathTV.DecMapBoolUint64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolUint64V(rv2i(rv).(map[bool]uint64), false, d)
}
func (f fastpathT) DecMapBoolUint64X(vp *map[bool]uint64, d *Decoder) {
	if v, changed := f.DecMapBoolUint64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUint64V(v map[bool]uint64, canChange bool,
	d *Decoder) (_ map[bool]uint64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]uint64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv uint64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeUint(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolUintptrR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]uintptr)
		if v, changed := fastpathTV.DecMapBoolUintptrV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolUintptrV(rv2i(rv).(map[bool]uintptr), false, d)
}
func (f fastpathT) DecMapBoolUintptrX(vp *map[bool]uintptr, d *Decoder) {
	if v, changed := f.DecMapBoolUintptrV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolUintptrV(v map[bool]uintptr, canChange bool,
	d *Decoder) (_ map[bool]uintptr, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]uintptr, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv uintptr
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = uintptr(dd.DecodeUint(uintBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolIntR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]int)
		if v, changed := fastpathTV.DecMapBoolIntV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolIntV(rv2i(rv).(map[bool]int), false, d)
}
func (f fastpathT) DecMapBoolIntX(vp *map[bool]int, d *Decoder) {
	if v, changed := f.DecMapBoolIntV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolIntV(v map[bool]int, canChange bool,
	d *Decoder) (_ map[bool]int, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]int, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv int
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int(dd.DecodeInt(intBitsize))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolInt8R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]int8)
		if v, changed := fastpathTV.DecMapBoolInt8V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolInt8V(rv2i(rv).(map[bool]int8), false, d)
}
func (f fastpathT) DecMapBoolInt8X(vp *map[bool]int8, d *Decoder) {
	if v, changed := f.DecMapBoolInt8V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt8V(v map[bool]int8, canChange bool,
	d *Decoder) (_ map[bool]int8, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[bool]int8, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv int8
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int8(dd.DecodeInt(8))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolInt16R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]int16)
		if v, changed := fastpathTV.DecMapBoolInt16V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolInt16V(rv2i(rv).(map[bool]int16), false, d)
}
func (f fastpathT) DecMapBoolInt16X(vp *map[bool]int16, d *Decoder) {
	if v, changed := f.DecMapBoolInt16V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt16V(v map[bool]int16, canChange bool,
	d *Decoder) (_ map[bool]int16, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 3)
		v = make(map[bool]int16, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv int16
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int16(dd.DecodeInt(16))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolInt32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]int32)
		if v, changed := fastpathTV.DecMapBoolInt32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolInt32V(rv2i(rv).(map[bool]int32), false, d)
}
func (f fastpathT) DecMapBoolInt32X(vp *map[bool]int32, d *Decoder) {
	if v, changed := f.DecMapBoolInt32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt32V(v map[bool]int32, canChange bool,
	d *Decoder) (_ map[bool]int32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[bool]int32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv int32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = int32(dd.DecodeInt(32))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolInt64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]int64)
		if v, changed := fastpathTV.DecMapBoolInt64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolInt64V(rv2i(rv).(map[bool]int64), false, d)
}
func (f fastpathT) DecMapBoolInt64X(vp *map[bool]int64, d *Decoder) {
	if v, changed := f.DecMapBoolInt64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolInt64V(v map[bool]int64, canChange bool,
	d *Decoder) (_ map[bool]int64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]int64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv int64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeInt(64)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolFloat32R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]float32)
		if v, changed := fastpathTV.DecMapBoolFloat32V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolFloat32V(rv2i(rv).(map[bool]float32), false, d)
}
func (f fastpathT) DecMapBoolFloat32X(vp *map[bool]float32, d *Decoder) {
	if v, changed := f.DecMapBoolFloat32V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolFloat32V(v map[bool]float32, canChange bool,
	d *Decoder) (_ map[bool]float32, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 5)
		v = make(map[bool]float32, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv float32
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = float32(dd.DecodeFloat(true))
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolFloat64R(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]float64)
		if v, changed := fastpathTV.DecMapBoolFloat64V(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolFloat64V(rv2i(rv).(map[bool]float64), false, d)
}
func (f fastpathT) DecMapBoolFloat64X(vp *map[bool]float64, d *Decoder) {
	if v, changed := f.DecMapBoolFloat64V(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolFloat64V(v map[bool]float64, canChange bool,
	d *Decoder) (_ map[bool]float64, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 9)
		v = make(map[bool]float64, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv float64
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = 0
			}
			continue
		}
		mv = dd.DecodeFloat(false)
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}

func (d *Decoder) fastpathDecMapBoolBoolR(f *codecFnInfo, rv reflect.Value) {
	if rv.Kind() == reflect.Ptr {
		vp := rv2i(rv).(*map[bool]bool)
		if v, changed := fastpathTV.DecMapBoolBoolV(*vp, true, d); changed {
			*vp = v
		}
		return
	}
	fastpathTV.DecMapBoolBoolV(rv2i(rv).(map[bool]bool), false, d)
}
func (f fastpathT) DecMapBoolBoolX(vp *map[bool]bool, d *Decoder) {
	if v, changed := f.DecMapBoolBoolV(*vp, true, d); changed {
		*vp = v
	}
}
func (_ fastpathT) DecMapBoolBoolV(v map[bool]bool, canChange bool,
	d *Decoder) (_ map[bool]bool, changed bool) {
	dd, esep := d.d, d.hh.hasElemSeparators()

	containerLen := dd.ReadMapStart()
	if canChange && v == nil {
		xlen := decInferLen(containerLen, d.h.MaxInitLen, 2)
		v = make(map[bool]bool, xlen)
		changed = true
	}
	if containerLen == 0 {
		dd.ReadMapEnd()
		return v, changed
	}

	var mk bool
	var mv bool
	hasLen := containerLen > 0
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if esep {
			dd.ReadMapElemKey()
		}
		mk = dd.DecodeBool()
		if esep {
			dd.ReadMapElemValue()
		}
		if dd.TryDecodeAsNil() {
			if d.h.DeleteOnNilMapValue {
				delete(v, mk)
			} else {
				v[mk] = false
			}
			continue
		}
		mv = dd.DecodeBool()
		if v != nil {
			v[mk] = mv
		}
	}
	dd.ReadMapEnd()
	return v, changed
}
