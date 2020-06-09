// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2016 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
)

// Merge merges the src message into dst.
// This assumes that dst and src of the same type and are non-nil.
func (a *InternalMessageInfo) Merge(dst, src Message) {
	mi := atomicLoadMergeInfo(&a.merge)
	if mi == nil {
		mi = getMergeInfo(reflect.TypeOf(dst).Elem())
		atomicStoreMergeInfo(&a.merge, mi)
	}
	mi.merge(toPointer(&dst), toPointer(&src))
}

type mergeInfo struct {
	typ reflect.Type

	initialized int32 // 0: only typ is valid, 1: everything is valid
	lock        sync.Mutex

	fields       []mergeFieldInfo
	unrecognized field // Offset of XXX_unrecognized
}

type mergeFieldInfo struct {
	field field // Offset of field, guaranteed to be valid

	// isPointer reports whether the value in the field is a pointer.
	// This is true for the following situations:
	//	* Pointer to struct
	//	* Pointer to basic type (proto2 only)
	//	* Slice (first value in slice header is a pointer)
	//	* String (first value in string header is a pointer)
	isPointer bool

	// basicWidth reports the width of the field assuming that it is directly
	// embedded in the struct (as is the case for basic types in proto3).
	// The possible values are:
	// 	0: invalid
	//	1: bool
	//	4: int32, uint32, float32
	//	8: int64, uint64, float64
	basicWidth int

	// Where dst and src are pointers to the types being merged.
	merge func(dst, src pointer)
}

var (
	mergeInfoMap  = map[reflect.Type]*mergeInfo{}
	mergeInfoLock sync.Mutex
)

func getMergeInfo(t reflect.Type) *mergeInfo {
	mergeInfoLock.Lock()
	defer mergeInfoLock.Unlock()
	mi := mergeInfoMap[t]
	if mi == nil {
		mi = &mergeInfo{typ: t}
		mergeInfoMap[t] = mi
	}
	return mi
}

// merge merges src into dst assuming they are both of type *mi.typ.
func (mi *mergeInfo) merge(dst, src pointer) {
	if dst.isNil() {
		panic("proto: nil destination")
	}
	if src.isNil() {
		return // Nothing to do.
	}

	if atomic.LoadInt32(&mi.initialized) == 0 {
		mi.computeMergeInfo()
	}

	for _, fi := range mi.fields {
		sfp := src.offset(fi.field)

		// As an optimization, we can avoid the merge function call cost
		// if we know for sure that the source will have no effect
		// by checking if it is the zero value.
		if unsafeAllowed {
			if fi.isPointer && sfp.getPointer().isNil() { // Could be slice or string
				continue
			}
			if fi.basicWidth > 0 {
				switch {
				case fi.basicWidth == 1 && !*sfp.toBool():
					continue
				case fi.basicWidth == 4 && *sfp.toUint32() == 0:
					continue
				case fi.basicWidth == 8 && *sfp.toUint64() == 0:
					continue
				}
			}
		}

		dfp := dst.offset(fi.field)
		fi.merge(dfp, sfp)
	}

	// TODO: Make this faster?
	out := dst.asPointerTo(mi.typ).Elem()
	in := src.asPointerTo(mi.typ).Elem()
	if emIn, err := extendable(in.Addr().Interface()); err == nil {
		emOut, _ := extendable(out.Addr().Interface())
		mIn, muIn := emIn.extensionsRead()
		if mIn != nil {
			mOut := emOut.extensionsWrite()
			muIn.Lock()
			mergeExtension(mOut, mIn)
			muIn.Unlock()
		}
	}

	if mi.unrecognized.IsValid() {
		if b := *src.offset(mi.unrecognized).toBytes(); len(b) > 0 {
			*dst.offset(mi.unrecognized).toBytes() = append([]byte(nil), b...)
		}
	}
}

func (mi *mergeInfo) computeMergeInfo() {
	mi.lock.Lock()
	defer mi.lock.Unlock()
	if mi.initialized != 0 {
		return
	}
	t := mi.typ
	n := t.NumField()

	props := GetProperties(t)
	for i := 0; i < n; i++ {
		f := t.Field(i)
		if strings.HasPrefix(f.Name, "XXX_") {
			continue
		}

		mfi := mergeFieldInfo{field: toField(&f)}
		tf := f.Type

		// As an optimization, we can avoid the merge function call cost
		// if we know for sure that the source will have no effect
		// by checking if it is the zero value.
		if unsafeAllowed {
			switch tf.Kind() {
			case reflect.Ptr, reflect.Slice, reflect.String:
				// As a special case, we assume slices and strings are pointers
				// since we know that the first field in the SliceSlice or
				// StringHeader is a data pointer.
				mfi.isPointer = true
			case reflect.Bool:
				mfi.basicWidth = 1
			case reflect.Int32, reflect.Uint32, reflect.Float32:
				mfi.basicWidth = 4
			case reflect.Int64, reflect.Uint64, reflect.Float64:
				mfi.basicWidth = 8
			}
		}

		// Unwrap tf to get at its most basic type.
		var isPointer, isSlice bool
		if tf.Kind() == reflect.Slice && tf.Elem().Kind() != reflect.Uint8 {
			isSlice = true
			tf = tf.Elem()
		}
		if tf.Kind() == reflect.Ptr {
			isPointer = true
			tf = tf.Elem()
		}
		if isPointer && isSlice && tf.Kind() != reflect.Struct {
			panic("both pointer and slice for basic type in " + tf.Name())
		}

		switch tf.Kind() {
		case reflect.Int32:
			switch {
			case isSlice: // E.g., []int32
				mfi.merge = func(dst, src pointer) {
					// NOTE: toInt32Slice is not defined (see pointer_reflect.go).
					/*
						sfsp := src.toInt32Slice()
						if *sfsp != nil {
							dfsp := dst.toInt32Slice()
							*dfsp = append(*dfsp, *sfsp...)
							if *dfsp == nil {
								*dfsp = []int64{}
							}
						}
					*/
					sfs := src.getInt32Slice()
					if sfs != nil {
						dfs := dst.getInt32Slice()
						dfs = append(dfs, sfs...)
						if dfs == nil {
							dfs = []int32{}
						}
						dst.setInt32Slice(dfs)
					}
				}
			case isPointer: // E.g., *int32
				mfi.merge = func(dst, src pointer) {
					// NOTE: toInt32Ptr is not defined (see pointer_reflect.go).
					/*
						sfpp := src.toInt32Ptr()
						if *sfpp != nil {
							dfpp := dst.toInt32Ptr()
							if *dfpp == nil {
								*dfpp = Int32(**sfpp)
							} else {
								**dfpp = **sfpp
							}
						}
					*/
					sfp := src.getInt32Ptr()
					if sfp != nil {
						dfp := dst.getInt32Ptr()
						if dfp == nil {
							dst.setInt32Ptr(*sfp)
						} else {
							*dfp = *sfp
						}
					}
				}
			default: // E.g., int32
				mfi.merge = func(dst, src pointer) {
					if v := *src.toInt32(); v != 0 {
						*dst.toInt32() = v
					}
				}
			}
		case reflect.Int64:
			switch {
			case isSlice: // E.g., []int64
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toInt64Slice()
					if *sfsp != nil {
						dfsp := dst.toInt64Slice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []int64{}
						}
					}
				}
			case isPointer: // E.g., *int64
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toInt64Ptr()
					if *sfpp != nil {
						dfpp := dst.toInt64Ptr()
						if *dfpp == nil {
							*dfpp = Int64(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., int64
				mfi.merge = func(dst, src pointer) {
					if v := *src.toInt64(); v != 0 {
						*dst.toInt64() = v
					}
				}
			}
		case reflect.Uint32:
			switch {
			case isSlice: // E.g., []uint32
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toUint32Slice()
					if *sfsp != nil {
						dfsp := dst.toUint32Slice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []uint32{}
						}
					}
				}
			case isPointer: // E.g., *uint32
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toUint32Ptr()
					if *sfpp != nil {
						dfpp := dst.toUint32Ptr()
						if *dfpp == nil {
							*dfpp = Uint32(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., uint32
				mfi.merge = func(dst, src pointer) {
					if v := *src.toUint32(); v != 0 {
						*dst.toUint32() = v
					}
				}
			}
		case reflect.Uint64:
			switch {
			case isSlice: // E.g., []uint64
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toUint64Slice()
					if *sfsp != nil {
						dfsp := dst.toUint64Slice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []uint64{}
						}
					}
				}
			case isPointer: // E.g., *uint64
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toUint64Ptr()
					if *sfpp != nil {
						dfpp := dst.toUint64Ptr()
						if *dfpp == nil {
							*dfpp = Uint64(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., uint64
				mfi.merge = func(dst, src pointer) {
					if v := *src.toUint64(); v != 0 {
						*dst.toUint64() = v
					}
				}
			}
		case reflect.Float32:
			switch {
			case isSlice: // E.g., []float32
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toFloat32Slice()
					if *sfsp != nil {
						dfsp := dst.toFloat32Slice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []float32{}
						}
					}
				}
			case isPointer: // E.g., *float32
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toFloat32Ptr()
					if *sfpp != nil {
						dfpp := dst.toFloat32Ptr()
						if *dfpp == nil {
							*dfpp = Float32(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., float32
				mfi.merge = func(dst, src pointer) {
					if v := *src.toFloat32(); v != 0 {
						*dst.toFloat32() = v
					}
				}
			}
		case reflect.Float64:
			switch {
			case isSlice: // E.g., []float64
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toFloat64Slice()
					if *sfsp != nil {
						dfsp := dst.toFloat64Slice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []float64{}
						}
					}
				}
			case isPointer: // E.g., *float64
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toFloat64Ptr()
					if *sfpp != nil {
						dfpp := dst.toFloat64Ptr()
						if *dfpp == nil {
							*dfpp = Float64(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., float64
				mfi.merge = func(dst, src pointer) {
					if v := *src.toFloat64(); v != 0 {
						*dst.toFloat64() = v
					}
				}
			}
		case reflect.Bool:
			switch {
			case isSlice: // E.g., []bool
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toBoolSlice()
					if *sfsp != nil {
						dfsp := dst.toBoolSlice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []bool{}
						}
					}
				}
			case isPointer: // E.g., *bool
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toBoolPtr()
					if *sfpp != nil {
						dfpp := dst.toBoolPtr()
						if *dfpp == nil {
							*dfpp = Bool(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., bool
				mfi.merge = func(dst, src pointer) {
					if v := *src.toBool(); v {
						*dst.toBool() = v
					}
				}
			}
		case reflect.String:
			switch {
			case isSlice: // E.g., []string
				mfi.merge = func(dst, src pointer) {
					sfsp := src.toStringSlice()
					if *sfsp != nil {
						dfsp := dst.toStringSlice()
						*dfsp = append(*dfsp, *sfsp...)
						if *dfsp == nil {
							*dfsp = []string{}
						}
					}
				}
			case isPointer: // E.g., *string
				mfi.merge = func(dst, src pointer) {
					sfpp := src.toStringPtr()
					if *sfpp != nil {
						dfpp := dst.toStringPtr()
						if *dfpp == nil {
							*dfpp = String(**sfpp)
						} else {
							**dfpp = **sfpp
						}
					}
				}
			default: // E.g., string
				mfi.merge = func(dst, src pointer) {
					if v := *src.toString(); v != "" {
						*dst.toString() = v
					}
				}
			}
		case reflect.Slice:
			isProto3 := props.Prop[i].proto3
			switch {
			case isPointer:
				panic("bad pointer in byte slice case in " + tf.Name())
			case tf.Elem().Kind() != reflect.Uint8:
				panic("bad element kind in byte slice case in " + tf.Name())
			case isSlice: // E.g., [][]byte
				mfi.merge = func(dst, src pointer) {
					sbsp := src.toBytesSlice()
					if *sbsp != nil {
						dbsp := dst.toBytesSlice()
						for _, sb := range *sbsp {
							if sb == nil {
								*dbsp = append(*dbsp, nil)
							} else {
								*dbsp = append(*dbsp, append([]byte{}, sb...))
							}
						}
						if *dbsp == nil {
							*dbsp = [][]byte{}
						}
					}
				}
			default: // E.g., []byte
				mfi.merge = func(dst, src pointer) {
					sbp := src.toBytes()
					if *sbp != nil {
						dbp := dst.toBytes()
						if !isProto3 || len(*sbp) > 0 {
							*dbp = append([]byte{}, *sbp...)
						}
					}
				}
			}
		case reflect.Struct:
			switch {
			case isSlice && !isPointer: // E.g. []pb.T
				mergeInfo := getMergeInfo(tf)
				zero := reflect.Zero(tf)
				mfi.merge = func(dst, src pointer) {
					// TODO: Make this faster?
					dstsp := dst.asPointerTo(f.Type)
					dsts := dstsp.Elem()
					srcs := src.asPointerTo(f.Type).Elem()
					for i := 0; i < srcs.Len(); i++ {
						dsts = reflect.Append(dsts, zero)
						srcElement := srcs.Index(i).Addr()
						dstElement := dsts.Index(dsts.Len() - 1).Addr()
						mergeInfo.merge(valToPointer(dstElement), valToPointer(srcElement))
					}
					if dsts.IsNil() {
						dsts = reflect.MakeSlice(f.Type, 0, 0)
					}
					dstsp.Elem().Set(dsts)
				}
			case !isPointer:
				mergeInfo := getMergeInfo(tf)
				mfi.merge = func(dst, src pointer) {
					mergeInfo.merge(dst, src)
				}
			case isSlice: // E.g., []*pb.T
				mergeInfo := getMergeInfo(tf)
				mfi.merge = func(dst, src pointer) {
					sps := src.getPointerSlice()
					if sps != nil {
						dps := dst.getPointerSlice()
						for _, sp := range sps {
							var dp pointer
							if !sp.isNil() {
								dp = valToPointer(reflect.New(tf))
								mergeInfo.merge(dp, sp)
							}
							dps = append(dps, dp)
						}
						if dps == nil {
							dps = []pointer{}
						}
						dst.setPointerSlice(dps)
					}
				}
			default: // E.g., *pb.T
				mergeInfo := getMergeInfo(tf)
				mfi.merge = func(dst, src pointer) {
					sp := src.getPointer()
					if !sp.isNil() {
						dp := dst.getPointer()
						if dp.isNil() {
							dp = valToPointer(reflect.New(tf))
							dst.setPointer(dp)
						}
						mergeInfo.merge(dp, sp)
					}
				}
			}
		case reflect.Map:
			switch {
			case isPointer || isSlice:
				panic("bad pointer or slice in map case in " + tf.Name())
			default: // E.g., map[K]V
				mfi.merge = func(dst, src pointer) {
					sm := src.asPointerTo(tf).Elem()
					if sm.Len() == 0 {
						return
					}
					dm := dst.asPointerTo(tf).Elem()
					if dm.IsNil() {
						dm.Set(reflect.MakeMap(tf))
					}

					switch tf.Elem().Kind() {
					case reflect.Ptr: // Proto struct (e.g., *T)
						for _, key := range sm.MapKeys() {
							val := sm.MapIndex(key)
							val = reflect.ValueOf(Clone(val.Interface().(Message)))
							dm.SetMapIndex(key, val)
						}
					case reflect.Slice: // E.g. Bytes type (e.g., []byte)
						for _, key := range sm.MapKeys() {
							val := sm.MapIndex(key)
							val = reflect.ValueOf(append([]byte{}, val.Bytes()...))
							dm.SetMapIndex(key, val)
						}
					default: // Basic type (e.g., string)
						for _, key := range sm.MapKeys() {
							val := sm.MapIndex(key)
							dm.SetMapIndex(key, val)
						}
					}
				}
			}
		case reflect.Interface:
			// Must be oneof field.
			switch {
			case isPointer || isSlice:
				panic("bad pointer or slice in interface case in " + tf.Name())
			default: // E.g., interface{}
				// TODO: Make this faster?
				mfi.merge = func(dst, src pointer) {
					su := src.asPointerTo(tf).Elem()
					if !su.IsNil() {
						du := dst.asPointerTo(tf).Elem()
						typ := su.Elem().Type()
						if du.IsNil() || du.Elem().Type() != typ {
							du.Set(reflect.New(typ.Elem())) // Initialize interface if empty
						}
						sv := su.Elem().Elem().Field(0)
						if sv.Kind() == reflect.Ptr && sv.IsNil() {
							return
						}
						dv := du.Elem().Elem().Field(0)
						if dv.Kind() == reflect.Ptr && dv.IsNil() {
							dv.Set(reflect.New(sv.Type().Elem())) // Initialize proto message if empty
						}
						switch sv.Type().Kind() {
						case reflect.Ptr: // Proto struct (e.g., *T)
							Merge(dv.Interface().(Message), sv.Interface().(Message))
						case reflect.Slice: // E.g. Bytes type (e.g., []byte)
							dv.Set(reflect.ValueOf(append([]byte{}, sv.Bytes()...)))
						default: // Basic type (e.g., string)
							dv.Set(sv)
						}
					}
				}
			}
		default:
			panic(fmt.Sprintf("merger not found for type:%s", tf))
		}
		mi.fields = append(mi.fields, mfi)
	}

	mi.unrecognized = invalidField
	if f, ok := t.FieldByName("XXX_unrecognized"); ok {
		if f.Type != reflect.TypeOf([]byte{}) {
			panic("expected XXX_unrecognized to be of type []byte")
		}
		mi.unrecognized = toField(&f)
	}

	atomic.StoreInt32(&mi.initialized, 1)
}
