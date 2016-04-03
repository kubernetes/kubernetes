// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

// Contains code shared by both encode and decode.

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	structTagName = "codec"

	// Support
	//    encoding.BinaryMarshaler: MarshalBinary() (data []byte, err error)
	//    encoding.BinaryUnmarshaler: UnmarshalBinary(data []byte) error
	// This constant flag will enable or disable it.
	supportBinaryMarshal = true

	// Each Encoder or Decoder uses a cache of functions based on conditionals,
	// so that the conditionals are not run every time.
	//
	// Either a map or a slice is used to keep track of the functions.
	// The map is more natural, but has a higher cost than a slice/array.
	// This flag (useMapForCodecCache) controls which is used.
	useMapForCodecCache = false

	// For some common container types, we can short-circuit an elaborate
	// reflection dance and call encode/decode directly.
	// The currently supported types are:
	//    - slices of strings, or id's (int64,uint64) or interfaces.
	//    - maps of str->str, str->intf, id(int64,uint64)->intf, intf->intf
	shortCircuitReflectToFastPath = true

	// for debugging, set this to false, to catch panic traces.
	// Note that this will always cause rpc tests to fail, since they need io.EOF sent via panic.
	recoverPanicToErr = true
)

type charEncoding uint8

const (
	c_RAW charEncoding = iota
	c_UTF8
	c_UTF16LE
	c_UTF16BE
	c_UTF32LE
	c_UTF32BE
)

// valueType is the stream type
type valueType uint8

const (
	valueTypeUnset valueType = iota
	valueTypeNil
	valueTypeInt
	valueTypeUint
	valueTypeFloat
	valueTypeBool
	valueTypeString
	valueTypeSymbol
	valueTypeBytes
	valueTypeMap
	valueTypeArray
	valueTypeTimestamp
	valueTypeExt

	valueTypeInvalid = 0xff
)

var (
	bigen               = binary.BigEndian
	structInfoFieldName = "_struct"

	cachedTypeInfo      = make(map[uintptr]*typeInfo, 4)
	cachedTypeInfoMutex sync.RWMutex

	intfSliceTyp = reflect.TypeOf([]interface{}(nil))
	intfTyp      = intfSliceTyp.Elem()

	strSliceTyp     = reflect.TypeOf([]string(nil))
	boolSliceTyp    = reflect.TypeOf([]bool(nil))
	uintSliceTyp    = reflect.TypeOf([]uint(nil))
	uint8SliceTyp   = reflect.TypeOf([]uint8(nil))
	uint16SliceTyp  = reflect.TypeOf([]uint16(nil))
	uint32SliceTyp  = reflect.TypeOf([]uint32(nil))
	uint64SliceTyp  = reflect.TypeOf([]uint64(nil))
	intSliceTyp     = reflect.TypeOf([]int(nil))
	int8SliceTyp    = reflect.TypeOf([]int8(nil))
	int16SliceTyp   = reflect.TypeOf([]int16(nil))
	int32SliceTyp   = reflect.TypeOf([]int32(nil))
	int64SliceTyp   = reflect.TypeOf([]int64(nil))
	float32SliceTyp = reflect.TypeOf([]float32(nil))
	float64SliceTyp = reflect.TypeOf([]float64(nil))

	mapIntfIntfTyp = reflect.TypeOf(map[interface{}]interface{}(nil))
	mapStrIntfTyp  = reflect.TypeOf(map[string]interface{}(nil))
	mapStrStrTyp   = reflect.TypeOf(map[string]string(nil))

	mapIntIntfTyp    = reflect.TypeOf(map[int]interface{}(nil))
	mapInt64IntfTyp  = reflect.TypeOf(map[int64]interface{}(nil))
	mapUintIntfTyp   = reflect.TypeOf(map[uint]interface{}(nil))
	mapUint64IntfTyp = reflect.TypeOf(map[uint64]interface{}(nil))

	stringTyp = reflect.TypeOf("")
	timeTyp   = reflect.TypeOf(time.Time{})
	rawExtTyp = reflect.TypeOf(RawExt{})

	mapBySliceTyp        = reflect.TypeOf((*MapBySlice)(nil)).Elem()
	binaryMarshalerTyp   = reflect.TypeOf((*binaryMarshaler)(nil)).Elem()
	binaryUnmarshalerTyp = reflect.TypeOf((*binaryUnmarshaler)(nil)).Elem()

	rawExtTypId = reflect.ValueOf(rawExtTyp).Pointer()
	intfTypId   = reflect.ValueOf(intfTyp).Pointer()
	timeTypId   = reflect.ValueOf(timeTyp).Pointer()

	intfSliceTypId = reflect.ValueOf(intfSliceTyp).Pointer()
	strSliceTypId  = reflect.ValueOf(strSliceTyp).Pointer()

	boolSliceTypId    = reflect.ValueOf(boolSliceTyp).Pointer()
	uintSliceTypId    = reflect.ValueOf(uintSliceTyp).Pointer()
	uint8SliceTypId   = reflect.ValueOf(uint8SliceTyp).Pointer()
	uint16SliceTypId  = reflect.ValueOf(uint16SliceTyp).Pointer()
	uint32SliceTypId  = reflect.ValueOf(uint32SliceTyp).Pointer()
	uint64SliceTypId  = reflect.ValueOf(uint64SliceTyp).Pointer()
	intSliceTypId     = reflect.ValueOf(intSliceTyp).Pointer()
	int8SliceTypId    = reflect.ValueOf(int8SliceTyp).Pointer()
	int16SliceTypId   = reflect.ValueOf(int16SliceTyp).Pointer()
	int32SliceTypId   = reflect.ValueOf(int32SliceTyp).Pointer()
	int64SliceTypId   = reflect.ValueOf(int64SliceTyp).Pointer()
	float32SliceTypId = reflect.ValueOf(float32SliceTyp).Pointer()
	float64SliceTypId = reflect.ValueOf(float64SliceTyp).Pointer()

	mapStrStrTypId     = reflect.ValueOf(mapStrStrTyp).Pointer()
	mapIntfIntfTypId   = reflect.ValueOf(mapIntfIntfTyp).Pointer()
	mapStrIntfTypId    = reflect.ValueOf(mapStrIntfTyp).Pointer()
	mapIntIntfTypId    = reflect.ValueOf(mapIntIntfTyp).Pointer()
	mapInt64IntfTypId  = reflect.ValueOf(mapInt64IntfTyp).Pointer()
	mapUintIntfTypId   = reflect.ValueOf(mapUintIntfTyp).Pointer()
	mapUint64IntfTypId = reflect.ValueOf(mapUint64IntfTyp).Pointer()
	// Id = reflect.ValueOf().Pointer()
	// mapBySliceTypId  = reflect.ValueOf(mapBySliceTyp).Pointer()

	binaryMarshalerTypId   = reflect.ValueOf(binaryMarshalerTyp).Pointer()
	binaryUnmarshalerTypId = reflect.ValueOf(binaryUnmarshalerTyp).Pointer()

	intBitsize  uint8 = uint8(reflect.TypeOf(int(0)).Bits())
	uintBitsize uint8 = uint8(reflect.TypeOf(uint(0)).Bits())

	bsAll0x00 = []byte{0, 0, 0, 0, 0, 0, 0, 0}
	bsAll0xff = []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
)

type binaryUnmarshaler interface {
	UnmarshalBinary(data []byte) error
}

type binaryMarshaler interface {
	MarshalBinary() (data []byte, err error)
}

// MapBySlice represents a slice which should be encoded as a map in the stream.
// The slice contains a sequence of key-value pairs.
type MapBySlice interface {
	MapBySlice()
}

// WARNING: DO NOT USE DIRECTLY. EXPORTED FOR GODOC BENEFIT. WILL BE REMOVED.
//
// BasicHandle encapsulates the common options and extension functions.
type BasicHandle struct {
	extHandle
	EncodeOptions
	DecodeOptions
}

// Handle is the interface for a specific encoding format.
//
// Typically, a Handle is pre-configured before first time use,
// and not modified while in use. Such a pre-configured Handle
// is safe for concurrent access.
type Handle interface {
	writeExt() bool
	getBasicHandle() *BasicHandle
	newEncDriver(w encWriter) encDriver
	newDecDriver(r decReader) decDriver
}

// RawExt represents raw unprocessed extension data.
type RawExt struct {
	Tag  byte
	Data []byte
}

type extTypeTagFn struct {
	rtid  uintptr
	rt    reflect.Type
	tag   byte
	encFn func(reflect.Value) ([]byte, error)
	decFn func(reflect.Value, []byte) error
}

type extHandle []*extTypeTagFn

// AddExt registers an encode and decode function for a reflect.Type.
// Note that the type must be a named type, and specifically not
// a pointer or Interface. An error is returned if that is not honored.
//
// To Deregister an ext, call AddExt with 0 tag, nil encfn and nil decfn.
func (o *extHandle) AddExt(
	rt reflect.Type,
	tag byte,
	encfn func(reflect.Value) ([]byte, error),
	decfn func(reflect.Value, []byte) error,
) (err error) {
	// o is a pointer, because we may need to initialize it
	if rt.PkgPath() == "" || rt.Kind() == reflect.Interface {
		err = fmt.Errorf("codec.Handle.AddExt: Takes named type, especially not a pointer or interface: %T",
			reflect.Zero(rt).Interface())
		return
	}

	// o cannot be nil, since it is always embedded in a Handle.
	// if nil, let it panic.
	// if o == nil {
	// 	err = errors.New("codec.Handle.AddExt: extHandle cannot be a nil pointer.")
	// 	return
	// }

	rtid := reflect.ValueOf(rt).Pointer()
	for _, v := range *o {
		if v.rtid == rtid {
			v.tag, v.encFn, v.decFn = tag, encfn, decfn
			return
		}
	}

	*o = append(*o, &extTypeTagFn{rtid, rt, tag, encfn, decfn})
	return
}

func (o extHandle) getExt(rtid uintptr) *extTypeTagFn {
	for _, v := range o {
		if v.rtid == rtid {
			return v
		}
	}
	return nil
}

func (o extHandle) getExtForTag(tag byte) *extTypeTagFn {
	for _, v := range o {
		if v.tag == tag {
			return v
		}
	}
	return nil
}

func (o extHandle) getDecodeExtForTag(tag byte) (
	rv reflect.Value, fn func(reflect.Value, []byte) error) {
	if x := o.getExtForTag(tag); x != nil {
		// ext is only registered for base
		rv = reflect.New(x.rt).Elem()
		fn = x.decFn
	}
	return
}

func (o extHandle) getDecodeExt(rtid uintptr) (tag byte, fn func(reflect.Value, []byte) error) {
	if x := o.getExt(rtid); x != nil {
		tag = x.tag
		fn = x.decFn
	}
	return
}

func (o extHandle) getEncodeExt(rtid uintptr) (tag byte, fn func(reflect.Value) ([]byte, error)) {
	if x := o.getExt(rtid); x != nil {
		tag = x.tag
		fn = x.encFn
	}
	return
}

type structFieldInfo struct {
	encName string // encode name

	// only one of 'i' or 'is' can be set. If 'i' is -1, then 'is' has been set.

	is        []int // (recursive/embedded) field index in struct
	i         int16 // field index in struct
	omitEmpty bool
	toArray   bool // if field is _struct, is the toArray set?

	// tag       string   // tag
	// name      string   // field name
	// encNameBs []byte   // encoded name as byte stream
	// ikind     int      // kind of the field as an int i.e. int(reflect.Kind)
}

func parseStructFieldInfo(fname string, stag string) *structFieldInfo {
	if fname == "" {
		panic("parseStructFieldInfo: No Field Name")
	}
	si := structFieldInfo{
		// name: fname,
		encName: fname,
		// tag: stag,
	}

	if stag != "" {
		for i, s := range strings.Split(stag, ",") {
			if i == 0 {
				if s != "" {
					si.encName = s
				}
			} else {
				switch s {
				case "omitempty":
					si.omitEmpty = true
				case "toarray":
					si.toArray = true
				}
			}
		}
	}
	// si.encNameBs = []byte(si.encName)
	return &si
}

type sfiSortedByEncName []*structFieldInfo

func (p sfiSortedByEncName) Len() int {
	return len(p)
}

func (p sfiSortedByEncName) Less(i, j int) bool {
	return p[i].encName < p[j].encName
}

func (p sfiSortedByEncName) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// typeInfo keeps information about each type referenced in the encode/decode sequence.
//
// During an encode/decode sequence, we work as below:
//   - If base is a built in type, en/decode base value
//   - If base is registered as an extension, en/decode base value
//   - If type is binary(M/Unm)arshaler, call Binary(M/Unm)arshal method
//   - Else decode appropriately based on the reflect.Kind
type typeInfo struct {
	sfi  []*structFieldInfo // sorted. Used when enc/dec struct to map.
	sfip []*structFieldInfo // unsorted. Used when enc/dec struct to array.

	rt   reflect.Type
	rtid uintptr

	// baseId gives pointer to the base reflect.Type, after deferencing
	// the pointers. E.g. base type of ***time.Time is time.Time.
	base      reflect.Type
	baseId    uintptr
	baseIndir int8 // number of indirections to get to base

	mbs bool // base type (T or *T) is a MapBySlice

	m        bool // base type (T or *T) is a binaryMarshaler
	unm      bool // base type (T or *T) is a binaryUnmarshaler
	mIndir   int8 // number of indirections to get to binaryMarshaler type
	unmIndir int8 // number of indirections to get to binaryUnmarshaler type
	toArray  bool // whether this (struct) type should be encoded as an array
}

func (ti *typeInfo) indexForEncName(name string) int {
	//tisfi := ti.sfi
	const binarySearchThreshold = 16
	if sfilen := len(ti.sfi); sfilen < binarySearchThreshold {
		// linear search. faster than binary search in my testing up to 16-field structs.
		for i, si := range ti.sfi {
			if si.encName == name {
				return i
			}
		}
	} else {
		// binary search. adapted from sort/search.go.
		h, i, j := 0, 0, sfilen
		for i < j {
			h = i + (j-i)/2
			if ti.sfi[h].encName < name {
				i = h + 1
			} else {
				j = h
			}
		}
		if i < sfilen && ti.sfi[i].encName == name {
			return i
		}
	}
	return -1
}

func getTypeInfo(rtid uintptr, rt reflect.Type) (pti *typeInfo) {
	var ok bool
	cachedTypeInfoMutex.RLock()
	pti, ok = cachedTypeInfo[rtid]
	cachedTypeInfoMutex.RUnlock()
	if ok {
		return
	}

	cachedTypeInfoMutex.Lock()
	defer cachedTypeInfoMutex.Unlock()
	if pti, ok = cachedTypeInfo[rtid]; ok {
		return
	}

	ti := typeInfo{rt: rt, rtid: rtid}
	pti = &ti

	var indir int8
	if ok, indir = implementsIntf(rt, binaryMarshalerTyp); ok {
		ti.m, ti.mIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, binaryUnmarshalerTyp); ok {
		ti.unm, ti.unmIndir = true, indir
	}
	if ok, _ = implementsIntf(rt, mapBySliceTyp); ok {
		ti.mbs = true
	}

	pt := rt
	var ptIndir int8
	// for ; pt.Kind() == reflect.Ptr; pt, ptIndir = pt.Elem(), ptIndir+1 { }
	for pt.Kind() == reflect.Ptr {
		pt = pt.Elem()
		ptIndir++
	}
	if ptIndir == 0 {
		ti.base = rt
		ti.baseId = rtid
	} else {
		ti.base = pt
		ti.baseId = reflect.ValueOf(pt).Pointer()
		ti.baseIndir = ptIndir
	}

	if rt.Kind() == reflect.Struct {
		var siInfo *structFieldInfo
		if f, ok := rt.FieldByName(structInfoFieldName); ok {
			siInfo = parseStructFieldInfo(structInfoFieldName, f.Tag.Get(structTagName))
			ti.toArray = siInfo.toArray
		}
		sfip := make([]*structFieldInfo, 0, rt.NumField())
		rgetTypeInfo(rt, nil, make(map[string]bool), &sfip, siInfo)

		// // try to put all si close together
		// const tryToPutAllStructFieldInfoTogether = true
		// if tryToPutAllStructFieldInfoTogether {
		// 	sfip2 := make([]structFieldInfo, len(sfip))
		// 	for i, si := range sfip {
		// 		sfip2[i] = *si
		// 	}
		// 	for i := range sfip {
		// 		sfip[i] = &sfip2[i]
		// 	}
		// }

		ti.sfip = make([]*structFieldInfo, len(sfip))
		ti.sfi = make([]*structFieldInfo, len(sfip))
		copy(ti.sfip, sfip)
		sort.Sort(sfiSortedByEncName(sfip))
		copy(ti.sfi, sfip)
	}
	// sfi = sfip
	cachedTypeInfo[rtid] = pti
	return
}

func rgetTypeInfo(rt reflect.Type, indexstack []int, fnameToHastag map[string]bool,
	sfi *[]*structFieldInfo, siInfo *structFieldInfo,
) {
	// for rt.Kind() == reflect.Ptr {
	// 	// indexstack = append(indexstack, 0)
	// 	rt = rt.Elem()
	// }
	for j := 0; j < rt.NumField(); j++ {
		f := rt.Field(j)
		stag := f.Tag.Get(structTagName)
		if stag == "-" {
			continue
		}
		if r1, _ := utf8.DecodeRuneInString(f.Name); r1 == utf8.RuneError || !unicode.IsUpper(r1) {
			continue
		}
		// if anonymous and there is no struct tag and its a struct (or pointer to struct), inline it.
		if f.Anonymous && stag == "" {
			ft := f.Type
			for ft.Kind() == reflect.Ptr {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				indexstack2 := append(append(make([]int, 0, len(indexstack)+4), indexstack...), j)
				rgetTypeInfo(ft, indexstack2, fnameToHastag, sfi, siInfo)
				continue
			}
		}
		// do not let fields with same name in embedded structs override field at higher level.
		// this must be done after anonymous check, to allow anonymous field
		// still include their child fields
		if _, ok := fnameToHastag[f.Name]; ok {
			continue
		}
		si := parseStructFieldInfo(f.Name, stag)
		// si.ikind = int(f.Type.Kind())
		if len(indexstack) == 0 {
			si.i = int16(j)
		} else {
			si.i = -1
			si.is = append(append(make([]int, 0, len(indexstack)+4), indexstack...), j)
		}

		if siInfo != nil {
			if siInfo.omitEmpty {
				si.omitEmpty = true
			}
		}
		*sfi = append(*sfi, si)
		fnameToHastag[f.Name] = stag != ""
	}
}

func panicToErr(err *error) {
	if recoverPanicToErr {
		if x := recover(); x != nil {
			//debug.PrintStack()
			panicValToErr(x, err)
		}
	}
}

func doPanic(tag string, format string, params ...interface{}) {
	params2 := make([]interface{}, len(params)+1)
	params2[0] = tag
	copy(params2[1:], params)
	panic(fmt.Errorf("%s: "+format, params2...))
}

func checkOverflowFloat32(f float64, doCheck bool) {
	if !doCheck {
		return
	}
	// check overflow (logic adapted from std pkg reflect/value.go OverflowFloat()
	f2 := f
	if f2 < 0 {
		f2 = -f
	}
	if math.MaxFloat32 < f2 && f2 <= math.MaxFloat64 {
		decErr("Overflow float32 value: %v", f2)
	}
}

func checkOverflow(ui uint64, i int64, bitsize uint8) {
	// check overflow (logic adapted from std pkg reflect/value.go OverflowUint()
	if bitsize == 0 {
		return
	}
	if i != 0 {
		if trunc := (i << (64 - bitsize)) >> (64 - bitsize); i != trunc {
			decErr("Overflow int value: %v", i)
		}
	}
	if ui != 0 {
		if trunc := (ui << (64 - bitsize)) >> (64 - bitsize); ui != trunc {
			decErr("Overflow uint value: %v", ui)
		}
	}
}
