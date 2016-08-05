// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// Contains code shared by both encode and decode.

// Some shared ideas around encoding/decoding
// ------------------------------------------
//
// If an interface{} is passed, we first do a type assertion to see if it is
// a primitive type or a map/slice of primitive types, and use a fastpath to handle it.
//
// If we start with a reflect.Value, we are already in reflect.Value land and
// will try to grab the function for the underlying Type and directly call that function.
// This is more performant than calling reflect.Value.Interface().
//
// This still helps us bypass many layers of reflection, and give best performance.
//
// Containers
// ------------
// Containers in the stream are either associative arrays (key-value pairs) or
// regular arrays (indexed by incrementing integers).
//
// Some streams support indefinite-length containers, and use a breaking
// byte-sequence to denote that the container has come to an end.
//
// Some streams also are text-based, and use explicit separators to denote the
// end/beginning of different values.
//
// During encode, we use a high-level condition to determine how to iterate through
// the container. That decision is based on whether the container is text-based (with
// separators) or binary (without separators). If binary, we do not even call the
// encoding of separators.
//
// During decode, we use a different high-level condition to determine how to iterate
// through the containers. That decision is based on whether the stream contained
// a length prefix, or if it used explicit breaks. If length-prefixed, we assume that
// it has to be binary, and we do not even try to read separators.
//
// The only codec that may suffer (slightly) is cbor, and only when decoding indefinite-length.
// It may suffer because we treat it like a text-based codec, and read separators.
// However, this read is a no-op and the cost is insignificant.
//
// Philosophy
// ------------
// On decode, this codec will update containers appropriately:
//    - If struct, update fields from stream into fields of struct.
//      If field in stream not found in struct, handle appropriately (based on option).
//      If a struct field has no corresponding value in the stream, leave it AS IS.
//      If nil in stream, set value to nil/zero value.
//    - If map, update map from stream.
//      If the stream value is NIL, set the map to nil.
//    - if slice, try to update up to length of array in stream.
//      if container len is less than stream array length,
//      and container cannot be expanded, handled (based on option).
//      This means you can decode 4-element stream array into 1-element array.
//
// ------------------------------------
// On encode, user can specify omitEmpty. This means that the value will be omitted
// if the zero value. The problem may occur during decode, where omitted values do not affect
// the value being decoded into. This means that if decoding into a struct with an
// int field with current value=5, and the field is omitted in the stream, then after
// decoding, the value will still be 5 (not 0).
// omitEmpty only works if you guarantee that you always decode into zero-values.
//
// ------------------------------------
// We could have truncated a map to remove keys not available in the stream,
// or set values in the struct which are not in the stream to their zero values.
// We decided against it because there is no efficient way to do it.
// We may introduce it as an option later.
// However, that will require enabling it for both runtime and code generation modes.
//
// To support truncate, we need to do 2 passes over the container:
//   map
//   - first collect all keys (e.g. in k1)
//   - for each key in stream, mark k1 that the key should not be removed
//   - after updating map, do second pass and call delete for all keys in k1 which are not marked
//   struct:
//   - for each field, track the *typeInfo s1
//   - iterate through all s1, and for each one not marked, set value to zero
//   - this involves checking the possible anonymous fields which are nil ptrs.
//     too much work.
//
// ------------------------------------------
// Error Handling is done within the library using panic.
//
// This way, the code doesn't have to keep checking if an error has happened,
// and we don't have to keep sending the error value along with each call
// or storing it in the En|Decoder and checking it constantly along the way.
//
// The disadvantage is that small functions which use panics cannot be inlined.
// The code accounts for that by only using panics behind an interface;
// since interface calls cannot be inlined, this is irrelevant.
//
// We considered storing the error is En|Decoder.
//   - once it has its err field set, it cannot be used again.
//   - panicing will be optional, controlled by const flag.
//   - code should always check error first and return early.
// We eventually decided against it as it makes the code clumsier to always
// check for these error conditions.

import (
	"bytes"
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	scratchByteArrayLen = 32
	initCollectionCap   = 32 // 32 is defensive. 16 is preferred.

	// Support encoding.(Binary|Text)(Unm|M)arshaler.
	// This constant flag will enable or disable it.
	supportMarshalInterfaces = true

	// Each Encoder or Decoder uses a cache of functions based on conditionals,
	// so that the conditionals are not run every time.
	//
	// Either a map or a slice is used to keep track of the functions.
	// The map is more natural, but has a higher cost than a slice/array.
	// This flag (useMapForCodecCache) controls which is used.
	//
	// From benchmarks, slices with linear search perform better with < 32 entries.
	// We have typically seen a high threshold of about 24 entries.
	useMapForCodecCache = false

	// for debugging, set this to false, to catch panic traces.
	// Note that this will always cause rpc tests to fail, since they need io.EOF sent via panic.
	recoverPanicToErr = true

	// Fast path functions try to create a fast path encode or decode implementation
	// for common maps and slices, by by-passing reflection altogether.
	fastpathEnabled = true

	// if checkStructForEmptyValue, check structs fields to see if an empty value.
	// This could be an expensive call, so possibly disable it.
	checkStructForEmptyValue = false

	// if derefForIsEmptyValue, deref pointers and interfaces when checking isEmptyValue
	derefForIsEmptyValue = false

	// if resetSliceElemToZeroValue, then on decoding a slice, reset the element to a zero value first.
	// Only concern is that, if the slice already contained some garbage, we will decode into that garbage.
	// The chances of this are slim, so leave this "optimization".
	// TODO: should this be true, to ensure that we always decode into a "zero" "empty" value?
	resetSliceElemToZeroValue bool = false
)

var (
	oneByteArr    = [1]byte{0}
	zeroByteSlice = oneByteArr[:0:0]
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

	// valueTypeInvalid = 0xff
)

type seqType uint8

const (
	_ seqType = iota
	seqTypeArray
	seqTypeSlice
	seqTypeChan
)

// note that containerMapStart and containerArraySend are not sent.
// This is because the ReadXXXStart and EncodeXXXStart already does these.
type containerState uint8

const (
	_ containerState = iota

	containerMapStart // slot left open, since Driver method already covers it
	containerMapKey
	containerMapValue
	containerMapEnd
	containerArrayStart // slot left open, since Driver methods already cover it
	containerArrayElem
	containerArrayEnd
)

type rgetPoolT struct {
	encNames [8]string
	fNames   [8]string
	etypes   [8]uintptr
	sfis     [8]*structFieldInfo
}

var rgetPool = sync.Pool{
	New: func() interface{} { return new(rgetPoolT) },
}

type rgetT struct {
	fNames   []string
	encNames []string
	etypes   []uintptr
	sfis     []*structFieldInfo
}

type containerStateRecv interface {
	sendContainerState(containerState)
}

// mirror json.Marshaler and json.Unmarshaler here,
// so we don't import the encoding/json package
type jsonMarshaler interface {
	MarshalJSON() ([]byte, error)
}
type jsonUnmarshaler interface {
	UnmarshalJSON([]byte) error
}

var (
	bigen               = binary.BigEndian
	structInfoFieldName = "_struct"

	mapStrIntfTyp  = reflect.TypeOf(map[string]interface{}(nil))
	mapIntfIntfTyp = reflect.TypeOf(map[interface{}]interface{}(nil))
	intfSliceTyp   = reflect.TypeOf([]interface{}(nil))
	intfTyp        = intfSliceTyp.Elem()

	stringTyp     = reflect.TypeOf("")
	timeTyp       = reflect.TypeOf(time.Time{})
	rawExtTyp     = reflect.TypeOf(RawExt{})
	uint8SliceTyp = reflect.TypeOf([]uint8(nil))

	mapBySliceTyp = reflect.TypeOf((*MapBySlice)(nil)).Elem()

	binaryMarshalerTyp   = reflect.TypeOf((*encoding.BinaryMarshaler)(nil)).Elem()
	binaryUnmarshalerTyp = reflect.TypeOf((*encoding.BinaryUnmarshaler)(nil)).Elem()

	textMarshalerTyp   = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
	textUnmarshalerTyp = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()

	jsonMarshalerTyp   = reflect.TypeOf((*jsonMarshaler)(nil)).Elem()
	jsonUnmarshalerTyp = reflect.TypeOf((*jsonUnmarshaler)(nil)).Elem()

	selferTyp = reflect.TypeOf((*Selfer)(nil)).Elem()

	uint8SliceTypId = reflect.ValueOf(uint8SliceTyp).Pointer()
	rawExtTypId     = reflect.ValueOf(rawExtTyp).Pointer()
	intfTypId       = reflect.ValueOf(intfTyp).Pointer()
	timeTypId       = reflect.ValueOf(timeTyp).Pointer()
	stringTypId     = reflect.ValueOf(stringTyp).Pointer()

	mapStrIntfTypId  = reflect.ValueOf(mapStrIntfTyp).Pointer()
	mapIntfIntfTypId = reflect.ValueOf(mapIntfIntfTyp).Pointer()
	intfSliceTypId   = reflect.ValueOf(intfSliceTyp).Pointer()
	// mapBySliceTypId  = reflect.ValueOf(mapBySliceTyp).Pointer()

	intBitsize  uint8 = uint8(reflect.TypeOf(int(0)).Bits())
	uintBitsize uint8 = uint8(reflect.TypeOf(uint(0)).Bits())

	bsAll0x00 = []byte{0, 0, 0, 0, 0, 0, 0, 0}
	bsAll0xff = []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}

	chkOvf checkOverflow

	noFieldNameToStructFieldInfoErr = errors.New("no field name passed to parseStructFieldInfo")
)

var defTypeInfos = NewTypeInfos([]string{"codec", "json"})

// Selfer defines methods by which a value can encode or decode itself.
//
// Any type which implements Selfer will be able to encode or decode itself.
// Consequently, during (en|de)code, this takes precedence over
// (text|binary)(M|Unm)arshal or extension support.
type Selfer interface {
	CodecEncodeSelf(*Encoder)
	CodecDecodeSelf(*Decoder)
}

// MapBySlice represents a slice which should be encoded as a map in the stream.
// The slice contains a sequence of key-value pairs.
// This affords storing a map in a specific sequence in the stream.
//
// The support of MapBySlice affords the following:
//   - A slice type which implements MapBySlice will be encoded as a map
//   - A slice can be decoded from a map in the stream
type MapBySlice interface {
	MapBySlice()
}

// WARNING: DO NOT USE DIRECTLY. EXPORTED FOR GODOC BENEFIT. WILL BE REMOVED.
//
// BasicHandle encapsulates the common options and extension functions.
type BasicHandle struct {
	// TypeInfos is used to get the type info for any type.
	//
	// If not configured, the default TypeInfos is used, which uses struct tag keys: codec, json
	TypeInfos *TypeInfos

	extHandle
	EncodeOptions
	DecodeOptions
}

func (x *BasicHandle) getBasicHandle() *BasicHandle {
	return x
}

func (x *BasicHandle) getTypeInfo(rtid uintptr, rt reflect.Type) (pti *typeInfo) {
	if x.TypeInfos != nil {
		return x.TypeInfos.get(rtid, rt)
	}
	return defTypeInfos.get(rtid, rt)
}

// Handle is the interface for a specific encoding format.
//
// Typically, a Handle is pre-configured before first time use,
// and not modified while in use. Such a pre-configured Handle
// is safe for concurrent access.
type Handle interface {
	getBasicHandle() *BasicHandle
	newEncDriver(w *Encoder) encDriver
	newDecDriver(r *Decoder) decDriver
	isBinary() bool
}

// RawExt represents raw unprocessed extension data.
// Some codecs will decode extension data as a *RawExt if there is no registered extension for the tag.
//
// Only one of Data or Value is nil. If Data is nil, then the content of the RawExt is in the Value.
type RawExt struct {
	Tag uint64
	// Data is the []byte which represents the raw ext. If Data is nil, ext is exposed in Value.
	// Data is used by codecs (e.g. binc, msgpack, simple) which do custom serialization of the types
	Data []byte
	// Value represents the extension, if Data is nil.
	// Value is used by codecs (e.g. cbor) which use the format to do custom serialization of the types.
	Value interface{}
}

// BytesExt handles custom (de)serialization of types to/from []byte.
// It is used by codecs (e.g. binc, msgpack, simple) which do custom serialization of the types.
type BytesExt interface {
	// WriteExt converts a value to a []byte.
	//
	// Note: v *may* be a pointer to the extension type, if the extension type was a struct or array.
	WriteExt(v interface{}) []byte

	// ReadExt updates a value from a []byte.
	ReadExt(dst interface{}, src []byte)
}

// InterfaceExt handles custom (de)serialization of types to/from another interface{} value.
// The Encoder or Decoder will then handle the further (de)serialization of that known type.
//
// It is used by codecs (e.g. cbor, json) which use the format to do custom serialization of the types.
type InterfaceExt interface {
	// ConvertExt converts a value into a simpler interface for easy encoding e.g. convert time.Time to int64.
	//
	// Note: v *may* be a pointer to the extension type, if the extension type was a struct or array.
	ConvertExt(v interface{}) interface{}

	// UpdateExt updates a value from a simpler interface for easy decoding e.g. convert int64 to time.Time.
	UpdateExt(dst interface{}, src interface{})
}

// Ext handles custom (de)serialization of custom types / extensions.
type Ext interface {
	BytesExt
	InterfaceExt
}

// addExtWrapper is a wrapper implementation to support former AddExt exported method.
type addExtWrapper struct {
	encFn func(reflect.Value) ([]byte, error)
	decFn func(reflect.Value, []byte) error
}

func (x addExtWrapper) WriteExt(v interface{}) []byte {
	bs, err := x.encFn(reflect.ValueOf(v))
	if err != nil {
		panic(err)
	}
	return bs
}

func (x addExtWrapper) ReadExt(v interface{}, bs []byte) {
	if err := x.decFn(reflect.ValueOf(v), bs); err != nil {
		panic(err)
	}
}

func (x addExtWrapper) ConvertExt(v interface{}) interface{} {
	return x.WriteExt(v)
}

func (x addExtWrapper) UpdateExt(dest interface{}, v interface{}) {
	x.ReadExt(dest, v.([]byte))
}

type setExtWrapper struct {
	b BytesExt
	i InterfaceExt
}

func (x *setExtWrapper) WriteExt(v interface{}) []byte {
	if x.b == nil {
		panic("BytesExt.WriteExt is not supported")
	}
	return x.b.WriteExt(v)
}

func (x *setExtWrapper) ReadExt(v interface{}, bs []byte) {
	if x.b == nil {
		panic("BytesExt.WriteExt is not supported")

	}
	x.b.ReadExt(v, bs)
}

func (x *setExtWrapper) ConvertExt(v interface{}) interface{} {
	if x.i == nil {
		panic("InterfaceExt.ConvertExt is not supported")

	}
	return x.i.ConvertExt(v)
}

func (x *setExtWrapper) UpdateExt(dest interface{}, v interface{}) {
	if x.i == nil {
		panic("InterfaceExxt.UpdateExt is not supported")

	}
	x.i.UpdateExt(dest, v)
}

// type errorString string
// func (x errorString) Error() string { return string(x) }

type binaryEncodingType struct{}

func (_ binaryEncodingType) isBinary() bool { return true }

type textEncodingType struct{}

func (_ textEncodingType) isBinary() bool { return false }

// noBuiltInTypes is embedded into many types which do not support builtins
// e.g. msgpack, simple, cbor.
type noBuiltInTypes struct{}

func (_ noBuiltInTypes) IsBuiltinType(rt uintptr) bool           { return false }
func (_ noBuiltInTypes) EncodeBuiltin(rt uintptr, v interface{}) {}
func (_ noBuiltInTypes) DecodeBuiltin(rt uintptr, v interface{}) {}

type noStreamingCodec struct{}

func (_ noStreamingCodec) CheckBreak() bool { return false }

// bigenHelper.
// Users must already slice the x completely, because we will not reslice.
type bigenHelper struct {
	x []byte // must be correctly sliced to appropriate len. slicing is a cost.
	w encWriter
}

func (z bigenHelper) writeUint16(v uint16) {
	bigen.PutUint16(z.x, v)
	z.w.writeb(z.x)
}

func (z bigenHelper) writeUint32(v uint32) {
	bigen.PutUint32(z.x, v)
	z.w.writeb(z.x)
}

func (z bigenHelper) writeUint64(v uint64) {
	bigen.PutUint64(z.x, v)
	z.w.writeb(z.x)
}

type extTypeTagFn struct {
	rtid uintptr
	rt   reflect.Type
	tag  uint64
	ext  Ext
}

type extHandle []extTypeTagFn

// DEPRECATED: Use SetBytesExt or SetInterfaceExt on the Handle instead.
//
// AddExt registes an encode and decode function for a reflect.Type.
// AddExt internally calls SetExt.
// To deregister an Ext, call AddExt with nil encfn and/or nil decfn.
func (o *extHandle) AddExt(
	rt reflect.Type, tag byte,
	encfn func(reflect.Value) ([]byte, error), decfn func(reflect.Value, []byte) error,
) (err error) {
	if encfn == nil || decfn == nil {
		return o.SetExt(rt, uint64(tag), nil)
	}
	return o.SetExt(rt, uint64(tag), addExtWrapper{encfn, decfn})
}

// DEPRECATED: Use SetBytesExt or SetInterfaceExt on the Handle instead.
//
// Note that the type must be a named type, and specifically not
// a pointer or Interface. An error is returned if that is not honored.
//
// To Deregister an ext, call SetExt with nil Ext
func (o *extHandle) SetExt(rt reflect.Type, tag uint64, ext Ext) (err error) {
	// o is a pointer, because we may need to initialize it
	if rt.PkgPath() == "" || rt.Kind() == reflect.Interface {
		err = fmt.Errorf("codec.Handle.AddExt: Takes named type, especially not a pointer or interface: %T",
			reflect.Zero(rt).Interface())
		return
	}

	rtid := reflect.ValueOf(rt).Pointer()
	for _, v := range *o {
		if v.rtid == rtid {
			v.tag, v.ext = tag, ext
			return
		}
	}

	if *o == nil {
		*o = make([]extTypeTagFn, 0, 4)
	}
	*o = append(*o, extTypeTagFn{rtid, rt, tag, ext})
	return
}

func (o extHandle) getExt(rtid uintptr) *extTypeTagFn {
	var v *extTypeTagFn
	for i := range o {
		v = &o[i]
		if v.rtid == rtid {
			return v
		}
	}
	return nil
}

func (o extHandle) getExtForTag(tag uint64) *extTypeTagFn {
	var v *extTypeTagFn
	for i := range o {
		v = &o[i]
		if v.tag == tag {
			return v
		}
	}
	return nil
}

type structFieldInfo struct {
	encName string // encode name

	// only one of 'i' or 'is' can be set. If 'i' is -1, then 'is' has been set.

	is        []int // (recursive/embedded) field index in struct
	i         int16 // field index in struct
	omitEmpty bool
	toArray   bool // if field is _struct, is the toArray set?
}

// func (si *structFieldInfo) isZero() bool {
// 	return si.encName == "" && len(si.is) == 0 && si.i == 0 && !si.omitEmpty && !si.toArray
// }

// rv returns the field of the struct.
// If anonymous, it returns an Invalid
func (si *structFieldInfo) field(v reflect.Value, update bool) (rv2 reflect.Value) {
	if si.i != -1 {
		v = v.Field(int(si.i))
		return v
	}
	// replicate FieldByIndex
	for _, x := range si.is {
		for v.Kind() == reflect.Ptr {
			if v.IsNil() {
				if !update {
					return
				}
				v.Set(reflect.New(v.Type().Elem()))
			}
			v = v.Elem()
		}
		v = v.Field(x)
	}
	return v
}

func (si *structFieldInfo) setToZeroValue(v reflect.Value) {
	if si.i != -1 {
		v = v.Field(int(si.i))
		v.Set(reflect.Zero(v.Type()))
		// v.Set(reflect.New(v.Type()).Elem())
		// v.Set(reflect.New(v.Type()))
	} else {
		// replicate FieldByIndex
		for _, x := range si.is {
			for v.Kind() == reflect.Ptr {
				if v.IsNil() {
					return
				}
				v = v.Elem()
			}
			v = v.Field(x)
		}
		v.Set(reflect.Zero(v.Type()))
	}
}

func parseStructFieldInfo(fname string, stag string) *structFieldInfo {
	// if fname == "" {
	// 	panic(noFieldNameToStructFieldInfoErr)
	// }
	si := structFieldInfo{
		encName: fname,
	}

	if stag != "" {
		for i, s := range strings.Split(stag, ",") {
			if i == 0 {
				if s != "" {
					si.encName = s
				}
			} else {
				if s == "omitempty" {
					si.omitEmpty = true
				} else if s == "toarray" {
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
//   - If type is text(M/Unm)arshaler, call Text(M/Unm)arshal method
//   - Else decode appropriately based on the reflect.Kind
type typeInfo struct {
	sfi  []*structFieldInfo // sorted. Used when enc/dec struct to map.
	sfip []*structFieldInfo // unsorted. Used when enc/dec struct to array.

	rt   reflect.Type
	rtid uintptr

	numMeth uint16 // number of methods

	// baseId gives pointer to the base reflect.Type, after deferencing
	// the pointers. E.g. base type of ***time.Time is time.Time.
	base      reflect.Type
	baseId    uintptr
	baseIndir int8 // number of indirections to get to base

	mbs bool // base type (T or *T) is a MapBySlice

	bm        bool // base type (T or *T) is a binaryMarshaler
	bunm      bool // base type (T or *T) is a binaryUnmarshaler
	bmIndir   int8 // number of indirections to get to binaryMarshaler type
	bunmIndir int8 // number of indirections to get to binaryUnmarshaler type

	tm        bool // base type (T or *T) is a textMarshaler
	tunm      bool // base type (T or *T) is a textUnmarshaler
	tmIndir   int8 // number of indirections to get to textMarshaler type
	tunmIndir int8 // number of indirections to get to textUnmarshaler type

	jm        bool // base type (T or *T) is a jsonMarshaler
	junm      bool // base type (T or *T) is a jsonUnmarshaler
	jmIndir   int8 // number of indirections to get to jsonMarshaler type
	junmIndir int8 // number of indirections to get to jsonUnmarshaler type

	cs      bool // base type (T or *T) is a Selfer
	csIndir int8 // number of indirections to get to Selfer type

	toArray bool // whether this (struct) type should be encoded as an array
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

// TypeInfos caches typeInfo for each type on first inspection.
//
// It is configured with a set of tag keys, which are used to get
// configuration for the type.
type TypeInfos struct {
	infos map[uintptr]*typeInfo
	mu    sync.RWMutex
	tags  []string
}

// NewTypeInfos creates a TypeInfos given a set of struct tags keys.
//
// This allows users customize the struct tag keys which contain configuration
// of their types.
func NewTypeInfos(tags []string) *TypeInfos {
	return &TypeInfos{tags: tags, infos: make(map[uintptr]*typeInfo, 64)}
}

func (x *TypeInfos) structTag(t reflect.StructTag) (s string) {
	// check for tags: codec, json, in that order.
	// this allows seamless support for many configured structs.
	for _, x := range x.tags {
		s = t.Get(x)
		if s != "" {
			return s
		}
	}
	return
}

func (x *TypeInfos) get(rtid uintptr, rt reflect.Type) (pti *typeInfo) {
	var ok bool
	x.mu.RLock()
	pti, ok = x.infos[rtid]
	x.mu.RUnlock()
	if ok {
		return
	}

	// do not hold lock while computing this.
	// it may lead to duplication, but that's ok.
	ti := typeInfo{rt: rt, rtid: rtid}
	ti.numMeth = uint16(rt.NumMethod())

	var indir int8
	if ok, indir = implementsIntf(rt, binaryMarshalerTyp); ok {
		ti.bm, ti.bmIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, binaryUnmarshalerTyp); ok {
		ti.bunm, ti.bunmIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, textMarshalerTyp); ok {
		ti.tm, ti.tmIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, textUnmarshalerTyp); ok {
		ti.tunm, ti.tunmIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, jsonMarshalerTyp); ok {
		ti.jm, ti.jmIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, jsonUnmarshalerTyp); ok {
		ti.junm, ti.junmIndir = true, indir
	}
	if ok, indir = implementsIntf(rt, selferTyp); ok {
		ti.cs, ti.csIndir = true, indir
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
			siInfo = parseStructFieldInfo(structInfoFieldName, x.structTag(f.Tag))
			ti.toArray = siInfo.toArray
		}
		pi := rgetPool.Get()
		pv := pi.(*rgetPoolT)
		pv.etypes[0] = ti.baseId
		vv := rgetT{pv.fNames[:0], pv.encNames[:0], pv.etypes[:1], pv.sfis[:0]}
		x.rget(rt, rtid, nil, &vv, siInfo)
		ti.sfip = make([]*structFieldInfo, len(vv.sfis))
		ti.sfi = make([]*structFieldInfo, len(vv.sfis))
		copy(ti.sfip, vv.sfis)
		sort.Sort(sfiSortedByEncName(vv.sfis))
		copy(ti.sfi, vv.sfis)
		rgetPool.Put(pi)
	}
	// sfi = sfip

	x.mu.Lock()
	if pti, ok = x.infos[rtid]; !ok {
		pti = &ti
		x.infos[rtid] = pti
	}
	x.mu.Unlock()
	return
}

func (x *TypeInfos) rget(rt reflect.Type, rtid uintptr,
	indexstack []int, pv *rgetT, siInfo *structFieldInfo,
) {
	// This will read up the fields and store how to access the value.
	// It uses the go language's rules for embedding, as below:
	//   - if a field has been seen while traversing, skip it
	//   - if an encName has been seen while traversing, skip it
	//   - if an embedded type has been seen, skip it
	//
	// Also, per Go's rules, embedded fields must be analyzed AFTER all top-level fields.
	//
	// Note: we consciously use slices, not a map, to simulate a set.
	//       Typically, types have < 16 fields, and iteration using equals is faster than maps there

	type anonField struct {
		ft  reflect.Type
		idx int
	}

	var anonFields []anonField

LOOP:
	for j, jlen := 0, rt.NumField(); j < jlen; j++ {
		f := rt.Field(j)
		fkind := f.Type.Kind()
		// skip if a func type, or is unexported, or structTag value == "-"
		switch fkind {
		case reflect.Func, reflect.Complex64, reflect.Complex128, reflect.UnsafePointer:
			continue LOOP
		}

		// if r1, _ := utf8.DecodeRuneInString(f.Name); r1 == utf8.RuneError || !unicode.IsUpper(r1) {
		if f.PkgPath != "" && !f.Anonymous { // unexported, not embedded
			continue
		}
		stag := x.structTag(f.Tag)
		if stag == "-" {
			continue
		}
		var si *structFieldInfo
		// if anonymous and no struct tag (or it's blank), and a struct (or pointer to struct), inline it.
		if f.Anonymous && fkind != reflect.Interface {
			doInline := stag == ""
			if !doInline {
				si = parseStructFieldInfo("", stag)
				doInline = si.encName == ""
				// doInline = si.isZero()
			}
			if doInline {
				ft := f.Type
				for ft.Kind() == reflect.Ptr {
					ft = ft.Elem()
				}
				if ft.Kind() == reflect.Struct {
					// handle anonymous fields after handling all the non-anon fields
					anonFields = append(anonFields, anonField{ft, j})
					continue
				}
			}
		}

		// after the anonymous dance: if an unexported field, skip
		if f.PkgPath != "" { // unexported
			continue
		}

		if f.Name == "" {
			panic(noFieldNameToStructFieldInfoErr)
		}

		for _, k := range pv.fNames {
			if k == f.Name {
				continue LOOP
			}
		}
		pv.fNames = append(pv.fNames, f.Name)

		if si == nil {
			si = parseStructFieldInfo(f.Name, stag)
		} else if si.encName == "" {
			si.encName = f.Name
		}

		for _, k := range pv.encNames {
			if k == si.encName {
				continue LOOP
			}
		}
		pv.encNames = append(pv.encNames, si.encName)

		// si.ikind = int(f.Type.Kind())
		if len(indexstack) == 0 {
			si.i = int16(j)
		} else {
			si.i = -1
			si.is = make([]int, len(indexstack)+1)
			copy(si.is, indexstack)
			si.is[len(indexstack)] = j
			// si.is = append(append(make([]int, 0, len(indexstack)+4), indexstack...), j)
		}

		if siInfo != nil {
			if siInfo.omitEmpty {
				si.omitEmpty = true
			}
		}
		pv.sfis = append(pv.sfis, si)
	}

	// now handle anonymous fields
LOOP2:
	for _, af := range anonFields {
		// if etypes contains this, then do not call rget again (as the fields are already seen here)
		ftid := reflect.ValueOf(af.ft).Pointer()
		for _, k := range pv.etypes {
			if k == ftid {
				continue LOOP2
			}
		}
		pv.etypes = append(pv.etypes, ftid)

		indexstack2 := make([]int, len(indexstack)+1)
		copy(indexstack2, indexstack)
		indexstack2[len(indexstack)] = af.idx
		// indexstack2 := append(append(make([]int, 0, len(indexstack)+4), indexstack...), j)
		x.rget(af.ft, ftid, indexstack2, pv, siInfo)
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

// func doPanic(tag string, format string, params ...interface{}) {
// 	params2 := make([]interface{}, len(params)+1)
// 	params2[0] = tag
// 	copy(params2[1:], params)
// 	panic(fmt.Errorf("%s: "+format, params2...))
// }

func isImmutableKind(k reflect.Kind) (v bool) {
	return false ||
		k == reflect.Int ||
		k == reflect.Int8 ||
		k == reflect.Int16 ||
		k == reflect.Int32 ||
		k == reflect.Int64 ||
		k == reflect.Uint ||
		k == reflect.Uint8 ||
		k == reflect.Uint16 ||
		k == reflect.Uint32 ||
		k == reflect.Uint64 ||
		k == reflect.Uintptr ||
		k == reflect.Float32 ||
		k == reflect.Float64 ||
		k == reflect.Bool ||
		k == reflect.String
}

// these functions must be inlinable, and not call anybody
type checkOverflow struct{}

func (_ checkOverflow) Float32(f float64) (overflow bool) {
	if f < 0 {
		f = -f
	}
	return math.MaxFloat32 < f && f <= math.MaxFloat64
}

func (_ checkOverflow) Uint(v uint64, bitsize uint8) (overflow bool) {
	if bitsize == 0 || bitsize >= 64 || v == 0 {
		return
	}
	if trunc := (v << (64 - bitsize)) >> (64 - bitsize); v != trunc {
		overflow = true
	}
	return
}

func (_ checkOverflow) Int(v int64, bitsize uint8) (overflow bool) {
	if bitsize == 0 || bitsize >= 64 || v == 0 {
		return
	}
	if trunc := (v << (64 - bitsize)) >> (64 - bitsize); v != trunc {
		overflow = true
	}
	return
}

func (_ checkOverflow) SignedInt(v uint64) (i int64, overflow bool) {
	//e.g. -127 to 128 for int8
	pos := (v >> 63) == 0
	ui2 := v & 0x7fffffffffffffff
	if pos {
		if ui2 > math.MaxInt64 {
			overflow = true
			return
		}
	} else {
		if ui2 > math.MaxInt64-1 {
			overflow = true
			return
		}
	}
	i = int64(v)
	return
}

// ------------------ SORT -----------------

func isNaN(f float64) bool { return f != f }

// -----------------------

type intSlice []int64
type uintSlice []uint64
type floatSlice []float64
type boolSlice []bool
type stringSlice []string
type bytesSlice [][]byte

func (p intSlice) Len() int           { return len(p) }
func (p intSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p intSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p uintSlice) Len() int           { return len(p) }
func (p uintSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p uintSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p floatSlice) Len() int { return len(p) }
func (p floatSlice) Less(i, j int) bool {
	return p[i] < p[j] || isNaN(p[i]) && !isNaN(p[j])
}
func (p floatSlice) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

func (p stringSlice) Len() int           { return len(p) }
func (p stringSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p stringSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p bytesSlice) Len() int           { return len(p) }
func (p bytesSlice) Less(i, j int) bool { return bytes.Compare(p[i], p[j]) == -1 }
func (p bytesSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p boolSlice) Len() int           { return len(p) }
func (p boolSlice) Less(i, j int) bool { return !p[i] && p[j] }
func (p boolSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// ---------------------

type intRv struct {
	v int64
	r reflect.Value
}
type intRvSlice []intRv
type uintRv struct {
	v uint64
	r reflect.Value
}
type uintRvSlice []uintRv
type floatRv struct {
	v float64
	r reflect.Value
}
type floatRvSlice []floatRv
type boolRv struct {
	v bool
	r reflect.Value
}
type boolRvSlice []boolRv
type stringRv struct {
	v string
	r reflect.Value
}
type stringRvSlice []stringRv
type bytesRv struct {
	v []byte
	r reflect.Value
}
type bytesRvSlice []bytesRv

func (p intRvSlice) Len() int           { return len(p) }
func (p intRvSlice) Less(i, j int) bool { return p[i].v < p[j].v }
func (p intRvSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p uintRvSlice) Len() int           { return len(p) }
func (p uintRvSlice) Less(i, j int) bool { return p[i].v < p[j].v }
func (p uintRvSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p floatRvSlice) Len() int { return len(p) }
func (p floatRvSlice) Less(i, j int) bool {
	return p[i].v < p[j].v || isNaN(p[i].v) && !isNaN(p[j].v)
}
func (p floatRvSlice) Swap(i, j int) { p[i], p[j] = p[j], p[i] }

func (p stringRvSlice) Len() int           { return len(p) }
func (p stringRvSlice) Less(i, j int) bool { return p[i].v < p[j].v }
func (p stringRvSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p bytesRvSlice) Len() int           { return len(p) }
func (p bytesRvSlice) Less(i, j int) bool { return bytes.Compare(p[i].v, p[j].v) == -1 }
func (p bytesRvSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

func (p boolRvSlice) Len() int           { return len(p) }
func (p boolRvSlice) Less(i, j int) bool { return !p[i].v && p[j].v }
func (p boolRvSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// -----------------

type bytesI struct {
	v []byte
	i interface{}
}

type bytesISlice []bytesI

func (p bytesISlice) Len() int           { return len(p) }
func (p bytesISlice) Less(i, j int) bool { return bytes.Compare(p[i].v, p[j].v) == -1 }
func (p bytesISlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// -----------------

type set []uintptr

func (s *set) add(v uintptr) (exists bool) {
	// e.ci is always nil, or len >= 1
	// defer func() { fmt.Printf("$$$$$$$$$$$ cirRef Add: %v, exists: %v\n", v, exists) }()
	x := *s
	if x == nil {
		x = make([]uintptr, 1, 8)
		x[0] = v
		*s = x
		return
	}
	// typically, length will be 1. make this perform.
	if len(x) == 1 {
		if j := x[0]; j == 0 {
			x[0] = v
		} else if j == v {
			exists = true
		} else {
			x = append(x, v)
			*s = x
		}
		return
	}
	// check if it exists
	for _, j := range x {
		if j == v {
			exists = true
			return
		}
	}
	// try to replace a "deleted" slot
	for i, j := range x {
		if j == 0 {
			x[i] = v
			return
		}
	}
	// if unable to replace deleted slot, just append it.
	x = append(x, v)
	*s = x
	return
}

func (s *set) remove(v uintptr) (exists bool) {
	// defer func() { fmt.Printf("$$$$$$$$$$$ cirRef Rm: %v, exists: %v\n", v, exists) }()
	x := *s
	if len(x) == 0 {
		return
	}
	if len(x) == 1 {
		if x[0] == v {
			x[0] = 0
		}
		return
	}
	for i, j := range x {
		if j == v {
			exists = true
			x[i] = 0 // set it to 0, as way to delete it.
			// copy(x[i:], x[i+1:])
			// x = x[:len(x)-1]
			return
		}
	}
	return
}
