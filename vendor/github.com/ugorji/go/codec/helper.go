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
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	scratchByteArrayLen = 32
	// initCollectionCap   = 16 // 32 is defensive. 16 is preferred.

	// Support encoding.(Binary|Text)(Unm|M)arshaler.
	// This constant flag will enable or disable it.
	supportMarshalInterfaces = true

	// for debugging, set this to false, to catch panic traces.
	// Note that this will always cause rpc tests to fail, since they need io.EOF sent via panic.
	recoverPanicToErr = true

	// arrayCacheLen is the length of the cache used in encoder or decoder for
	// allowing zero-alloc initialization.
	arrayCacheLen = 8

	// We tried an optimization, where we detect if a type is one of the known types
	// we optimized for (e.g. int, []uint64, etc).
	//
	// However, we notice some worse performance when using this optimization.
	// So we hide it behind a flag, to turn on if needed.
	useLookupRecognizedTypes = false

	// using recognized allows us to do d.decode(interface{}) instead of d.decodeValue(reflect.Value)
	// when we can infer that the kind of the interface{} is one of the ones hard-coded in the
	// type switch for known types or the ones defined by fast-path.
	//
	// However, it seems we get better performance when we don't recognize, and just let
	// reflection handle it.
	//
	// Reasoning is as below:
	// typeswitch is a binary search with a branch to a code-point.
	// getdecfn is a binary search with a call to a function pointer.
	//
	// both are about the same.
	//
	// so: why prefer typeswitch?
	//
	// is recognized does the following:
	// - lookup rtid
	// - check if in sorted list
	// - calls decode(type switch)
	//   - 1 or 2 binary search to a point in code
	//   - branch there
	//
	// vs getdecfn
	// - lookup rtid
	// - check in sorted list for a function pointer
	// - calls it to decode using reflection (optimized)

	// always set xDebug = false before releasing software
	xDebug = true
)

var (
	oneByteArr    = [1]byte{0}
	zeroByteSlice = oneByteArr[:0:0]
)

var pool pooler

func init() {
	pool.init()
}

// type findCodecFnMode uint8

// const (
// 	findCodecFnModeMap findCodecFnMode = iota
// 	findCodecFnModeBinarySearch
// 	findCodecFnModeLinearSearch
// )

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

func (x valueType) String() string {
	switch x {
	case valueTypeNil:
		return "Nil"
	case valueTypeInt:
		return "Int"
	case valueTypeUint:
		return "Uint"
	case valueTypeFloat:
		return "Float"
	case valueTypeBool:
		return "Bool"
	case valueTypeString:
		return "String"
	case valueTypeSymbol:
		return "Symbol"
	case valueTypeBytes:
		return "Bytes"
	case valueTypeMap:
		return "Map"
	case valueTypeArray:
		return "Array"
	case valueTypeTimestamp:
		return "Timestamp"
	case valueTypeExt:
		return "Ext"
	}
	return strconv.FormatInt(int64(x), 10)
}

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

// sfiIdx used for tracking where a (field/enc)Name is seen in a []*structFieldInfo
type sfiIdx struct {
	name  string
	index int
}

// do not recurse if a containing type refers to an embedded type
// which refers back to its containing type (via a pointer).
// The second time this back-reference happens, break out,
// so as not to cause an infinite loop.
const rgetMaxRecursion = 2

// Anecdotally, we believe most types have <= 12 fields.
// Java's PMD rules set TooManyFields threshold to 15.
const typeInfoLoadArrayLen = 12

type typeInfoLoad struct {
	fNames   []string
	encNames []string
	etypes   []uintptr
	sfis     []*structFieldInfo
}

type typeInfoLoadArray struct {
	fNames   [typeInfoLoadArrayLen]string
	encNames [typeInfoLoadArrayLen]string
	etypes   [typeInfoLoadArrayLen]uintptr
	sfis     [typeInfoLoadArrayLen]*structFieldInfo
	sfiidx   [typeInfoLoadArrayLen]sfiIdx
}

// type containerStateRecv interface {
// 	sendContainerState(containerState)
// }

// mirror json.Marshaler and json.Unmarshaler here,
// so we don't import the encoding/json package
type jsonMarshaler interface {
	MarshalJSON() ([]byte, error)
}
type jsonUnmarshaler interface {
	UnmarshalJSON([]byte) error
}

// type byteAccepter func(byte) bool

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
	rawTyp        = reflect.TypeOf(Raw{})
	uint8SliceTyp = reflect.TypeOf([]uint8(nil))

	mapBySliceTyp = reflect.TypeOf((*MapBySlice)(nil)).Elem()

	binaryMarshalerTyp   = reflect.TypeOf((*encoding.BinaryMarshaler)(nil)).Elem()
	binaryUnmarshalerTyp = reflect.TypeOf((*encoding.BinaryUnmarshaler)(nil)).Elem()

	textMarshalerTyp   = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
	textUnmarshalerTyp = reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()

	jsonMarshalerTyp   = reflect.TypeOf((*jsonMarshaler)(nil)).Elem()
	jsonUnmarshalerTyp = reflect.TypeOf((*jsonUnmarshaler)(nil)).Elem()

	selferTyp = reflect.TypeOf((*Selfer)(nil)).Elem()

	uint8SliceTypId = rt2id(uint8SliceTyp)
	rawExtTypId     = rt2id(rawExtTyp)
	rawTypId        = rt2id(rawTyp)
	intfTypId       = rt2id(intfTyp)
	timeTypId       = rt2id(timeTyp)
	stringTypId     = rt2id(stringTyp)

	mapStrIntfTypId  = rt2id(mapStrIntfTyp)
	mapIntfIntfTypId = rt2id(mapIntfIntfTyp)
	intfSliceTypId   = rt2id(intfSliceTyp)
	// mapBySliceTypId  = rt2id(mapBySliceTyp)

	intBitsize  uint8 = uint8(reflect.TypeOf(int(0)).Bits())
	uintBitsize uint8 = uint8(reflect.TypeOf(uint(0)).Bits())

	bsAll0x00 = []byte{0, 0, 0, 0, 0, 0, 0, 0}
	bsAll0xff = []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}

	chkOvf checkOverflow

	noFieldNameToStructFieldInfoErr = errors.New("no field name passed to parseStructFieldInfo")
)

var defTypeInfos = NewTypeInfos([]string{"codec", "json"})

var immutableKindsSet = [32]bool{
	// reflect.Invalid:  ,
	reflect.Bool:       true,
	reflect.Int:        true,
	reflect.Int8:       true,
	reflect.Int16:      true,
	reflect.Int32:      true,
	reflect.Int64:      true,
	reflect.Uint:       true,
	reflect.Uint8:      true,
	reflect.Uint16:     true,
	reflect.Uint32:     true,
	reflect.Uint64:     true,
	reflect.Uintptr:    true,
	reflect.Float32:    true,
	reflect.Float64:    true,
	reflect.Complex64:  true,
	reflect.Complex128: true,
	// reflect.Array
	// reflect.Chan
	// reflect.Func: true,
	// reflect.Interface
	// reflect.Map
	// reflect.Ptr
	// reflect.Slice
	reflect.String: true,
	// reflect.Struct
	// reflect.UnsafePointer
}

var recognizedRtids []uintptr
var recognizedRtidPtrs []uintptr
var recognizedRtidOrPtrs []uintptr

func init() {
	if !useLookupRecognizedTypes {
		return
	}
	for _, v := range [...]interface{}{
		float32(0),
		float64(0),
		uintptr(0),
		uint(0),
		uint8(0),
		uint16(0),
		uint32(0),
		uint64(0),
		uintptr(0),
		int(0),
		int8(0),
		int16(0),
		int32(0),
		int64(0),
		bool(false),
		string(""),
		Raw{},
		[]byte(nil),
	} {
		rt := reflect.TypeOf(v)
		recognizedRtids = append(recognizedRtids, rt2id(rt))
		recognizedRtidPtrs = append(recognizedRtidPtrs, rt2id(reflect.PtrTo(rt)))
	}
}

func containsU(s []uintptr, v uintptr) bool {
	// return false // TODO: REMOVE
	h, i, j := 0, 0, len(s)
	for i < j {
		h = i + (j-i)/2
		if s[h] < v {
			i = h + 1
		} else {
			j = h
		}
	}
	if i < len(s) && s[i] == v {
		return true
	}
	return false
}

func isRecognizedRtid(rtid uintptr) bool {
	return containsU(recognizedRtids, rtid)
}

func isRecognizedRtidPtr(rtid uintptr) bool {
	return containsU(recognizedRtidPtrs, rtid)
}

func isRecognizedRtidOrPtr(rtid uintptr) bool {
	return containsU(recognizedRtidOrPtrs, rtid)
}

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
	noBuiltInTypeChecker
}

func (x *BasicHandle) getBasicHandle() *BasicHandle {
	return x
}

func (x *BasicHandle) getTypeInfo(rtid uintptr, rt reflect.Type) (pti *typeInfo) {
	if x.TypeInfos == nil {
		return defTypeInfos.get(rtid, rt)
	}
	return x.TypeInfos.get(rtid, rt)
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
	hasElemSeparators() bool
	IsBuiltinType(rtid uintptr) bool
}

// Raw represents raw formatted bytes.
// We "blindly" store it during encode and store the raw bytes during decode.
// Note: it is dangerous during encode, so we may gate the behaviour behind an Encode flag which must be explicitly set.
type Raw []byte

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
	// Value is used by codecs (e.g. cbor, json) which use the format to do custom serialization of the types.
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

type binaryEncodingType struct{}

func (_ binaryEncodingType) isBinary() bool { return true }

type textEncodingType struct{}

func (_ textEncodingType) isBinary() bool { return false }

// noBuiltInTypes is embedded into many types which do not support builtins
// e.g. msgpack, simple, cbor.

type noBuiltInTypeChecker struct{}

func (_ noBuiltInTypeChecker) IsBuiltinType(rt uintptr) bool { return false }

type noBuiltInTypes struct{ noBuiltInTypeChecker }

func (_ noBuiltInTypes) EncodeBuiltin(rt uintptr, v interface{}) {}
func (_ noBuiltInTypes) DecodeBuiltin(rt uintptr, v interface{}) {}

// type noStreamingCodec struct{}
// func (_ noStreamingCodec) CheckBreak() bool { return false }
// func (_ noStreamingCodec) hasElemSeparators() bool { return false }

type noElemSeparators struct{}

func (_ noElemSeparators) hasElemSeparators() (v bool) { return }

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
		err = fmt.Errorf("codec.Handle.AddExt: Takes named type, not a pointer or interface: %T",
			reflect.Zero(rt).Interface())
		return
	}

	rtid := rt2id(rt)
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

const maxLevelsEmbedding = 16

type structFieldInfo struct {
	encName   string // encode name
	fieldName string // field name

	is        [maxLevelsEmbedding]uint16 // (recursive/embedded) field index in struct
	nis       uint8                      // num levels of embedding. if 1, then it's not embedded.
	omitEmpty bool
	toArray   bool // if field is _struct, is the toArray set?
}

func (si *structFieldInfo) setToZeroValue(v reflect.Value) {
	if v, valid := si.field(v, false); valid {
		v.Set(reflect.Zero(v.Type()))
	}
}

// rv returns the field of the struct.
// If anonymous, it returns an Invalid
func (si *structFieldInfo) field(v reflect.Value, update bool) (rv2 reflect.Value, valid bool) {
	// replicate FieldByIndex
	for i, x := range si.is {
		if uint8(i) == si.nis {
			break
		}
		if v, valid = baseStructRv(v, update); !valid {
			return
		}
		v = v.Field(int(x))
	}

	return v, true
}

func (si *structFieldInfo) fieldval(v reflect.Value, update bool) reflect.Value {
	v, _ = si.field(v, update)
	return v
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

const structFieldNodeNumToCache = 4

type structFieldNodeCache struct {
	rv  [structFieldNodeNumToCache]reflect.Value
	idx [structFieldNodeNumToCache]uint32
	num uint8
}

func (x *structFieldNodeCache) get(key uint32) (fv reflect.Value, valid bool) {
	// defer func() { fmt.Printf(">>>> found in cache2? %v\n", valid) }()
	for i, k := range &x.idx {
		if uint8(i) == x.num {
			return // break
		}
		if key == k {
			return x.rv[i], true
		}
	}
	return
}

func (x *structFieldNodeCache) tryAdd(fv reflect.Value, key uint32) {
	if x.num < structFieldNodeNumToCache {
		x.rv[x.num] = fv
		x.idx[x.num] = key
		x.num++
		return
	}
}

type structFieldNode struct {
	v      reflect.Value
	cache2 structFieldNodeCache
	cache3 structFieldNodeCache
	update bool
}

func (x *structFieldNode) field(si *structFieldInfo) (fv reflect.Value) {
	// return si.fieldval(x.v, x.update)
	// Note: we only cache if nis=2 or nis=3 i.e. up to 2 levels of embedding
	// This mostly saves us time on the repeated calls to v.Elem, v.Field, etc.
	var valid bool
	switch si.nis {
	case 1:
		fv = x.v.Field(int(si.is[0]))
	case 2:
		if fv, valid = x.cache2.get(uint32(si.is[0])); valid {
			fv = fv.Field(int(si.is[1]))
			return
		}
		fv = x.v.Field(int(si.is[0]))
		if fv, valid = baseStructRv(fv, x.update); !valid {
			return
		}
		x.cache2.tryAdd(fv, uint32(si.is[0]))
		fv = fv.Field(int(si.is[1]))
	case 3:
		var key uint32 = uint32(si.is[0])<<16 | uint32(si.is[1])
		if fv, valid = x.cache3.get(key); valid {
			fv = fv.Field(int(si.is[2]))
			return
		}
		fv = x.v.Field(int(si.is[0]))
		if fv, valid = baseStructRv(fv, x.update); !valid {
			return
		}
		fv = fv.Field(int(si.is[1]))
		if fv, valid = baseStructRv(fv, x.update); !valid {
			return
		}
		x.cache3.tryAdd(fv, key)
		fv = fv.Field(int(si.is[2]))
	default:
		fv, _ = si.field(x.v, x.update)
	}
	return
}

func baseStructRv(v reflect.Value, update bool) (v2 reflect.Value, valid bool) {
	for v.Kind() == reflect.Ptr {
		if v.IsNil() {
			if !update {
				return
			}
			v.Set(reflect.New(v.Type().Elem()))
		}
		v = v.Elem()
	}
	return v, true
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
	// rv0  reflect.Value // saved zero value, used if immutableKind

	numMeth uint16 // number of methods

	// baseId gives pointer to the base reflect.Type, after deferencing
	// the pointers. E.g. base type of ***time.Time is time.Time.
	base      reflect.Type
	baseId    uintptr
	baseIndir int8 // number of indirections to get to base

	anyOmitEmpty bool

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

// define length beyond which we do a binary search instead of a linear search.
// From our testing, linear search seems faster than binary search up to 16-field structs.
// However, we set to 8 similar to what python does for hashtables.
const indexForEncNameBinarySearchThreshold = 8

func (ti *typeInfo) indexForEncName(name string) int {
	// NOTE: name may be a stringView, so don't pass it to another function.
	//tisfi := ti.sfi
	sfilen := len(ti.sfi)
	if sfilen < indexForEncNameBinarySearchThreshold {
		for i, si := range ti.sfi {
			if si.encName == name {
				return i
			}
		}
		return -1
	}
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
	return -1
}

type rtid2ti struct {
	rtid uintptr
	ti   *typeInfo
}

// TypeInfos caches typeInfo for each type on first inspection.
//
// It is configured with a set of tag keys, which are used to get
// configuration for the type.
type TypeInfos struct {
	infos atomicTypeInfoSlice // formerly map[uintptr]*typeInfo, now *[]rtid2ti
	mu    sync.Mutex
	tags  []string
}

// NewTypeInfos creates a TypeInfos given a set of struct tags keys.
//
// This allows users customize the struct tag keys which contain configuration
// of their types.
func NewTypeInfos(tags []string) *TypeInfos {
	return &TypeInfos{tags: tags}
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

func (x *TypeInfos) find(sp *[]rtid2ti, rtid uintptr) (idx int, ti *typeInfo) {
	// binary search. adapted from sort/search.go.
	// if sp == nil {
	// 	return -1, nil
	// }
	s := *sp
	h, i, j := 0, 0, len(s)
	for i < j {
		h = i + (j-i)/2
		if s[h].rtid < rtid {
			i = h + 1
		} else {
			j = h
		}
	}
	if i < len(s) && s[i].rtid == rtid {
		return i, s[i].ti
	}
	return i, nil
}

func (x *TypeInfos) get(rtid uintptr, rt reflect.Type) (pti *typeInfo) {
	sp := x.infos.load()
	var idx int
	if sp != nil {
		idx, pti = x.find(sp, rtid)
		if pti != nil {
			return
		}
	}

	// do not hold lock while computing this.
	// it may lead to duplication, but that's ok.
	ti := typeInfo{rt: rt, rtid: rtid}
	// ti.rv0 = reflect.Zero(rt)

	ti.numMeth = uint16(rt.NumMethod())
	var ok bool
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
		ti.baseId = rt2id(pt)
		ti.baseIndir = ptIndir
	}

	if rt.Kind() == reflect.Struct {
		var omitEmpty bool
		if f, ok := rt.FieldByName(structInfoFieldName); ok {
			siInfo := parseStructFieldInfo(structInfoFieldName, x.structTag(f.Tag))
			ti.toArray = siInfo.toArray
			omitEmpty = siInfo.omitEmpty
		}
		pp, pi := pool.tiLoad()
		pv := pi.(*typeInfoLoadArray)
		pv.etypes[0] = ti.baseId
		vv := typeInfoLoad{pv.fNames[:0], pv.encNames[:0], pv.etypes[:1], pv.sfis[:0]}
		x.rget(rt, rtid, omitEmpty, nil, &vv)
		ti.sfip, ti.sfi, ti.anyOmitEmpty = rgetResolveSFI(vv.sfis, pv.sfiidx[:0])
		pp.Put(pi)
	}
	// sfi = sfip

	var vs []rtid2ti
	x.mu.Lock()
	sp = x.infos.load()
	if sp == nil {
		pti = &ti
		vs = []rtid2ti{{rtid, pti}}
		x.infos.store(&vs)
	} else {
		idx, pti = x.find(sp, rtid)
		if pti == nil {
			s := *sp
			pti = &ti
			vs = make([]rtid2ti, len(s)+1)
			copy(vs, s[:idx])
			vs[idx] = rtid2ti{rtid, pti}
			copy(vs[idx+1:], s[idx:])
			x.infos.store(&vs)
		}
	}
	x.mu.Unlock()
	return
}

func (x *TypeInfos) rget(rt reflect.Type, rtid uintptr, omitEmpty bool,
	indexstack []uint16, pv *typeInfoLoad,
) {
	// Read up fields and store how to access the value.
	//
	// It uses go's rules for message selectors,
	// which say that the field with the shallowest depth is selected.
	//
	// Note: we consciously use slices, not a map, to simulate a set.
	//       Typically, types have < 16 fields,
	//       and iteration using equals is faster than maps there
	flen := rt.NumField()
	if flen > (1<<maxLevelsEmbedding - 1) {
		panic(fmt.Errorf("codec: types with more than %v fields are not supported - has %v fields", (1<<maxLevelsEmbedding - 1), flen))
	}
LOOP:
	for j, jlen := uint16(0), uint16(flen); j < jlen; j++ {
		f := rt.Field(int(j))
		fkind := f.Type.Kind()
		// skip if a func type, or is unexported, or structTag value == "-"
		switch fkind {
		case reflect.Func, reflect.Complex64, reflect.Complex128, reflect.UnsafePointer:
			continue LOOP
		}

		// if r1, _ := utf8.DecodeRuneInString(f.Name);
		// r1 == utf8.RuneError || !unicode.IsUpper(r1) {
		if f.PkgPath != "" && !f.Anonymous { // unexported, not embedded
			continue
		}
		stag := x.structTag(f.Tag)
		if stag == "-" {
			continue
		}
		var si *structFieldInfo
		// if anonymous and no struct tag (or it's blank),
		// and a struct (or pointer to struct), inline it.
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
					// if etypes contains this, don't call rget again (as fields are already seen here)
					ftid := rt2id(ft)
					// We cannot recurse forever, but we need to track other field depths.
					// So - we break if we see a type twice (not the first time).
					// This should be sufficient to handle an embedded type that refers to its
					// owning type, which then refers to its embedded type.
					processIt := true
					numk := 0
					for _, k := range pv.etypes {
						if k == ftid {
							numk++
							if numk == rgetMaxRecursion {
								processIt = false
								break
							}
						}
					}
					if processIt {
						pv.etypes = append(pv.etypes, ftid)
						indexstack2 := make([]uint16, len(indexstack)+1)
						copy(indexstack2, indexstack)
						indexstack2[len(indexstack)] = j
						// indexstack2 := append(append(make([]int, 0, len(indexstack)+4), indexstack...), j)
						x.rget(ft, ftid, omitEmpty, indexstack2, pv)
					}
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

		pv.fNames = append(pv.fNames, f.Name)

		if si == nil {
			si = parseStructFieldInfo(f.Name, stag)
		} else if si.encName == "" {
			si.encName = f.Name
		}
		si.fieldName = f.Name

		pv.encNames = append(pv.encNames, si.encName)

		// si.ikind = int(f.Type.Kind())
		if len(indexstack) > maxLevelsEmbedding-1 {
			panic(fmt.Errorf("codec: only supports up to %v depth of embedding - type has %v depth", maxLevelsEmbedding-1, len(indexstack)))
		}
		si.nis = uint8(len(indexstack)) + 1
		copy(si.is[:], indexstack)
		si.is[len(indexstack)] = j

		if omitEmpty {
			si.omitEmpty = true
		}
		pv.sfis = append(pv.sfis, si)
	}
}

// resolves the struct field info got from a call to rget.
// Returns a trimmed, unsorted and sorted []*structFieldInfo.
func rgetResolveSFI(x []*structFieldInfo, pv []sfiIdx) (y, z []*structFieldInfo, anyOmitEmpty bool) {
	var n int
	for i, v := range x {
		xn := v.encName // TODO: fieldName or encName? use encName for now.
		var found bool
		for j, k := range pv {
			if k.name == xn {
				// one of them must be reset to nil, and the index updated appropriately to the other one
				if v.nis == x[k.index].nis {
				} else if v.nis < x[k.index].nis {
					pv[j].index = i
					if x[k.index] != nil {
						x[k.index] = nil
						n++
					}
				} else {
					if x[i] != nil {
						x[i] = nil
						n++
					}
				}
				found = true
				break
			}
		}
		if !found {
			pv = append(pv, sfiIdx{xn, i})
		}
	}

	// remove all the nils
	y = make([]*structFieldInfo, len(x)-n)
	n = 0
	for _, v := range x {
		if v == nil {
			continue
		}
		if !anyOmitEmpty && v.omitEmpty {
			anyOmitEmpty = true
		}
		y[n] = v
		n++
	}

	z = make([]*structFieldInfo, len(y))
	copy(z, y)
	sort.Sort(sfiSortedByEncName(z))
	return
}

func xprintf(format string, a ...interface{}) {
	if xDebug {
		fmt.Fprintf(os.Stderr, format, a...)
	}
}

func panicToErr(err *error) {
	if recoverPanicToErr {
		if x := recover(); x != nil {
			// if false && xDebug {
			// 	fmt.Printf("panic'ing with: %v\n", x)
			// 	debug.PrintStack()
			// }
			panicValToErr(x, err)
		}
	}
}

func panicToErrs2(err1, err2 *error) {
	if recoverPanicToErr {
		if x := recover(); x != nil {
			panicValToErr(x, err1)
			panicValToErr(x, err2)
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
	return immutableKindsSet[k]
	// return false ||
	// 	k == reflect.Int ||
	// 	k == reflect.Int8 ||
	// 	k == reflect.Int16 ||
	// 	k == reflect.Int32 ||
	// 	k == reflect.Int64 ||
	// 	k == reflect.Uint ||
	// 	k == reflect.Uint8 ||
	// 	k == reflect.Uint16 ||
	// 	k == reflect.Uint32 ||
	// 	k == reflect.Uint64 ||
	// 	k == reflect.Uintptr ||
	// 	k == reflect.Float32 ||
	// 	k == reflect.Float64 ||
	// 	k == reflect.Bool ||
	// 	k == reflect.String
}

// ----

type codecFnInfo struct {
	ti    *typeInfo
	xfFn  Ext
	xfTag uint64
	seq   seqType
	addr  bool
}

// codecFn encapsulates the captured variables and the encode function.
// This way, we only do some calculations one times, and pass to the
// code block that should be called (encapsulated in a function)
// instead of executing the checks every time.
type codecFn struct {
	i  codecFnInfo
	fe func(*Encoder, *codecFnInfo, reflect.Value)
	fd func(*Decoder, *codecFnInfo, reflect.Value)
}

type codecRtidFn struct {
	rtid uintptr
	fn   codecFn
}

type codecFner struct {
	hh Handle
	h  *BasicHandle
	cs [arrayCacheLen]*[arrayCacheLen]codecRtidFn
	s  []*[arrayCacheLen]codecRtidFn
	sn uint32
	be bool
	js bool
	cf [arrayCacheLen]codecRtidFn
}

func (c *codecFner) reset(hh Handle) {
	c.hh = hh
	c.h = hh.getBasicHandle()
	_, c.js = hh.(*JsonHandle)
	c.be = hh.isBinary()
}

func (c *codecFner) get(rt reflect.Type, checkFastpath, checkCodecSelfer bool) (fn *codecFn) {
	rtid := rt2id(rt)
	var j uint32
	var sn uint32 = c.sn
	if sn == 0 {
		c.s = c.cs[:1]
		c.s[0] = &c.cf
		c.cf[0].rtid = rtid
		fn = &(c.cf[0].fn)
		c.sn = 1
	} else {
	LOOP1:
		for _, x := range c.s {
			for i := range x {
				if j == sn {
					break LOOP1
				}
				if x[i].rtid == rtid {
					fn = &(x[i].fn)
					return
				}
				j++
			}
		}
		sx, sy := sn/arrayCacheLen, sn%arrayCacheLen
		if sy == 0 {
			c.s = append(c.s, &[arrayCacheLen]codecRtidFn{})
		}
		c.s[sx][sy].rtid = rtid
		fn = &(c.s[sx][sy].fn)
		c.sn++
	}

	ti := c.h.getTypeInfo(rtid, rt)
	fi := &(fn.i)
	fi.ti = ti

	if checkCodecSelfer && ti.cs {
		fn.fe = (*Encoder).selferMarshal
		fn.fd = (*Decoder).selferUnmarshal
	} else if rtid == rawTypId {
		fn.fe = (*Encoder).raw
		fn.fd = (*Decoder).raw
	} else if rtid == rawExtTypId {
		fn.fe = (*Encoder).rawExt
		fn.fd = (*Decoder).rawExt
		fn.i.addr = true
	} else if c.hh.IsBuiltinType(rtid) {
		fn.fe = (*Encoder).builtin
		fn.fd = (*Decoder).builtin
		fn.i.addr = true
	} else if xfFn := c.h.getExt(rtid); xfFn != nil {
		fi.xfTag, fi.xfFn = xfFn.tag, xfFn.ext
		fn.fe = (*Encoder).ext
		fn.fd = (*Decoder).ext
		fn.i.addr = true
	} else if supportMarshalInterfaces && c.be && ti.bm {
		fn.fe = (*Encoder).binaryMarshal
		fn.fd = (*Decoder).binaryUnmarshal
	} else if supportMarshalInterfaces && !c.be && c.js && ti.jm {
		//If JSON, we should check JSONMarshal before textMarshal
		fn.fe = (*Encoder).jsonMarshal
		fn.fd = (*Decoder).jsonUnmarshal
	} else if supportMarshalInterfaces && !c.be && ti.tm {
		fn.fe = (*Encoder).textMarshal
		fn.fd = (*Decoder).textUnmarshal
	} else {
		rk := rt.Kind()
		if fastpathEnabled && checkFastpath && (rk == reflect.Map || rk == reflect.Slice) {
			if rt.PkgPath() == "" { // un-named slice or map
				if idx := fastpathAV.index(rtid); idx != -1 {
					fn.fe = fastpathAV[idx].encfn
					fn.fd = fastpathAV[idx].decfn
					fn.i.addr = true
				}
			} else {
				// use mapping for underlying type if there
				var rtu reflect.Type
				if rk == reflect.Map {
					rtu = reflect.MapOf(rt.Key(), rt.Elem())
				} else {
					rtu = reflect.SliceOf(rt.Elem())
				}
				rtuid := rt2id(rtu)
				if idx := fastpathAV.index(rtuid); idx != -1 {
					xfnf := fastpathAV[idx].encfn
					xrt := fastpathAV[idx].rt
					fn.fe = func(e *Encoder, xf *codecFnInfo, xrv reflect.Value) {
						xfnf(e, xf, xrv.Convert(xrt))
					}
					fn.i.addr = true
					xfnf2 := fastpathAV[idx].decfn
					fn.fd = func(d *Decoder, xf *codecFnInfo, xrv reflect.Value) {
						xfnf2(d, xf, xrv.Convert(reflect.PtrTo(xrt)))
					}
				}
			}
		}
		if fn.fe == nil && fn.fd == nil {
			switch rk {
			case reflect.Bool:
				fn.fe = (*Encoder).kBool
				fn.fd = (*Decoder).kBool
			case reflect.String:
				fn.fe = (*Encoder).kString
				fn.fd = (*Decoder).kString
			case reflect.Int:
				fn.fd = (*Decoder).kInt
				fn.fe = (*Encoder).kInt
			case reflect.Int8:
				fn.fe = (*Encoder).kInt
				fn.fd = (*Decoder).kInt8
			case reflect.Int16:
				fn.fe = (*Encoder).kInt
				fn.fd = (*Decoder).kInt16
			case reflect.Int32:
				fn.fe = (*Encoder).kInt
				fn.fd = (*Decoder).kInt32
			case reflect.Int64:
				fn.fe = (*Encoder).kInt
				fn.fd = (*Decoder).kInt64
			case reflect.Uint:
				fn.fd = (*Decoder).kUint
				fn.fe = (*Encoder).kUint
			case reflect.Uint8:
				fn.fe = (*Encoder).kUint
				fn.fd = (*Decoder).kUint8
			case reflect.Uint16:
				fn.fe = (*Encoder).kUint
				fn.fd = (*Decoder).kUint16
			case reflect.Uint32:
				fn.fe = (*Encoder).kUint
				fn.fd = (*Decoder).kUint32
			case reflect.Uint64:
				fn.fe = (*Encoder).kUint
				fn.fd = (*Decoder).kUint64
				// case reflect.Ptr:
				// 	fn.fd = (*Decoder).kPtr
			case reflect.Uintptr:
				fn.fe = (*Encoder).kUint
				fn.fd = (*Decoder).kUintptr
			case reflect.Float32:
				fn.fe = (*Encoder).kFloat32
				fn.fd = (*Decoder).kFloat32
			case reflect.Float64:
				fn.fe = (*Encoder).kFloat64
				fn.fd = (*Decoder).kFloat64
			case reflect.Invalid:
				fn.fe = (*Encoder).kInvalid
			case reflect.Chan:
				fi.seq = seqTypeChan
				fn.fe = (*Encoder).kSlice
				fn.fd = (*Decoder).kSlice
			case reflect.Slice:
				fi.seq = seqTypeSlice
				fn.fe = (*Encoder).kSlice
				fn.fd = (*Decoder).kSlice
			case reflect.Array:
				fi.seq = seqTypeArray
				fn.fe = (*Encoder).kSlice
				fi.addr = false
				rt2 := reflect.SliceOf(rt.Elem())
				fn.fd = func(d *Decoder, xf *codecFnInfo, xrv reflect.Value) {
					// println(">>>>>> decoding an array ... ")
					d.cf.get(rt2, true, false).fd(d, xf, xrv.Slice(0, xrv.Len()))
					// println(">>>>>> decoding an array ... DONE")
				}
				// fn.fd = (*Decoder).kArray
			case reflect.Struct:
				if ti.anyOmitEmpty {
					fn.fe = (*Encoder).kStruct
				} else {
					fn.fe = (*Encoder).kStructNoOmitempty
				}
				fn.fd = (*Decoder).kStruct
				// reflect.Ptr and reflect.Interface are handled already by preEncodeValue
				// case reflect.Ptr:
				// 	fn.fe = (*Encoder).kPtr
				// case reflect.Interface:
				// 	fn.fe = (*Encoder).kInterface
			case reflect.Map:
				fn.fe = (*Encoder).kMap
				fn.fd = (*Decoder).kMap
			case reflect.Interface:
				// encode: reflect.Interface are handled already by preEncodeValue
				fn.fd = (*Decoder).kInterface
			default:
				fn.fe = (*Encoder).kErr
				fn.fd = (*Decoder).kErr
			}
		}
	}

	return
}

// ----

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
type uintptrSlice []uintptr
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

func (p uintptrSlice) Len() int           { return len(p) }
func (p uintptrSlice) Less(i, j int) bool { return p[i] < p[j] }
func (p uintptrSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

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

// ------

// bitset types are better than [256]bool, because they permit the whole
// bitset array being on a single cache line and use less memory.

// given x > 0 and n > 0 and x is exactly 2^n, then pos/x === pos>>n AND pos%x === pos&(x-1).
// consequently, pos/32 === pos>>5, pos/16 === pos>>4, pos/8 === pos>>3, pos%8 == pos&7

type bitset256 [32]byte

func (x *bitset256) set(pos byte) {
	x[pos>>3] |= (1 << (pos & 7))
}
func (x *bitset256) unset(pos byte) {
	x[pos>>3] &^= (1 << (pos & 7))
}
func (x *bitset256) isset(pos byte) bool {
	return x[pos>>3]&(1<<(pos&7)) != 0
}

type bitset128 [16]byte

func (x *bitset128) set(pos byte) {
	x[pos>>3] |= (1 << (pos & 7))
}
func (x *bitset128) unset(pos byte) {
	x[pos>>3] &^= (1 << (pos & 7))
}
func (x *bitset128) isset(pos byte) bool {
	return x[pos>>3]&(1<<(pos&7)) != 0
}

// ------------

type pooler struct {
	// for stringRV
	strRv8, strRv16, strRv32, strRv64, strRv128 sync.Pool
	// for the decNaked
	dn     sync.Pool
	tiload sync.Pool
}

func (p *pooler) init() {
	p.strRv8.New = func() interface{} { return new([8]stringRv) }
	p.strRv16.New = func() interface{} { return new([16]stringRv) }
	p.strRv32.New = func() interface{} { return new([32]stringRv) }
	p.strRv64.New = func() interface{} { return new([64]stringRv) }
	p.strRv128.New = func() interface{} { return new([128]stringRv) }
	p.dn.New = func() interface{} { x := new(decNaked); x.init(); return x }
	p.tiload.New = func() interface{} { return new(typeInfoLoadArray) }
}

func (p *pooler) stringRv8() (sp *sync.Pool, v interface{}) {
	return &p.strRv8, p.strRv8.Get()
}
func (p *pooler) stringRv16() (sp *sync.Pool, v interface{}) {
	return &p.strRv16, p.strRv16.Get()
}
func (p *pooler) stringRv32() (sp *sync.Pool, v interface{}) {
	return &p.strRv32, p.strRv32.Get()
}
func (p *pooler) stringRv64() (sp *sync.Pool, v interface{}) {
	return &p.strRv64, p.strRv64.Get()
}
func (p *pooler) stringRv128() (sp *sync.Pool, v interface{}) {
	return &p.strRv128, p.strRv128.Get()
}
func (p *pooler) decNaked() (sp *sync.Pool, v interface{}) {
	return &p.dn, p.dn.Get()
}
func (p *pooler) tiLoad() (sp *sync.Pool, v interface{}) {
	return &p.tiload, p.tiload.Get()
}
