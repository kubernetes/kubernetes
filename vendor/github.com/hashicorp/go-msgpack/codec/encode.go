// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

import (
	"io"
	"reflect"
)

const (
	// Some tagging information for error messages.
	msgTagEnc         = "codec.encoder"
	defEncByteBufSize = 1 << 6 // 4:16, 6:64, 8:256, 10:1024
	// maxTimeSecs32 = math.MaxInt32 / 60 / 24 / 366
)

// AsSymbolFlag defines what should be encoded as symbols.
type AsSymbolFlag uint8

const (
	// AsSymbolDefault is default.
	// Currently, this means only encode struct field names as symbols.
	// The default is subject to change.
	AsSymbolDefault AsSymbolFlag = iota

	// AsSymbolAll means encode anything which could be a symbol as a symbol.
	AsSymbolAll = 0xfe

	// AsSymbolNone means do not encode anything as a symbol.
	AsSymbolNone = 1 << iota

	// AsSymbolMapStringKeys means encode keys in map[string]XXX as symbols.
	AsSymbolMapStringKeysFlag

	// AsSymbolStructFieldName means encode struct field names as symbols.
	AsSymbolStructFieldNameFlag
)

// encWriter abstracting writing to a byte array or to an io.Writer.
type encWriter interface {
	writeUint16(uint16)
	writeUint32(uint32)
	writeUint64(uint64)
	writeb([]byte)
	writestr(string)
	writen1(byte)
	writen2(byte, byte)
	atEndOfEncode()
}

// encDriver abstracts the actual codec (binc vs msgpack, etc)
type encDriver interface {
	isBuiltinType(rt uintptr) bool
	encodeBuiltin(rt uintptr, v interface{})
	encodeNil()
	encodeInt(i int64)
	encodeUint(i uint64)
	encodeBool(b bool)
	encodeFloat32(f float32)
	encodeFloat64(f float64)
	encodeExtPreamble(xtag byte, length int)
	encodeArrayPreamble(length int)
	encodeMapPreamble(length int)
	encodeString(c charEncoding, v string)
	encodeSymbol(v string)
	encodeStringBytes(c charEncoding, v []byte)
	//TODO
	//encBignum(f *big.Int)
	//encStringRunes(c charEncoding, v []rune)
}

type ioEncWriterWriter interface {
	WriteByte(c byte) error
	WriteString(s string) (n int, err error)
	Write(p []byte) (n int, err error)
}

type ioEncStringWriter interface {
	WriteString(s string) (n int, err error)
}

type EncodeOptions struct {
	// Encode a struct as an array, and not as a map.
	StructToArray bool

	// AsSymbols defines what should be encoded as symbols.
	//
	// Encoding as symbols can reduce the encoded size significantly.
	//
	// However, during decoding, each string to be encoded as a symbol must
	// be checked to see if it has been seen before. Consequently, encoding time
	// will increase if using symbols, because string comparisons has a clear cost.
	//
	// Sample values:
	//   AsSymbolNone
	//   AsSymbolAll
	//   AsSymbolMapStringKeys
	//   AsSymbolMapStringKeysFlag | AsSymbolStructFieldNameFlag
	AsSymbols AsSymbolFlag
}

// ---------------------------------------------

type simpleIoEncWriterWriter struct {
	w  io.Writer
	bw io.ByteWriter
	sw ioEncStringWriter
}

func (o *simpleIoEncWriterWriter) WriteByte(c byte) (err error) {
	if o.bw != nil {
		return o.bw.WriteByte(c)
	}
	_, err = o.w.Write([]byte{c})
	return
}

func (o *simpleIoEncWriterWriter) WriteString(s string) (n int, err error) {
	if o.sw != nil {
		return o.sw.WriteString(s)
	}
	return o.w.Write([]byte(s))
}

func (o *simpleIoEncWriterWriter) Write(p []byte) (n int, err error) {
	return o.w.Write(p)
}

// ----------------------------------------

// ioEncWriter implements encWriter and can write to an io.Writer implementation
type ioEncWriter struct {
	w ioEncWriterWriter
	x [8]byte // temp byte array re-used internally for efficiency
}

func (z *ioEncWriter) writeUint16(v uint16) {
	bigen.PutUint16(z.x[:2], v)
	z.writeb(z.x[:2])
}

func (z *ioEncWriter) writeUint32(v uint32) {
	bigen.PutUint32(z.x[:4], v)
	z.writeb(z.x[:4])
}

func (z *ioEncWriter) writeUint64(v uint64) {
	bigen.PutUint64(z.x[:8], v)
	z.writeb(z.x[:8])
}

func (z *ioEncWriter) writeb(bs []byte) {
	if len(bs) == 0 {
		return
	}
	n, err := z.w.Write(bs)
	if err != nil {
		panic(err)
	}
	if n != len(bs) {
		encErr("write: Incorrect num bytes written. Expecting: %v, Wrote: %v", len(bs), n)
	}
}

func (z *ioEncWriter) writestr(s string) {
	n, err := z.w.WriteString(s)
	if err != nil {
		panic(err)
	}
	if n != len(s) {
		encErr("write: Incorrect num bytes written. Expecting: %v, Wrote: %v", len(s), n)
	}
}

func (z *ioEncWriter) writen1(b byte) {
	if err := z.w.WriteByte(b); err != nil {
		panic(err)
	}
}

func (z *ioEncWriter) writen2(b1 byte, b2 byte) {
	z.writen1(b1)
	z.writen1(b2)
}

func (z *ioEncWriter) atEndOfEncode() {}

// ----------------------------------------

// bytesEncWriter implements encWriter and can write to an byte slice.
// It is used by Marshal function.
type bytesEncWriter struct {
	b   []byte
	c   int     // cursor
	out *[]byte // write out on atEndOfEncode
}

func (z *bytesEncWriter) writeUint16(v uint16) {
	c := z.grow(2)
	z.b[c] = byte(v >> 8)
	z.b[c+1] = byte(v)
}

func (z *bytesEncWriter) writeUint32(v uint32) {
	c := z.grow(4)
	z.b[c] = byte(v >> 24)
	z.b[c+1] = byte(v >> 16)
	z.b[c+2] = byte(v >> 8)
	z.b[c+3] = byte(v)
}

func (z *bytesEncWriter) writeUint64(v uint64) {
	c := z.grow(8)
	z.b[c] = byte(v >> 56)
	z.b[c+1] = byte(v >> 48)
	z.b[c+2] = byte(v >> 40)
	z.b[c+3] = byte(v >> 32)
	z.b[c+4] = byte(v >> 24)
	z.b[c+5] = byte(v >> 16)
	z.b[c+6] = byte(v >> 8)
	z.b[c+7] = byte(v)
}

func (z *bytesEncWriter) writeb(s []byte) {
	if len(s) == 0 {
		return
	}
	c := z.grow(len(s))
	copy(z.b[c:], s)
}

func (z *bytesEncWriter) writestr(s string) {
	c := z.grow(len(s))
	copy(z.b[c:], s)
}

func (z *bytesEncWriter) writen1(b1 byte) {
	c := z.grow(1)
	z.b[c] = b1
}

func (z *bytesEncWriter) writen2(b1 byte, b2 byte) {
	c := z.grow(2)
	z.b[c] = b1
	z.b[c+1] = b2
}

func (z *bytesEncWriter) atEndOfEncode() {
	*(z.out) = z.b[:z.c]
}

func (z *bytesEncWriter) grow(n int) (oldcursor int) {
	oldcursor = z.c
	z.c = oldcursor + n
	if z.c > cap(z.b) {
		// Tried using appendslice logic: (if cap < 1024, *2, else *1.25).
		// However, it was too expensive, causing too many iterations of copy.
		// Using bytes.Buffer model was much better (2*cap + n)
		bs := make([]byte, 2*cap(z.b)+n)
		copy(bs, z.b[:oldcursor])
		z.b = bs
	} else if z.c > len(z.b) {
		z.b = z.b[:cap(z.b)]
	}
	return
}

// ---------------------------------------------

type encFnInfo struct {
	ti    *typeInfo
	e     *Encoder
	ee    encDriver
	xfFn  func(reflect.Value) ([]byte, error)
	xfTag byte
}

func (f *encFnInfo) builtin(rv reflect.Value) {
	f.ee.encodeBuiltin(f.ti.rtid, rv.Interface())
}

func (f *encFnInfo) rawExt(rv reflect.Value) {
	f.e.encRawExt(rv.Interface().(RawExt))
}

func (f *encFnInfo) ext(rv reflect.Value) {
	bs, fnerr := f.xfFn(rv)
	if fnerr != nil {
		panic(fnerr)
	}
	if bs == nil {
		f.ee.encodeNil()
		return
	}
	if f.e.hh.writeExt() {
		f.ee.encodeExtPreamble(f.xfTag, len(bs))
		f.e.w.writeb(bs)
	} else {
		f.ee.encodeStringBytes(c_RAW, bs)
	}

}

func (f *encFnInfo) binaryMarshal(rv reflect.Value) {
	var bm binaryMarshaler
	if f.ti.mIndir == 0 {
		bm = rv.Interface().(binaryMarshaler)
	} else if f.ti.mIndir == -1 {
		bm = rv.Addr().Interface().(binaryMarshaler)
	} else {
		for j, k := int8(0), f.ti.mIndir; j < k; j++ {
			if rv.IsNil() {
				f.ee.encodeNil()
				return
			}
			rv = rv.Elem()
		}
		bm = rv.Interface().(binaryMarshaler)
	}
	// debugf(">>>> binaryMarshaler: %T", rv.Interface())
	bs, fnerr := bm.MarshalBinary()
	if fnerr != nil {
		panic(fnerr)
	}
	if bs == nil {
		f.ee.encodeNil()
	} else {
		f.ee.encodeStringBytes(c_RAW, bs)
	}
}

func (f *encFnInfo) kBool(rv reflect.Value) {
	f.ee.encodeBool(rv.Bool())
}

func (f *encFnInfo) kString(rv reflect.Value) {
	f.ee.encodeString(c_UTF8, rv.String())
}

func (f *encFnInfo) kFloat64(rv reflect.Value) {
	f.ee.encodeFloat64(rv.Float())
}

func (f *encFnInfo) kFloat32(rv reflect.Value) {
	f.ee.encodeFloat32(float32(rv.Float()))
}

func (f *encFnInfo) kInt(rv reflect.Value) {
	f.ee.encodeInt(rv.Int())
}

func (f *encFnInfo) kUint(rv reflect.Value) {
	f.ee.encodeUint(rv.Uint())
}

func (f *encFnInfo) kInvalid(rv reflect.Value) {
	f.ee.encodeNil()
}

func (f *encFnInfo) kErr(rv reflect.Value) {
	encErr("Unsupported kind: %s, for: %#v", rv.Kind(), rv)
}

func (f *encFnInfo) kSlice(rv reflect.Value) {
	if rv.IsNil() {
		f.ee.encodeNil()
		return
	}

	if shortCircuitReflectToFastPath {
		switch f.ti.rtid {
		case intfSliceTypId:
			f.e.encSliceIntf(rv.Interface().([]interface{}))
			return
		case strSliceTypId:
			f.e.encSliceStr(rv.Interface().([]string))
			return
		case uint64SliceTypId:
			f.e.encSliceUint64(rv.Interface().([]uint64))
			return
		case int64SliceTypId:
			f.e.encSliceInt64(rv.Interface().([]int64))
			return
		}
	}

	// If in this method, then there was no extension function defined.
	// So it's okay to treat as []byte.
	if f.ti.rtid == uint8SliceTypId || f.ti.rt.Elem().Kind() == reflect.Uint8 {
		f.ee.encodeStringBytes(c_RAW, rv.Bytes())
		return
	}

	l := rv.Len()
	if f.ti.mbs {
		if l%2 == 1 {
			encErr("mapBySlice: invalid length (must be divisible by 2): %v", l)
		}
		f.ee.encodeMapPreamble(l / 2)
	} else {
		f.ee.encodeArrayPreamble(l)
	}
	if l == 0 {
		return
	}
	for j := 0; j < l; j++ {
		// TODO: Consider perf implication of encoding odd index values as symbols if type is string
		f.e.encodeValue(rv.Index(j))
	}
}

func (f *encFnInfo) kArray(rv reflect.Value) {
	// We cannot share kSlice method, because the array may be non-addressable.
	// E.g. type struct S{B [2]byte}; Encode(S{}) will bomb on "panic: slice of unaddressable array".
	// So we have to duplicate the functionality here.
	// f.e.encodeValue(rv.Slice(0, rv.Len()))
	// f.kSlice(rv.Slice(0, rv.Len()))

	l := rv.Len()
	// Handle an array of bytes specially (in line with what is done for slices)
	if f.ti.rt.Elem().Kind() == reflect.Uint8 {
		if l == 0 {
			f.ee.encodeStringBytes(c_RAW, nil)
			return
		}
		var bs []byte
		if rv.CanAddr() {
			bs = rv.Slice(0, l).Bytes()
		} else {
			bs = make([]byte, l)
			for i := 0; i < l; i++ {
				bs[i] = byte(rv.Index(i).Uint())
			}
		}
		f.ee.encodeStringBytes(c_RAW, bs)
		return
	}

	if f.ti.mbs {
		if l%2 == 1 {
			encErr("mapBySlice: invalid length (must be divisible by 2): %v", l)
		}
		f.ee.encodeMapPreamble(l / 2)
	} else {
		f.ee.encodeArrayPreamble(l)
	}
	if l == 0 {
		return
	}
	for j := 0; j < l; j++ {
		// TODO: Consider perf implication of encoding odd index values as symbols if type is string
		f.e.encodeValue(rv.Index(j))
	}
}

func (f *encFnInfo) kStruct(rv reflect.Value) {
	fti := f.ti
	newlen := len(fti.sfi)
	rvals := make([]reflect.Value, newlen)
	var encnames []string
	e := f.e
	tisfi := fti.sfip
	toMap := !(fti.toArray || e.h.StructToArray)
	// if toMap, use the sorted array. If toArray, use unsorted array (to match sequence in struct)
	if toMap {
		tisfi = fti.sfi
		encnames = make([]string, newlen)
	}
	newlen = 0
	for _, si := range tisfi {
		if si.i != -1 {
			rvals[newlen] = rv.Field(int(si.i))
		} else {
			rvals[newlen] = rv.FieldByIndex(si.is)
		}
		if toMap {
			if si.omitEmpty && isEmptyValue(rvals[newlen]) {
				continue
			}
			encnames[newlen] = si.encName
		} else {
			if si.omitEmpty && isEmptyValue(rvals[newlen]) {
				rvals[newlen] = reflect.Value{} //encode as nil
			}
		}
		newlen++
	}

	// debugf(">>>> kStruct: newlen: %v", newlen)
	if toMap {
		ee := f.ee //don't dereference everytime
		ee.encodeMapPreamble(newlen)
		// asSymbols := e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		asSymbols := e.h.AsSymbols == AsSymbolDefault || e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		for j := 0; j < newlen; j++ {
			if asSymbols {
				ee.encodeSymbol(encnames[j])
			} else {
				ee.encodeString(c_UTF8, encnames[j])
			}
			e.encodeValue(rvals[j])
		}
	} else {
		f.ee.encodeArrayPreamble(newlen)
		for j := 0; j < newlen; j++ {
			e.encodeValue(rvals[j])
		}
	}
}

// func (f *encFnInfo) kPtr(rv reflect.Value) {
// 	debugf(">>>>>>> ??? encode kPtr called - shouldn't get called")
// 	if rv.IsNil() {
// 		f.ee.encodeNil()
// 		return
// 	}
// 	f.e.encodeValue(rv.Elem())
// }

func (f *encFnInfo) kInterface(rv reflect.Value) {
	if rv.IsNil() {
		f.ee.encodeNil()
		return
	}
	f.e.encodeValue(rv.Elem())
}

func (f *encFnInfo) kMap(rv reflect.Value) {
	if rv.IsNil() {
		f.ee.encodeNil()
		return
	}

	if shortCircuitReflectToFastPath {
		switch f.ti.rtid {
		case mapIntfIntfTypId:
			f.e.encMapIntfIntf(rv.Interface().(map[interface{}]interface{}))
			return
		case mapStrIntfTypId:
			f.e.encMapStrIntf(rv.Interface().(map[string]interface{}))
			return
		case mapStrStrTypId:
			f.e.encMapStrStr(rv.Interface().(map[string]string))
			return
		case mapInt64IntfTypId:
			f.e.encMapInt64Intf(rv.Interface().(map[int64]interface{}))
			return
		case mapUint64IntfTypId:
			f.e.encMapUint64Intf(rv.Interface().(map[uint64]interface{}))
			return
		}
	}

	l := rv.Len()
	f.ee.encodeMapPreamble(l)
	if l == 0 {
		return
	}
	// keyTypeIsString := f.ti.rt.Key().Kind() == reflect.String
	keyTypeIsString := f.ti.rt.Key() == stringTyp
	var asSymbols bool
	if keyTypeIsString {
		asSymbols = f.e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	}
	mks := rv.MapKeys()
	// for j, lmks := 0, len(mks); j < lmks; j++ {
	for j := range mks {
		if keyTypeIsString {
			if asSymbols {
				f.ee.encodeSymbol(mks[j].String())
			} else {
				f.ee.encodeString(c_UTF8, mks[j].String())
			}
		} else {
			f.e.encodeValue(mks[j])
		}
		f.e.encodeValue(rv.MapIndex(mks[j]))
	}

}

// --------------------------------------------------

// encFn encapsulates the captured variables and the encode function.
// This way, we only do some calculations one times, and pass to the
// code block that should be called (encapsulated in a function)
// instead of executing the checks every time.
type encFn struct {
	i *encFnInfo
	f func(*encFnInfo, reflect.Value)
}

// --------------------------------------------------

// An Encoder writes an object to an output stream in the codec format.
type Encoder struct {
	w  encWriter
	e  encDriver
	h  *BasicHandle
	hh Handle
	f  map[uintptr]encFn
	x  []uintptr
	s  []encFn
}

// NewEncoder returns an Encoder for encoding into an io.Writer.
//
// For efficiency, Users are encouraged to pass in a memory buffered writer
// (eg bufio.Writer, bytes.Buffer).
func NewEncoder(w io.Writer, h Handle) *Encoder {
	ww, ok := w.(ioEncWriterWriter)
	if !ok {
		sww := simpleIoEncWriterWriter{w: w}
		sww.bw, _ = w.(io.ByteWriter)
		sww.sw, _ = w.(ioEncStringWriter)
		ww = &sww
		//ww = bufio.NewWriterSize(w, defEncByteBufSize)
	}
	z := ioEncWriter{
		w: ww,
	}
	return &Encoder{w: &z, hh: h, h: h.getBasicHandle(), e: h.newEncDriver(&z)}
}

// NewEncoderBytes returns an encoder for encoding directly and efficiently
// into a byte slice, using zero-copying to temporary slices.
//
// It will potentially replace the output byte slice pointed to.
// After encoding, the out parameter contains the encoded contents.
func NewEncoderBytes(out *[]byte, h Handle) *Encoder {
	in := *out
	if in == nil {
		in = make([]byte, defEncByteBufSize)
	}
	z := bytesEncWriter{
		b:   in,
		out: out,
	}
	return &Encoder{w: &z, hh: h, h: h.getBasicHandle(), e: h.newEncDriver(&z)}
}

// Encode writes an object into a stream in the codec format.
//
// Encoding can be configured via the "codec" struct tag for the fields.
//
// The "codec" key in struct field's tag value is the key name,
// followed by an optional comma and options.
//
// To set an option on all fields (e.g. omitempty on all fields), you
// can create a field called _struct, and set flags on it.
//
// Struct values "usually" encode as maps. Each exported struct field is encoded unless:
//    - the field's codec tag is "-", OR
//    - the field is empty and its codec tag specifies the "omitempty" option.
//
// When encoding as a map, the first string in the tag (before the comma)
// is the map key string to use when encoding.
//
// However, struct values may encode as arrays. This happens when:
//    - StructToArray Encode option is set, OR
//    - the codec tag on the _struct field sets the "toarray" option
//
// Values with types that implement MapBySlice are encoded as stream maps.
//
// The empty values (for omitempty option) are false, 0, any nil pointer
// or interface value, and any array, slice, map, or string of length zero.
//
// Anonymous fields are encoded inline if no struct tag is present.
// Else they are encoded as regular fields.
//
// Examples:
//
//      type MyStruct struct {
//          _struct bool    `codec:",omitempty"`   //set omitempty for every field
//          Field1 string   `codec:"-"`            //skip this field
//          Field2 int      `codec:"myName"`       //Use key "myName" in encode stream
//          Field3 int32    `codec:",omitempty"`   //use key "Field3". Omit if empty.
//          Field4 bool     `codec:"f4,omitempty"` //use key "f4". Omit if empty.
//          ...
//      }
//
//      type MyStruct struct {
//          _struct bool    `codec:",omitempty,toarray"`   //set omitempty for every field
//                                                         //and encode struct as an array
//      }
//
// The mode of encoding is based on the type of the value. When a value is seen:
//   - If an extension is registered for it, call that extension function
//   - If it implements BinaryMarshaler, call its MarshalBinary() (data []byte, err error)
//   - Else encode it based on its reflect.Kind
//
// Note that struct field names and keys in map[string]XXX will be treated as symbols.
// Some formats support symbols (e.g. binc) and will properly encode the string
// only once in the stream, and use a tag to refer to it thereafter.
func (e *Encoder) Encode(v interface{}) (err error) {
	defer panicToErr(&err)
	e.encode(v)
	e.w.atEndOfEncode()
	return
}

func (e *Encoder) encode(iv interface{}) {
	switch v := iv.(type) {
	case nil:
		e.e.encodeNil()

	case reflect.Value:
		e.encodeValue(v)

	case string:
		e.e.encodeString(c_UTF8, v)
	case bool:
		e.e.encodeBool(v)
	case int:
		e.e.encodeInt(int64(v))
	case int8:
		e.e.encodeInt(int64(v))
	case int16:
		e.e.encodeInt(int64(v))
	case int32:
		e.e.encodeInt(int64(v))
	case int64:
		e.e.encodeInt(v)
	case uint:
		e.e.encodeUint(uint64(v))
	case uint8:
		e.e.encodeUint(uint64(v))
	case uint16:
		e.e.encodeUint(uint64(v))
	case uint32:
		e.e.encodeUint(uint64(v))
	case uint64:
		e.e.encodeUint(v)
	case float32:
		e.e.encodeFloat32(v)
	case float64:
		e.e.encodeFloat64(v)

	case []interface{}:
		e.encSliceIntf(v)
	case []string:
		e.encSliceStr(v)
	case []int64:
		e.encSliceInt64(v)
	case []uint64:
		e.encSliceUint64(v)
	case []uint8:
		e.e.encodeStringBytes(c_RAW, v)

	case map[interface{}]interface{}:
		e.encMapIntfIntf(v)
	case map[string]interface{}:
		e.encMapStrIntf(v)
	case map[string]string:
		e.encMapStrStr(v)
	case map[int64]interface{}:
		e.encMapInt64Intf(v)
	case map[uint64]interface{}:
		e.encMapUint64Intf(v)

	case *string:
		e.e.encodeString(c_UTF8, *v)
	case *bool:
		e.e.encodeBool(*v)
	case *int:
		e.e.encodeInt(int64(*v))
	case *int8:
		e.e.encodeInt(int64(*v))
	case *int16:
		e.e.encodeInt(int64(*v))
	case *int32:
		e.e.encodeInt(int64(*v))
	case *int64:
		e.e.encodeInt(*v)
	case *uint:
		e.e.encodeUint(uint64(*v))
	case *uint8:
		e.e.encodeUint(uint64(*v))
	case *uint16:
		e.e.encodeUint(uint64(*v))
	case *uint32:
		e.e.encodeUint(uint64(*v))
	case *uint64:
		e.e.encodeUint(*v)
	case *float32:
		e.e.encodeFloat32(*v)
	case *float64:
		e.e.encodeFloat64(*v)

	case *[]interface{}:
		e.encSliceIntf(*v)
	case *[]string:
		e.encSliceStr(*v)
	case *[]int64:
		e.encSliceInt64(*v)
	case *[]uint64:
		e.encSliceUint64(*v)
	case *[]uint8:
		e.e.encodeStringBytes(c_RAW, *v)

	case *map[interface{}]interface{}:
		e.encMapIntfIntf(*v)
	case *map[string]interface{}:
		e.encMapStrIntf(*v)
	case *map[string]string:
		e.encMapStrStr(*v)
	case *map[int64]interface{}:
		e.encMapInt64Intf(*v)
	case *map[uint64]interface{}:
		e.encMapUint64Intf(*v)

	default:
		e.encodeValue(reflect.ValueOf(iv))
	}
}

func (e *Encoder) encodeValue(rv reflect.Value) {
	for rv.Kind() == reflect.Ptr {
		if rv.IsNil() {
			e.e.encodeNil()
			return
		}
		rv = rv.Elem()
	}

	rt := rv.Type()
	rtid := reflect.ValueOf(rt).Pointer()

	// if e.f == nil && e.s == nil { debugf("---->Creating new enc f map for type: %v\n", rt) }
	var fn encFn
	var ok bool
	if useMapForCodecCache {
		fn, ok = e.f[rtid]
	} else {
		for i, v := range e.x {
			if v == rtid {
				fn, ok = e.s[i], true
				break
			}
		}
	}
	if !ok {
		// debugf("\tCreating new enc fn for type: %v\n", rt)
		fi := encFnInfo{ti: getTypeInfo(rtid, rt), e: e, ee: e.e}
		fn.i = &fi
		if rtid == rawExtTypId {
			fn.f = (*encFnInfo).rawExt
		} else if e.e.isBuiltinType(rtid) {
			fn.f = (*encFnInfo).builtin
		} else if xfTag, xfFn := e.h.getEncodeExt(rtid); xfFn != nil {
			fi.xfTag, fi.xfFn = xfTag, xfFn
			fn.f = (*encFnInfo).ext
		} else if supportBinaryMarshal && fi.ti.m {
			fn.f = (*encFnInfo).binaryMarshal
		} else {
			switch rk := rt.Kind(); rk {
			case reflect.Bool:
				fn.f = (*encFnInfo).kBool
			case reflect.String:
				fn.f = (*encFnInfo).kString
			case reflect.Float64:
				fn.f = (*encFnInfo).kFloat64
			case reflect.Float32:
				fn.f = (*encFnInfo).kFloat32
			case reflect.Int, reflect.Int8, reflect.Int64, reflect.Int32, reflect.Int16:
				fn.f = (*encFnInfo).kInt
			case reflect.Uint8, reflect.Uint64, reflect.Uint, reflect.Uint32, reflect.Uint16:
				fn.f = (*encFnInfo).kUint
			case reflect.Invalid:
				fn.f = (*encFnInfo).kInvalid
			case reflect.Slice:
				fn.f = (*encFnInfo).kSlice
			case reflect.Array:
				fn.f = (*encFnInfo).kArray
			case reflect.Struct:
				fn.f = (*encFnInfo).kStruct
			// case reflect.Ptr:
			// 	fn.f = (*encFnInfo).kPtr
			case reflect.Interface:
				fn.f = (*encFnInfo).kInterface
			case reflect.Map:
				fn.f = (*encFnInfo).kMap
			default:
				fn.f = (*encFnInfo).kErr
			}
		}
		if useMapForCodecCache {
			if e.f == nil {
				e.f = make(map[uintptr]encFn, 16)
			}
			e.f[rtid] = fn
		} else {
			e.s = append(e.s, fn)
			e.x = append(e.x, rtid)
		}
	}

	fn.f(fn.i, rv)

}

func (e *Encoder) encRawExt(re RawExt) {
	if re.Data == nil {
		e.e.encodeNil()
		return
	}
	if e.hh.writeExt() {
		e.e.encodeExtPreamble(re.Tag, len(re.Data))
		e.w.writeb(re.Data)
	} else {
		e.e.encodeStringBytes(c_RAW, re.Data)
	}
}

// ---------------------------------------------
// short circuit functions for common maps and slices

func (e *Encoder) encSliceIntf(v []interface{}) {
	e.e.encodeArrayPreamble(len(v))
	for _, v2 := range v {
		e.encode(v2)
	}
}

func (e *Encoder) encSliceStr(v []string) {
	e.e.encodeArrayPreamble(len(v))
	for _, v2 := range v {
		e.e.encodeString(c_UTF8, v2)
	}
}

func (e *Encoder) encSliceInt64(v []int64) {
	e.e.encodeArrayPreamble(len(v))
	for _, v2 := range v {
		e.e.encodeInt(v2)
	}
}

func (e *Encoder) encSliceUint64(v []uint64) {
	e.e.encodeArrayPreamble(len(v))
	for _, v2 := range v {
		e.e.encodeUint(v2)
	}
}

func (e *Encoder) encMapStrStr(v map[string]string) {
	e.e.encodeMapPreamble(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			e.e.encodeSymbol(k2)
		} else {
			e.e.encodeString(c_UTF8, k2)
		}
		e.e.encodeString(c_UTF8, v2)
	}
}

func (e *Encoder) encMapStrIntf(v map[string]interface{}) {
	e.e.encodeMapPreamble(len(v))
	asSymbols := e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	for k2, v2 := range v {
		if asSymbols {
			e.e.encodeSymbol(k2)
		} else {
			e.e.encodeString(c_UTF8, k2)
		}
		e.encode(v2)
	}
}

func (e *Encoder) encMapInt64Intf(v map[int64]interface{}) {
	e.e.encodeMapPreamble(len(v))
	for k2, v2 := range v {
		e.e.encodeInt(k2)
		e.encode(v2)
	}
}

func (e *Encoder) encMapUint64Intf(v map[uint64]interface{}) {
	e.e.encodeMapPreamble(len(v))
	for k2, v2 := range v {
		e.e.encodeUint(uint64(k2))
		e.encode(v2)
	}
}

func (e *Encoder) encMapIntfIntf(v map[interface{}]interface{}) {
	e.e.encodeMapPreamble(len(v))
	for k2, v2 := range v {
		e.encode(k2)
		e.encode(v2)
	}
}

// ----------------------------------------

func encErr(format string, params ...interface{}) {
	doPanic(msgTagEnc, format, params...)
}
