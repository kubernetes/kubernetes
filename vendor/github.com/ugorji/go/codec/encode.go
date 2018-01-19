// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"bufio"
	"encoding"
	"fmt"
	"io"
	"reflect"
	"sort"
	"sync"
)

const defEncByteBufSize = 1 << 6 // 4:16, 6:64, 8:256, 10:1024

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

// encWriter abstracts writing to a byte array or to an io.Writer.
type encWriter interface {
	writeb([]byte)
	writestr(string)
	writen1(byte)
	writen2(byte, byte)
	writen4(byte, byte, byte, byte)
	writen5(byte, byte, byte, byte, byte)
	atEndOfEncode()
}

// encDriver abstracts the actual codec (binc vs msgpack, etc)
type encDriver interface {
	// IsBuiltinType(rt uintptr) bool
	EncodeBuiltin(rt uintptr, v interface{})
	EncodeNil()
	EncodeInt(i int64)
	EncodeUint(i uint64)
	EncodeBool(b bool)
	EncodeFloat32(f float32)
	EncodeFloat64(f float64)
	// encodeExtPreamble(xtag byte, length int)
	EncodeRawExt(re *RawExt, e *Encoder)
	EncodeExt(v interface{}, xtag uint64, ext Ext, e *Encoder)
	WriteArrayStart(length int)
	WriteArrayElem()
	WriteArrayEnd()
	WriteMapStart(length int)
	WriteMapElemKey()
	WriteMapElemValue()
	WriteMapEnd()
	EncodeString(c charEncoding, v string)
	EncodeSymbol(v string)
	EncodeStringBytes(c charEncoding, v []byte)
	//TODO
	//encBignum(f *big.Int)
	//encStringRunes(c charEncoding, v []rune)

	reset()
	atEndOfEncode()
}

type ioEncStringWriter interface {
	WriteString(s string) (n int, err error)
}

type ioEncFlusher interface {
	Flush() error
}

type encDriverAsis interface {
	EncodeAsis(v []byte)
}

// type encNoSeparator struct{}
// func (_ encNoSeparator) EncodeEnd() {}

type encDriverNoopContainerWriter struct{}

func (_ encDriverNoopContainerWriter) WriteArrayStart(length int) {}
func (_ encDriverNoopContainerWriter) WriteArrayElem()            {}
func (_ encDriverNoopContainerWriter) WriteArrayEnd()             {}
func (_ encDriverNoopContainerWriter) WriteMapStart(length int)   {}
func (_ encDriverNoopContainerWriter) WriteMapElemKey()           {}
func (_ encDriverNoopContainerWriter) WriteMapElemValue()         {}
func (_ encDriverNoopContainerWriter) WriteMapEnd()               {}
func (_ encDriverNoopContainerWriter) atEndOfEncode()             {}

// type ioEncWriterWriter interface {
// 	WriteByte(c byte) error
// 	WriteString(s string) (n int, err error)
// 	Write(p []byte) (n int, err error)
// }

type EncodeOptions struct {
	// Encode a struct as an array, and not as a map
	StructToArray bool

	// Canonical representation means that encoding a value will always result in the same
	// sequence of bytes.
	//
	// This only affects maps, as the iteration order for maps is random.
	//
	// The implementation MAY use the natural sort order for the map keys if possible:
	//
	//     - If there is a natural sort order (ie for number, bool, string or []byte keys),
	//       then the map keys are first sorted in natural order and then written
	//       with corresponding map values to the strema.
	//     - If there is no natural sort order, then the map keys will first be
	//       encoded into []byte, and then sorted,
	//       before writing the sorted keys and the corresponding map values to the stream.
	//
	Canonical bool

	// CheckCircularRef controls whether we check for circular references
	// and error fast during an encode.
	//
	// If enabled, an error is received if a pointer to a struct
	// references itself either directly or through one of its fields (iteratively).
	//
	// This is opt-in, as there may be a performance hit to checking circular references.
	CheckCircularRef bool

	// RecursiveEmptyCheck controls whether we descend into interfaces, structs and pointers
	// when checking if a value is empty.
	//
	// Note that this may make OmitEmpty more expensive, as it incurs a lot more reflect calls.
	RecursiveEmptyCheck bool

	// Raw controls whether we encode Raw values.
	// This is a "dangerous" option and must be explicitly set.
	// If set, we blindly encode Raw values as-is, without checking
	// if they are a correct representation of a value in that format.
	// If unset, we error out.
	Raw bool

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

	// WriterBufferSize is the size of the buffer used when writing.
	//
	// if > 0, we use a smart buffer internally for performance purposes.
	WriterBufferSize int
}

// ---------------------------------------------

type simpleIoEncWriter struct {
	io.Writer
}

// type bufIoEncWriter struct {
// 	w   io.Writer
// 	buf []byte
// 	err error
// }

// func (x *bufIoEncWriter) Write(b []byte) (n int, err error) {
// 	if x.err != nil {
// 		return 0, x.err
// 	}
// 	if cap(x.buf)-len(x.buf) >= len(b) {
// 		x.buf = append(x.buf, b)
// 		return len(b), nil
// 	}
// 	n, err = x.w.Write(x.buf)
// 	if err != nil {
// 		x.err = err
// 		return 0, x.err
// 	}
// 	n, err = x.w.Write(b)
// 	x.err = err
// 	return
// }

// ioEncWriter implements encWriter and can write to an io.Writer implementation
type ioEncWriter struct {
	w  io.Writer
	ww io.Writer
	bw io.ByteWriter
	sw ioEncStringWriter
	fw ioEncFlusher
	b  [8]byte
}

func (z *ioEncWriter) WriteByte(b byte) (err error) {
	// x.bs[0] = b
	// _, err = x.ww.Write(x.bs[:])
	z.b[0] = b
	_, err = z.w.Write(z.b[:1])
	return
}

func (z *ioEncWriter) WriteString(s string) (n int, err error) {
	return z.w.Write(bytesView(s))
}

func (z *ioEncWriter) writeb(bs []byte) {
	// if len(bs) == 0 {
	// 	return
	// }
	if _, err := z.ww.Write(bs); err != nil {
		panic(err)
	}
}

func (z *ioEncWriter) writestr(s string) {
	// if len(s) == 0 {
	// 	return
	// }
	if _, err := z.sw.WriteString(s); err != nil {
		panic(err)
	}
}

func (z *ioEncWriter) writen1(b byte) {
	if err := z.bw.WriteByte(b); err != nil {
		panic(err)
	}
}

func (z *ioEncWriter) writen2(b1, b2 byte) {
	var err error
	if err = z.bw.WriteByte(b1); err == nil {
		if err = z.bw.WriteByte(b2); err == nil {
			return
		}
	}
	panic(err)
}

func (z *ioEncWriter) writen4(b1, b2, b3, b4 byte) {
	z.b[0], z.b[1], z.b[2], z.b[3] = b1, b2, b3, b4
	if _, err := z.ww.Write(z.b[:4]); err != nil {
		panic(err)
	}
}

func (z *ioEncWriter) writen5(b1, b2, b3, b4, b5 byte) {
	z.b[0], z.b[1], z.b[2], z.b[3], z.b[4] = b1, b2, b3, b4, b5
	if _, err := z.ww.Write(z.b[:5]); err != nil {
		panic(err)
	}
}

func (z *ioEncWriter) atEndOfEncode() {
	if z.fw != nil {
		z.fw.Flush()
	}
}

// ----------------------------------------

// bytesEncWriter implements encWriter and can write to an byte slice.
// It is used by Marshal function.
type bytesEncWriter struct {
	b   []byte
	c   int     // cursor
	out *[]byte // write out on atEndOfEncode
}

func (z *bytesEncWriter) writeb(s []byte) {
	oc, a := z.growNoAlloc(len(s))
	if a {
		z.growAlloc(len(s), oc)
	}
	copy(z.b[oc:], s)
}

func (z *bytesEncWriter) writestr(s string) {
	oc, a := z.growNoAlloc(len(s))
	if a {
		z.growAlloc(len(s), oc)
	}
	copy(z.b[oc:], s)
}

func (z *bytesEncWriter) writen1(b1 byte) {
	oc, a := z.growNoAlloc(1)
	if a {
		z.growAlloc(1, oc)
	}
	z.b[oc] = b1
}

func (z *bytesEncWriter) writen2(b1, b2 byte) {
	oc, a := z.growNoAlloc(2)
	if a {
		z.growAlloc(2, oc)
	}
	z.b[oc+1] = b2
	z.b[oc] = b1
}

func (z *bytesEncWriter) writen4(b1, b2, b3, b4 byte) {
	oc, a := z.growNoAlloc(4)
	if a {
		z.growAlloc(4, oc)
	}
	z.b[oc+3] = b4
	z.b[oc+2] = b3
	z.b[oc+1] = b2
	z.b[oc] = b1
}

func (z *bytesEncWriter) writen5(b1, b2, b3, b4, b5 byte) {
	oc, a := z.growNoAlloc(5)
	if a {
		z.growAlloc(5, oc)
	}
	z.b[oc+4] = b5
	z.b[oc+3] = b4
	z.b[oc+2] = b3
	z.b[oc+1] = b2
	z.b[oc] = b1
}

func (z *bytesEncWriter) atEndOfEncode() {
	*(z.out) = z.b[:z.c]
}

// have a growNoalloc(n int), which can be inlined.
// if allocation is needed, then call growAlloc(n int)

func (z *bytesEncWriter) growNoAlloc(n int) (oldcursor int, allocNeeded bool) {
	oldcursor = z.c
	z.c = z.c + n
	if z.c > len(z.b) {
		if z.c > cap(z.b) {
			allocNeeded = true
		} else {
			z.b = z.b[:cap(z.b)]
		}
	}
	return
}

func (z *bytesEncWriter) growAlloc(n int, oldcursor int) {
	// appendslice logic (if cap < 1024, *2, else *1.25): more expensive. many copy calls.
	// bytes.Buffer model (2*cap + n): much better
	// bs := make([]byte, 2*cap(z.b)+n)
	bs := make([]byte, growCap(cap(z.b), 1, n))
	copy(bs, z.b[:oldcursor])
	z.b = bs
}

// ---------------------------------------------

func (e *Encoder) builtin(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeBuiltin(f.ti.rtid, rv2i(rv))
}

func (e *Encoder) raw(f *codecFnInfo, rv reflect.Value) {
	e.rawBytes(rv2i(rv).(Raw))
}

func (e *Encoder) rawExt(f *codecFnInfo, rv reflect.Value) {
	// rev := rv2i(rv).(RawExt)
	// e.e.EncodeRawExt(&rev, e)
	var re *RawExt
	if rv.CanAddr() {
		re = rv2i(rv.Addr()).(*RawExt)
	} else {
		rev := rv2i(rv).(RawExt)
		re = &rev
	}
	e.e.EncodeRawExt(re, e)
}

func (e *Encoder) ext(f *codecFnInfo, rv reflect.Value) {
	// if this is a struct|array and it was addressable, then pass the address directly (not the value)
	if k := rv.Kind(); (k == reflect.Struct || k == reflect.Array) && rv.CanAddr() {
		rv = rv.Addr()
	}
	e.e.EncodeExt(rv2i(rv), f.xfTag, f.xfFn, e)
}

func (e *Encoder) getValueForMarshalInterface(rv reflect.Value, indir int8) (v interface{}, proceed bool) {
	if indir == 0 {
		v = rv2i(rv)
	} else if indir == -1 {
		// If a non-pointer was passed to Encode(), then that value is not addressable.
		// Take addr if addressable, else copy value to an addressable value.
		if rv.CanAddr() {
			v = rv2i(rv.Addr())
		} else {
			rv2 := reflect.New(rv.Type())
			rv2.Elem().Set(rv)
			v = rv2i(rv2)
		}
	} else {
		for j := int8(0); j < indir; j++ {
			if rv.IsNil() {
				e.e.EncodeNil()
				return
			}
			rv = rv.Elem()
		}
		v = rv2i(rv)
	}
	return v, true
}

func (e *Encoder) selferMarshal(f *codecFnInfo, rv reflect.Value) {
	if v, proceed := e.getValueForMarshalInterface(rv, f.ti.csIndir); proceed {
		v.(Selfer).CodecEncodeSelf(e)
	}
}

func (e *Encoder) binaryMarshal(f *codecFnInfo, rv reflect.Value) {
	if v, proceed := e.getValueForMarshalInterface(rv, f.ti.bmIndir); proceed {
		bs, fnerr := v.(encoding.BinaryMarshaler).MarshalBinary()
		e.marshal(bs, fnerr, false, c_RAW)
	}
}

func (e *Encoder) textMarshal(f *codecFnInfo, rv reflect.Value) {
	if v, proceed := e.getValueForMarshalInterface(rv, f.ti.tmIndir); proceed {
		bs, fnerr := v.(encoding.TextMarshaler).MarshalText()
		e.marshal(bs, fnerr, false, c_UTF8)
	}
}

func (e *Encoder) jsonMarshal(f *codecFnInfo, rv reflect.Value) {
	if v, proceed := e.getValueForMarshalInterface(rv, f.ti.jmIndir); proceed {
		bs, fnerr := v.(jsonMarshaler).MarshalJSON()
		e.marshal(bs, fnerr, true, c_UTF8)
	}
}

func (e *Encoder) kBool(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeBool(rv.Bool())
}

func (e *Encoder) kString(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeString(c_UTF8, rv.String())
}

func (e *Encoder) kFloat64(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeFloat64(rv.Float())
}

func (e *Encoder) kFloat32(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeFloat32(float32(rv.Float()))
}

func (e *Encoder) kInt(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeInt(rv.Int())
}

func (e *Encoder) kUint(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeUint(rv.Uint())
}

func (e *Encoder) kInvalid(f *codecFnInfo, rv reflect.Value) {
	e.e.EncodeNil()
}

func (e *Encoder) kErr(f *codecFnInfo, rv reflect.Value) {
	e.errorf("unsupported kind %s, for %#v", rv.Kind(), rv)
}

func (e *Encoder) kSlice(f *codecFnInfo, rv reflect.Value) {
	ti := f.ti
	ee := e.e
	// array may be non-addressable, so we have to manage with care
	//   (don't call rv.Bytes, rv.Slice, etc).
	// E.g. type struct S{B [2]byte};
	//   Encode(S{}) will bomb on "panic: slice of unaddressable array".
	if f.seq != seqTypeArray {
		if rv.IsNil() {
			ee.EncodeNil()
			return
		}
		// If in this method, then there was no extension function defined.
		// So it's okay to treat as []byte.
		if ti.rtid == uint8SliceTypId {
			ee.EncodeStringBytes(c_RAW, rv.Bytes())
			return
		}
	}
	elemsep := e.hh.hasElemSeparators()
	rtelem := ti.rt.Elem()
	l := rv.Len()
	if ti.rtid == uint8SliceTypId || rtelem.Kind() == reflect.Uint8 {
		switch f.seq {
		case seqTypeArray:
			if rv.CanAddr() {
				ee.EncodeStringBytes(c_RAW, rv.Slice(0, l).Bytes())
			} else {
				var bs []byte
				if l <= cap(e.b) {
					bs = e.b[:l]
				} else {
					bs = make([]byte, l)
				}
				reflect.Copy(reflect.ValueOf(bs), rv)
				ee.EncodeStringBytes(c_RAW, bs)
			}
			return
		case seqTypeSlice:
			ee.EncodeStringBytes(c_RAW, rv.Bytes())
			return
		}
	}
	if ti.rtid == uint8SliceTypId && f.seq == seqTypeChan {
		bs := e.b[:0]
		// do not use range, so that the number of elements encoded
		// does not change, and encoding does not hang waiting on someone to close chan.
		// for b := range rv2i(rv).(<-chan byte) { bs = append(bs, b) }
		ch := rv2i(rv).(<-chan byte)
		for i := 0; i < l; i++ {
			bs = append(bs, <-ch)
		}
		ee.EncodeStringBytes(c_RAW, bs)
		return
	}

	if ti.mbs {
		if l%2 == 1 {
			e.errorf("mapBySlice requires even slice length, but got %v", l)
			return
		}
		ee.WriteMapStart(l / 2)
	} else {
		ee.WriteArrayStart(l)
	}

	if l > 0 {
		var fn *codecFn
		var recognizedVtyp = useLookupRecognizedTypes && isRecognizedRtidOrPtr(rt2id(rtelem))
		if !recognizedVtyp {
			for rtelem.Kind() == reflect.Ptr {
				rtelem = rtelem.Elem()
			}
			// if kind is reflect.Interface, do not pre-determine the
			// encoding type, because preEncodeValue may break it down to
			// a concrete type and kInterface will bomb.
			if rtelem.Kind() != reflect.Interface {
				fn = e.cf.get(rtelem, true, true)
			}
		}
		// TODO: Consider perf implication of encoding odd index values as symbols if type is string
		for j := 0; j < l; j++ {
			if elemsep {
				if ti.mbs {
					if j%2 == 0 {
						ee.WriteMapElemKey()
					} else {
						ee.WriteMapElemValue()
					}
				} else {
					ee.WriteArrayElem()
				}
			}
			if f.seq == seqTypeChan {
				if rv2, ok2 := rv.Recv(); ok2 {
					if useLookupRecognizedTypes && recognizedVtyp {
						e.encode(rv2i(rv2))
					} else {
						e.encodeValue(rv2, fn, true)
					}
				} else {
					ee.EncodeNil() // WE HAVE TO DO SOMETHING, so nil if nothing received.
				}
			} else {
				if useLookupRecognizedTypes && recognizedVtyp {
					e.encode(rv2i(rv.Index(j)))
				} else {
					e.encodeValue(rv.Index(j), fn, true)
				}
			}
		}
	}

	if ti.mbs {
		ee.WriteMapEnd()
	} else {
		ee.WriteArrayEnd()
	}
}

func (e *Encoder) kStructNoOmitempty(f *codecFnInfo, rv reflect.Value) {
	fti := f.ti
	elemsep := e.hh.hasElemSeparators()
	tisfi := fti.sfip
	toMap := !(fti.toArray || e.h.StructToArray)
	if toMap {
		tisfi = fti.sfi
	}
	ee := e.e

	sfn := structFieldNode{v: rv, update: false}
	if toMap {
		ee.WriteMapStart(len(tisfi))
		// asSymbols := e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		asSymbols := e.h.AsSymbols == AsSymbolDefault || e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		if !elemsep {
			for _, si := range tisfi {
				if asSymbols {
					ee.EncodeSymbol(si.encName)
				} else {
					ee.EncodeString(c_UTF8, si.encName)
				}
				e.encodeValue(sfn.field(si), nil, true)
			}
		} else {
			for _, si := range tisfi {
				ee.WriteMapElemKey()
				if asSymbols {
					ee.EncodeSymbol(si.encName)
				} else {
					ee.EncodeString(c_UTF8, si.encName)
				}
				ee.WriteMapElemValue()
				e.encodeValue(sfn.field(si), nil, true)
			}
		}
		ee.WriteMapEnd()
	} else {
		ee.WriteArrayStart(len(tisfi))
		if !elemsep {
			for _, si := range tisfi {
				e.encodeValue(sfn.field(si), nil, true)
			}
		} else {
			for _, si := range tisfi {
				ee.WriteArrayElem()
				e.encodeValue(sfn.field(si), nil, true)
			}
		}
		ee.WriteArrayEnd()
	}
}

func (e *Encoder) kStruct(f *codecFnInfo, rv reflect.Value) {
	fti := f.ti
	elemsep := e.hh.hasElemSeparators()
	tisfi := fti.sfip
	toMap := !(fti.toArray || e.h.StructToArray)
	// if toMap, use the sorted array. If toArray, use unsorted array (to match sequence in struct)
	if toMap {
		tisfi = fti.sfi
	}
	newlen := len(fti.sfi)
	ee := e.e

	// Use sync.Pool to reduce allocating slices unnecessarily.
	// The cost of sync.Pool is less than the cost of new allocation.
	//
	// Each element of the array pools one of encStructPool(8|16|32|64).
	// It allows the re-use of slices up to 64 in length.
	// A performance cost of encoding structs was collecting
	// which values were empty and should be omitted.
	// We needed slices of reflect.Value and string to collect them.
	// This shared pool reduces the amount of unnecessary creation we do.
	// The cost is that of locking sometimes, but sync.Pool is efficient
	// enough to reduce thread contention.

	var spool *sync.Pool
	var poolv interface{}
	var fkvs []stringRv
	if newlen <= 8 {
		spool, poolv = pool.stringRv8()
		fkvs = poolv.(*[8]stringRv)[:newlen]
	} else if newlen <= 16 {
		spool, poolv = pool.stringRv16()
		fkvs = poolv.(*[16]stringRv)[:newlen]
	} else if newlen <= 32 {
		spool, poolv = pool.stringRv32()
		fkvs = poolv.(*[32]stringRv)[:newlen]
	} else if newlen <= 64 {
		spool, poolv = pool.stringRv64()
		fkvs = poolv.(*[64]stringRv)[:newlen]
	} else if newlen <= 128 {
		spool, poolv = pool.stringRv128()
		fkvs = poolv.(*[128]stringRv)[:newlen]
	} else {
		fkvs = make([]stringRv, newlen)
	}

	newlen = 0
	var kv stringRv
	recur := e.h.RecursiveEmptyCheck
	sfn := structFieldNode{v: rv, update: false}
	for _, si := range tisfi {
		// kv.r = si.field(rv, false)
		kv.r = sfn.field(si)
		if toMap {
			if si.omitEmpty && isEmptyValue(kv.r, recur, recur) {
				continue
			}
			kv.v = si.encName
		} else {
			// use the zero value.
			// if a reference or struct, set to nil (so you do not output too much)
			if si.omitEmpty && isEmptyValue(kv.r, recur, recur) {
				switch kv.r.Kind() {
				case reflect.Struct, reflect.Interface, reflect.Ptr, reflect.Array, reflect.Map, reflect.Slice:
					kv.r = reflect.Value{} //encode as nil
				}
			}
		}
		fkvs[newlen] = kv
		newlen++
	}

	if toMap {
		ee.WriteMapStart(newlen)
		// asSymbols := e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		asSymbols := e.h.AsSymbols == AsSymbolDefault || e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		if !elemsep {
			for j := 0; j < newlen; j++ {
				kv = fkvs[j]
				if asSymbols {
					ee.EncodeSymbol(kv.v)
				} else {
					ee.EncodeString(c_UTF8, kv.v)
				}
				e.encodeValue(kv.r, nil, true)
			}
		} else {
			for j := 0; j < newlen; j++ {
				kv = fkvs[j]
				ee.WriteMapElemKey()
				if asSymbols {
					ee.EncodeSymbol(kv.v)
				} else {
					ee.EncodeString(c_UTF8, kv.v)
				}
				ee.WriteMapElemValue()
				e.encodeValue(kv.r, nil, true)
			}
		}
		ee.WriteMapEnd()
	} else {
		ee.WriteArrayStart(newlen)
		if !elemsep {
			for j := 0; j < newlen; j++ {
				e.encodeValue(fkvs[j].r, nil, true)
			}
		} else {
			for j := 0; j < newlen; j++ {
				ee.WriteArrayElem()
				e.encodeValue(fkvs[j].r, nil, true)
			}
		}
		ee.WriteArrayEnd()
	}

	// do not use defer. Instead, use explicit pool return at end of function.
	// defer has a cost we are trying to avoid.
	// If there is a panic and these slices are not returned, it is ok.
	if spool != nil {
		spool.Put(poolv)
	}
}

func (e *Encoder) kMap(f *codecFnInfo, rv reflect.Value) {
	ee := e.e
	if rv.IsNil() {
		ee.EncodeNil()
		return
	}

	l := rv.Len()
	ee.WriteMapStart(l)
	elemsep := e.hh.hasElemSeparators()
	if l == 0 {
		ee.WriteMapEnd()
		return
	}
	var asSymbols bool
	// determine the underlying key and val encFn's for the map.
	// This eliminates some work which is done for each loop iteration i.e.
	// rv.Type(), ref.ValueOf(rt).Pointer(), then check map/list for fn.
	//
	// However, if kind is reflect.Interface, do not pre-determine the
	// encoding type, because preEncodeValue may break it down to
	// a concrete type and kInterface will bomb.
	var keyFn, valFn *codecFn
	ti := f.ti
	rtkey0 := ti.rt.Key()
	rtkey := rtkey0
	rtval0 := ti.rt.Elem()
	rtval := rtval0
	rtkeyid := rt2id(rtkey0)
	rtvalid := rt2id(rtval0)
	for rtval.Kind() == reflect.Ptr {
		rtval = rtval.Elem()
	}
	if rtval.Kind() != reflect.Interface {
		valFn = e.cf.get(rtval, true, true)
	}
	mks := rv.MapKeys()

	if e.h.Canonical {
		e.kMapCanonical(rtkey, rv, mks, valFn, asSymbols)
		ee.WriteMapEnd()
		return
	}

	var recognizedKtyp, recognizedVtyp bool
	var keyTypeIsString = rtkeyid == stringTypId
	if keyTypeIsString {
		asSymbols = e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	} else {
		if useLookupRecognizedTypes {
			if recognizedKtyp = isRecognizedRtidOrPtr(rtkeyid); recognizedKtyp {
				goto LABEL1
			}
		}
		for rtkey.Kind() == reflect.Ptr {
			rtkey = rtkey.Elem()
		}
		if rtkey.Kind() != reflect.Interface {
			rtkeyid = rt2id(rtkey)
			keyFn = e.cf.get(rtkey, true, true)
		}
	}

	// for j, lmks := 0, len(mks); j < lmks; j++ {
LABEL1:
	recognizedVtyp = useLookupRecognizedTypes && isRecognizedRtidOrPtr(rtvalid)
	for j := range mks {
		if elemsep {
			ee.WriteMapElemKey()
		}
		if keyTypeIsString {
			if asSymbols {
				ee.EncodeSymbol(mks[j].String())
			} else {
				ee.EncodeString(c_UTF8, mks[j].String())
			}
		} else if useLookupRecognizedTypes && recognizedKtyp {
			e.encode(rv2i(mks[j]))
		} else {
			e.encodeValue(mks[j], keyFn, true)
		}
		if elemsep {
			ee.WriteMapElemValue()
		}
		if useLookupRecognizedTypes && recognizedVtyp {
			e.encode(rv2i(rv.MapIndex(mks[j])))
		} else {
			e.encodeValue(rv.MapIndex(mks[j]), valFn, true)
		}
	}
	ee.WriteMapEnd()
}

func (e *Encoder) kMapCanonical(rtkey reflect.Type, rv reflect.Value, mks []reflect.Value, valFn *codecFn, asSymbols bool) {
	ee := e.e
	elemsep := e.hh.hasElemSeparators()
	// we previously did out-of-band if an extension was registered.
	// This is not necessary, as the natural kind is sufficient for ordering.

	// WHAT IS THIS? rtkeyid can never be a []uint8, per spec
	// if rtkeyid == uint8SliceTypId {
	// 	mksv := make([]bytesRv, len(mks))
	// 	for i, k := range mks {
	// 		v := &mksv[i]
	// 		v.r = k
	// 		v.v = k.Bytes()
	// 	}
	// 	sort.Sort(bytesRvSlice(mksv))
	// 	for i := range mksv {
	// 		if elemsep {
	// 			ee.WriteMapElemKey()
	// 		}
	// 		ee.EncodeStringBytes(c_RAW, mksv[i].v)
	// 		if elemsep {
	// 			ee.WriteMapElemValue()
	// 		}
	// 		e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
	// 	}
	// 	return
	// }

	switch rtkey.Kind() {
	case reflect.Bool:
		mksv := make([]boolRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.Bool()
		}
		sort.Sort(boolRvSlice(mksv))
		for i := range mksv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			ee.EncodeBool(mksv[i].v)
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
		}
	case reflect.String:
		mksv := make([]stringRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.String()
		}
		sort.Sort(stringRvSlice(mksv))
		for i := range mksv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			if asSymbols {
				ee.EncodeSymbol(mksv[i].v)
			} else {
				ee.EncodeString(c_UTF8, mksv[i].v)
			}
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
		}
	case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uint, reflect.Uintptr:
		mksv := make([]uintRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.Uint()
		}
		sort.Sort(uintRvSlice(mksv))
		for i := range mksv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			ee.EncodeUint(mksv[i].v)
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
		}
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Int:
		mksv := make([]intRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.Int()
		}
		sort.Sort(intRvSlice(mksv))
		for i := range mksv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			ee.EncodeInt(mksv[i].v)
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
		}
	case reflect.Float32:
		mksv := make([]floatRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.Float()
		}
		sort.Sort(floatRvSlice(mksv))
		for i := range mksv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat32(float32(mksv[i].v))
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
		}
	case reflect.Float64:
		mksv := make([]floatRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.Float()
		}
		sort.Sort(floatRvSlice(mksv))
		for i := range mksv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			ee.EncodeFloat64(mksv[i].v)
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn, true)
		}
	default:
		// out-of-band
		// first encode each key to a []byte first, then sort them, then record
		var mksv []byte = make([]byte, 0, len(mks)*16) // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		mksbv := make([]bytesRv, len(mks))
		for i, k := range mks {
			v := &mksbv[i]
			l := len(mksv)
			e2.MustEncode(k)
			v.r = k
			v.v = mksv[l:]
		}
		sort.Sort(bytesRvSlice(mksbv))
		for j := range mksbv {
			if elemsep {
				ee.WriteMapElemKey()
			}
			e.asis(mksbv[j].v)
			if elemsep {
				ee.WriteMapElemValue()
			}
			e.encodeValue(rv.MapIndex(mksbv[j].r), valFn, true)
		}
	}
}

// // --------------------------------------------------

// An Encoder writes an object to an output stream in the codec format.
type Encoder struct {
	// hopefully, reduce derefencing cost by laying the encWriter inside the Encoder
	e encDriver
	// NOTE: Encoder shouldn't call it's write methods,
	// as the handler MAY need to do some coordination.
	w encWriter

	hh Handle
	h  *BasicHandle

	// ---- cpu cache line boundary?

	wi ioEncWriter
	wb bytesEncWriter
	bw bufio.Writer

	// cr containerStateRecv
	as encDriverAsis
	// ---- cpu cache line boundary?

	ci  set
	err error

	b  [scratchByteArrayLen]byte
	cf codecFner
}

// NewEncoder returns an Encoder for encoding into an io.Writer.
//
// For efficiency, Users are encouraged to pass in a memory buffered writer
// (eg bufio.Writer, bytes.Buffer).
func NewEncoder(w io.Writer, h Handle) *Encoder {
	e := newEncoder(h)
	e.Reset(w)
	return e
}

// NewEncoderBytes returns an encoder for encoding directly and efficiently
// into a byte slice, using zero-copying to temporary slices.
//
// It will potentially replace the output byte slice pointed to.
// After encoding, the out parameter contains the encoded contents.
func NewEncoderBytes(out *[]byte, h Handle) *Encoder {
	e := newEncoder(h)
	e.ResetBytes(out)
	return e
}

func newEncoder(h Handle) *Encoder {
	e := &Encoder{hh: h, h: h.getBasicHandle()}
	e.e = h.newEncDriver(e)
	e.as, _ = e.e.(encDriverAsis)
	// e.cr, _ = e.e.(containerStateRecv)
	return e
}

// Reset the Encoder with a new output stream.
//
// This accommodates using the state of the Encoder,
// where it has "cached" information about sub-engines.
func (e *Encoder) Reset(w io.Writer) {
	var ok bool
	e.wi.w = w
	if e.h.WriterBufferSize > 0 {
		bw := bufio.NewWriterSize(w, e.h.WriterBufferSize)
		e.bw = *bw
		e.wi.bw = &e.bw
		e.wi.sw = &e.bw
		e.wi.fw = &e.bw
		e.wi.ww = &e.bw
	} else {
		if e.wi.bw, ok = w.(io.ByteWriter); !ok {
			e.wi.bw = &e.wi
		}
		if e.wi.sw, ok = w.(ioEncStringWriter); !ok {
			e.wi.sw = &e.wi
		}
		e.wi.fw, _ = w.(ioEncFlusher)
		e.wi.ww = w
	}
	e.w = &e.wi
	e.e.reset()
	e.cf.reset(e.hh)
	e.err = nil
}

func (e *Encoder) ResetBytes(out *[]byte) {
	in := *out
	if in == nil {
		in = make([]byte, defEncByteBufSize)
	}
	e.wb.b, e.wb.out, e.wb.c = in, out, 0
	e.w = &e.wb
	e.e.reset()
	e.cf.reset(e.hh)
	e.err = nil
}

// Encode writes an object into a stream.
//
// Encoding can be configured via the struct tag for the fields.
// The "codec" key in struct field's tag value is the key name,
// followed by an optional comma and options.
// Note that the "json" key is used in the absence of the "codec" key.
//
// To set an option on all fields (e.g. omitempty on all fields), you
// can create a field called _struct, and set flags on it.
//
// Struct values "usually" encode as maps. Each exported struct field is encoded unless:
//    - the field's tag is "-", OR
//    - the field is empty (empty or the zero value) and its tag specifies the "omitempty" option.
//
// When encoding as a map, the first string in the tag (before the comma)
// is the map key string to use when encoding.
//
// However, struct values may encode as arrays. This happens when:
//    - StructToArray Encode option is set, OR
//    - the tag on the _struct field sets the "toarray" option
// Note that omitempty is ignored when encoding struct values as arrays,
// as an entry must be encoded for each field, to maintain its position.
//
// Values with types that implement MapBySlice are encoded as stream maps.
//
// The empty values (for omitempty option) are false, 0, any nil pointer
// or interface value, and any array, slice, map, or string of length zero.
//
// Anonymous fields are encoded inline except:
//    - the struct tag specifies a replacement name (first value)
//    - the field is of an interface type
//
// Examples:
//
//      // NOTE: 'json:' can be used as struct tag key, in place 'codec:' below.
//      type MyStruct struct {
//          _struct bool    `codec:",omitempty"`   //set omitempty for every field
//          Field1 string   `codec:"-"`            //skip this field
//          Field2 int      `codec:"myName"`       //Use key "myName" in encode stream
//          Field3 int32    `codec:",omitempty"`   //use key "Field3". Omit if empty.
//          Field4 bool     `codec:"f4,omitempty"` //use key "f4". Omit if empty.
//          io.Reader                              //use key "Reader".
//          MyStruct        `codec:"my1"           //use key "my1".
//          MyStruct                               //inline it
//          ...
//      }
//
//      type MyStruct struct {
//          _struct bool    `codec:",toarray"`   //encode struct as an array
//      }
//
// The mode of encoding is based on the type of the value. When a value is seen:
//   - If a Selfer, call its CodecEncodeSelf method
//   - If an extension is registered for it, call that extension function
//   - If it implements encoding.(Binary|Text|JSON)Marshaler, call its Marshal(Binary|Text|JSON) method
//   - Else encode it based on its reflect.Kind
//
// Note that struct field names and keys in map[string]XXX will be treated as symbols.
// Some formats support symbols (e.g. binc) and will properly encode the string
// only once in the stream, and use a tag to refer to it thereafter.
func (e *Encoder) Encode(v interface{}) (err error) {
	defer panicToErrs2(&e.err, &err)
	e.MustEncode(v)
	return
}

// MustEncode is like Encode, but panics if unable to Encode.
// This provides insight to the code location that triggered the error.
func (e *Encoder) MustEncode(v interface{}) {
	if e.err != nil {
		panic(e.err)
	}
	e.encode(v)
	e.e.atEndOfEncode()
	e.w.atEndOfEncode()
}

func (e *Encoder) encode(iv interface{}) {
	if iv == nil || definitelyNil(iv) {
		e.e.EncodeNil()
		return
	}
	if v, ok := iv.(Selfer); ok {
		v.CodecEncodeSelf(e)
		return
	}

	switch v := iv.(type) {
	// case nil:
	// 	e.e.EncodeNil()
	// case Selfer:
	// 	v.CodecEncodeSelf(e)
	case Raw:
		e.rawBytes(v)
	case reflect.Value:
		e.encodeValue(v, nil, true)

	case string:
		e.e.EncodeString(c_UTF8, v)
	case bool:
		e.e.EncodeBool(v)
	case int:
		e.e.EncodeInt(int64(v))
	case int8:
		e.e.EncodeInt(int64(v))
	case int16:
		e.e.EncodeInt(int64(v))
	case int32:
		e.e.EncodeInt(int64(v))
	case int64:
		e.e.EncodeInt(v)
	case uint:
		e.e.EncodeUint(uint64(v))
	case uint8:
		e.e.EncodeUint(uint64(v))
	case uint16:
		e.e.EncodeUint(uint64(v))
	case uint32:
		e.e.EncodeUint(uint64(v))
	case uint64:
		e.e.EncodeUint(v)
	case uintptr:
		e.e.EncodeUint(uint64(v))
	case float32:
		e.e.EncodeFloat32(v)
	case float64:
		e.e.EncodeFloat64(v)

	case []uint8:
		e.e.EncodeStringBytes(c_RAW, v)

	case *string:
		e.e.EncodeString(c_UTF8, *v)
	case *bool:
		e.e.EncodeBool(*v)
	case *int:
		e.e.EncodeInt(int64(*v))
	case *int8:
		e.e.EncodeInt(int64(*v))
	case *int16:
		e.e.EncodeInt(int64(*v))
	case *int32:
		e.e.EncodeInt(int64(*v))
	case *int64:
		e.e.EncodeInt(*v)
	case *uint:
		e.e.EncodeUint(uint64(*v))
	case *uint8:
		e.e.EncodeUint(uint64(*v))
	case *uint16:
		e.e.EncodeUint(uint64(*v))
	case *uint32:
		e.e.EncodeUint(uint64(*v))
	case *uint64:
		e.e.EncodeUint(*v)
	case *uintptr:
		e.e.EncodeUint(uint64(*v))
	case *float32:
		e.e.EncodeFloat32(*v)
	case *float64:
		e.e.EncodeFloat64(*v)

	case *[]uint8:
		e.e.EncodeStringBytes(c_RAW, *v)

	default:
		if !fastpathEncodeTypeSwitch(iv, e) {
			e.encodeValue(reflect.ValueOf(iv), nil, false)
		}
	}
}

func (e *Encoder) encodeValue(rv reflect.Value, fn *codecFn, checkFastpath bool) {
	// if a valid fn is passed, it MUST BE for the dereferenced type of rv
	var sptr uintptr
TOP:
	switch rv.Kind() {
	case reflect.Ptr:
		if rv.IsNil() {
			e.e.EncodeNil()
			return
		}
		rv = rv.Elem()
		if e.h.CheckCircularRef && rv.Kind() == reflect.Struct {
			// TODO: Movable pointers will be an issue here. Future problem.
			sptr = rv.UnsafeAddr()
			break TOP
		}
		goto TOP
	case reflect.Interface:
		if rv.IsNil() {
			e.e.EncodeNil()
			return
		}
		rv = rv.Elem()
		goto TOP
	case reflect.Slice, reflect.Map:
		if rv.IsNil() {
			e.e.EncodeNil()
			return
		}
	case reflect.Invalid, reflect.Func:
		e.e.EncodeNil()
		return
	}

	if sptr != 0 && (&e.ci).add(sptr) {
		e.errorf("circular reference found: # %d", sptr)
	}

	if fn == nil {
		rt := rv.Type()
		// TODO: calling isRecognizedRtid here is a major slowdown
		if false && useLookupRecognizedTypes && isRecognizedRtidOrPtr(rt2id(rt)) {
			e.encode(rv2i(rv))
			return
		}
		// always pass checkCodecSelfer=true, in case T or ****T is passed, where *T is a Selfer
		fn = e.cf.get(rt, checkFastpath, true)
	}
	fn.fe(e, &fn.i, rv)
	if sptr != 0 {
		(&e.ci).remove(sptr)
	}
}

func (e *Encoder) marshal(bs []byte, fnerr error, asis bool, c charEncoding) {
	if fnerr != nil {
		panic(fnerr)
	}
	if bs == nil {
		e.e.EncodeNil()
	} else if asis {
		e.asis(bs)
	} else {
		e.e.EncodeStringBytes(c, bs)
	}
}

func (e *Encoder) asis(v []byte) {
	if e.as == nil {
		e.w.writeb(v)
	} else {
		e.as.EncodeAsis(v)
	}
}

func (e *Encoder) rawBytes(vv Raw) {
	v := []byte(vv)
	if !e.h.Raw {
		e.errorf("Raw values cannot be encoded: %v", v)
	}
	if e.as == nil {
		e.w.writeb(v)
	} else {
		e.as.EncodeAsis(v)
	}
}

func (e *Encoder) errorf(format string, params ...interface{}) {
	err := fmt.Errorf(format, params...)
	panic(err)
}
