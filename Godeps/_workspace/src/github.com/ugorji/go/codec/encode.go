// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

import (
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"sync"
)

const (
	defEncByteBufSize = 1 << 6 // 4:16, 6:64, 8:256, 10:1024
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

// encWriter abstracts writing to a byte array or to an io.Writer.
type encWriter interface {
	writeb([]byte)
	writestr(string)
	writen1(byte)
	writen2(byte, byte)
	atEndOfEncode()
}

// encDriver abstracts the actual codec (binc vs msgpack, etc)
type encDriver interface {
	IsBuiltinType(rt uintptr) bool
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
	EncodeArrayStart(length int)
	EncodeArrayEnd()
	EncodeArrayEntrySeparator()
	EncodeMapStart(length int)
	EncodeMapEnd()
	EncodeMapEntrySeparator()
	EncodeMapKVSeparator()
	EncodeString(c charEncoding, v string)
	EncodeSymbol(v string)
	EncodeStringBytes(c charEncoding, v []byte)
	//TODO
	//encBignum(f *big.Int)
	//encStringRunes(c charEncoding, v []rune)
}

type encNoSeparator struct{}

func (_ encNoSeparator) EncodeMapEnd()              {}
func (_ encNoSeparator) EncodeArrayEnd()            {}
func (_ encNoSeparator) EncodeArrayEntrySeparator() {}
func (_ encNoSeparator) EncodeMapEntrySeparator()   {}
func (_ encNoSeparator) EncodeMapKVSeparator()      {}

type encStructFieldBytesV struct {
	b []byte
	v reflect.Value
}

type encStructFieldBytesVslice []encStructFieldBytesV

func (p encStructFieldBytesVslice) Len() int           { return len(p) }
func (p encStructFieldBytesVslice) Less(i, j int) bool { return bytes.Compare(p[i].b, p[j].b) == -1 }
func (p encStructFieldBytesVslice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

type ioEncWriterWriter interface {
	WriteByte(c byte) error
	WriteString(s string) (n int, err error)
	Write(p []byte) (n int, err error)
}

type ioEncStringWriter interface {
	WriteString(s string) (n int, err error)
}

type EncodeOptions struct {
	// Encode a struct as an array, and not as a map
	StructToArray bool

	// Canonical representation means that encoding a value will always result in the same
	// sequence of bytes.
	//
	// This mostly will apply to maps. In this case, codec will do more work to encode the
	// map keys out of band, and then sort them, before writing out the map to the stream.
	Canonical bool

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
	// return o.w.Write([]byte(s))
	return o.w.Write(bytesView(s))
}

func (o *simpleIoEncWriterWriter) Write(p []byte) (n int, err error) {
	return o.w.Write(p)
}

// ----------------------------------------

// ioEncWriter implements encWriter and can write to an io.Writer implementation
type ioEncWriter struct {
	w ioEncWriterWriter
	// x [8]byte // temp byte array re-used internally for efficiency
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
		panic(fmt.Errorf("incorrect num bytes written. Expecting: %v, Wrote: %v", len(bs), n))
	}
}

func (z *ioEncWriter) writestr(s string) {
	n, err := z.w.WriteString(s)
	if err != nil {
		panic(err)
	}
	if n != len(s) {
		panic(fmt.Errorf("incorrect num bytes written. Expecting: %v, Wrote: %v", len(s), n))
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

func (z *bytesEncWriter) writeb(s []byte) {
	if len(s) > 0 {
		c := z.grow(len(s))
		copy(z.b[c:], s)
	}
}

func (z *bytesEncWriter) writestr(s string) {
	if len(s) > 0 {
		c := z.grow(len(s))
		copy(z.b[c:], s)
	}
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
	if z.c > len(z.b) {
		if z.c > cap(z.b) {
			// Tried using appendslice logic: (if cap < 1024, *2, else *1.25).
			// However, it was too expensive, causing too many iterations of copy.
			// Using bytes.Buffer model was much better (2*cap + n)
			bs := make([]byte, 2*cap(z.b)+n)
			copy(bs, z.b[:oldcursor])
			z.b = bs
		} else {
			z.b = z.b[:cap(z.b)]
		}
	}
	return
}

// ---------------------------------------------

type encFnInfoX struct {
	e     *Encoder
	ti    *typeInfo
	xfFn  Ext
	xfTag uint64
	seq   seqType
}

type encFnInfo struct {
	// use encFnInfo as a value receiver.
	// keep most of it less-used variables accessible via a pointer (*encFnInfoX).
	// As sweet spot for value-receiver is 3 words, keep everything except
	// encDriver (which everyone needs) directly accessible.
	// ensure encFnInfoX is set for everyone who needs it i.e.
	// rawExt, ext, builtin, (selfer|binary|text)Marshal, kSlice, kStruct, kMap, kInterface, fastpath

	ee encDriver
	*encFnInfoX
}

func (f encFnInfo) builtin(rv reflect.Value) {
	f.ee.EncodeBuiltin(f.ti.rtid, rv.Interface())
}

func (f encFnInfo) rawExt(rv reflect.Value) {
	// rev := rv.Interface().(RawExt)
	// f.ee.EncodeRawExt(&rev, f.e)
	var re *RawExt
	if rv.CanAddr() {
		re = rv.Addr().Interface().(*RawExt)
	} else {
		rev := rv.Interface().(RawExt)
		re = &rev
	}
	f.ee.EncodeRawExt(re, f.e)
}

func (f encFnInfo) ext(rv reflect.Value) {
	// if this is a struct and it was addressable, then pass the address directly (not the value)
	if rv.CanAddr() && rv.Kind() == reflect.Struct {
		rv = rv.Addr()
	}
	f.ee.EncodeExt(rv.Interface(), f.xfTag, f.xfFn, f.e)
}

func (f encFnInfo) getValueForMarshalInterface(rv reflect.Value, indir int8) (v interface{}, proceed bool) {
	if indir == 0 {
		v = rv.Interface()
	} else if indir == -1 {
		v = rv.Addr().Interface()
	} else {
		for j := int8(0); j < indir; j++ {
			if rv.IsNil() {
				f.ee.EncodeNil()
				return
			}
			rv = rv.Elem()
		}
		v = rv.Interface()
	}
	return v, true
}

func (f encFnInfo) selferMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.csIndir); proceed {
		v.(Selfer).CodecEncodeSelf(f.e)
	}
}

func (f encFnInfo) binaryMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.bmIndir); proceed {
		bs, fnerr := v.(encoding.BinaryMarshaler).MarshalBinary()
		if fnerr != nil {
			panic(fnerr)
		}
		if bs == nil {
			f.ee.EncodeNil()
		} else {
			f.ee.EncodeStringBytes(c_RAW, bs)
		}
	}
}

func (f encFnInfo) textMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.tmIndir); proceed {
		// debugf(">>>> encoding.TextMarshaler: %T", rv.Interface())
		bs, fnerr := v.(encoding.TextMarshaler).MarshalText()
		if fnerr != nil {
			panic(fnerr)
		}
		if bs == nil {
			f.ee.EncodeNil()
		} else {
			f.ee.EncodeStringBytes(c_UTF8, bs)
		}
	}
}

func (f encFnInfo) kBool(rv reflect.Value) {
	f.ee.EncodeBool(rv.Bool())
}

func (f encFnInfo) kString(rv reflect.Value) {
	f.ee.EncodeString(c_UTF8, rv.String())
}

func (f encFnInfo) kFloat64(rv reflect.Value) {
	f.ee.EncodeFloat64(rv.Float())
}

func (f encFnInfo) kFloat32(rv reflect.Value) {
	f.ee.EncodeFloat32(float32(rv.Float()))
}

func (f encFnInfo) kInt(rv reflect.Value) {
	f.ee.EncodeInt(rv.Int())
}

func (f encFnInfo) kUint(rv reflect.Value) {
	f.ee.EncodeUint(rv.Uint())
}

func (f encFnInfo) kInvalid(rv reflect.Value) {
	f.ee.EncodeNil()
}

func (f encFnInfo) kErr(rv reflect.Value) {
	f.e.errorf("unsupported kind %s, for %#v", rv.Kind(), rv)
}

func (f encFnInfo) kSlice(rv reflect.Value) {
	ti := f.ti
	// array may be non-addressable, so we have to manage with care
	//   (don't call rv.Bytes, rv.Slice, etc).
	// E.g. type struct S{B [2]byte};
	//   Encode(S{}) will bomb on "panic: slice of unaddressable array".
	if f.seq != seqTypeArray {
		if rv.IsNil() {
			f.ee.EncodeNil()
			return
		}
		// If in this method, then there was no extension function defined.
		// So it's okay to treat as []byte.
		if ti.rtid == uint8SliceTypId {
			f.ee.EncodeStringBytes(c_RAW, rv.Bytes())
			return
		}
	}
	rtelem := ti.rt.Elem()
	l := rv.Len()
	if rtelem.Kind() == reflect.Uint8 {
		switch f.seq {
		case seqTypeArray:
			// if l == 0 { f.ee.encodeStringBytes(c_RAW, nil) } else
			if rv.CanAddr() {
				f.ee.EncodeStringBytes(c_RAW, rv.Slice(0, l).Bytes())
			} else {
				var bs []byte
				if l <= cap(f.e.b) {
					bs = f.e.b[:l]
				} else {
					bs = make([]byte, l)
				}
				reflect.Copy(reflect.ValueOf(bs), rv)
				// TODO: Test that reflect.Copy works instead of manual one-by-one
				// for i := 0; i < l; i++ {
				// 	bs[i] = byte(rv.Index(i).Uint())
				// }
				f.ee.EncodeStringBytes(c_RAW, bs)
			}
		case seqTypeSlice:
			f.ee.EncodeStringBytes(c_RAW, rv.Bytes())
		case seqTypeChan:
			bs := f.e.b[:0]
			// do not use range, so that the number of elements encoded
			// does not change, and encoding does not hang waiting on someone to close chan.
			// for b := range rv.Interface().(<-chan byte) {
			// 	bs = append(bs, b)
			// }
			ch := rv.Interface().(<-chan byte)
			for i := 0; i < l; i++ {
				bs = append(bs, <-ch)
			}
			f.ee.EncodeStringBytes(c_RAW, bs)
		}
		return
	}

	if ti.mbs {
		if l%2 == 1 {
			f.e.errorf("mapBySlice requires even slice length, but got %v", l)
			return
		}
		f.ee.EncodeMapStart(l / 2)
	} else {
		f.ee.EncodeArrayStart(l)
	}

	e := f.e
	sep := !e.be
	if l > 0 {
		for rtelem.Kind() == reflect.Ptr {
			rtelem = rtelem.Elem()
		}
		// if kind is reflect.Interface, do not pre-determine the
		// encoding type, because preEncodeValue may break it down to
		// a concrete type and kInterface will bomb.
		var fn encFn
		if rtelem.Kind() != reflect.Interface {
			rtelemid := reflect.ValueOf(rtelem).Pointer()
			fn = e.getEncFn(rtelemid, rtelem, true, true)
		}
		// TODO: Consider perf implication of encoding odd index values as symbols if type is string
		if sep {
			for j := 0; j < l; j++ {
				if j > 0 {
					if ti.mbs {
						if j%2 == 0 {
							f.ee.EncodeMapEntrySeparator()
						} else {
							f.ee.EncodeMapKVSeparator()
						}
					} else {
						f.ee.EncodeArrayEntrySeparator()
					}
				}
				if f.seq == seqTypeChan {
					if rv2, ok2 := rv.Recv(); ok2 {
						e.encodeValue(rv2, fn)
					}
				} else {
					e.encodeValue(rv.Index(j), fn)
				}
			}
		} else {
			for j := 0; j < l; j++ {
				if f.seq == seqTypeChan {
					if rv2, ok2 := rv.Recv(); ok2 {
						e.encodeValue(rv2, fn)
					}
				} else {
					e.encodeValue(rv.Index(j), fn)
				}
			}
		}
	}

	if sep {
		if ti.mbs {
			f.ee.EncodeMapEnd()
		} else {
			f.ee.EncodeArrayEnd()
		}
	}
}

func (f encFnInfo) kStruct(rv reflect.Value) {
	fti := f.ti
	e := f.e
	tisfi := fti.sfip
	toMap := !(fti.toArray || e.h.StructToArray)
	newlen := len(fti.sfi)
	// Use sync.Pool to reduce allocating slices unnecessarily.
	// The cost of the occasional locking is less than the cost of locking.

	var fkvs []encStructFieldKV
	var pool *sync.Pool
	var poolv interface{}
	idxpool := newlen / 8
	if encStructPoolLen != 4 {
		panic(errors.New("encStructPoolLen must be equal to 4")) // defensive, in case it is changed
	}
	if idxpool < encStructPoolLen {
		pool = &encStructPool[idxpool]
		poolv = pool.Get()
		switch vv := poolv.(type) {
		case *[8]encStructFieldKV:
			fkvs = vv[:newlen]
		case *[16]encStructFieldKV:
			fkvs = vv[:newlen]
		case *[32]encStructFieldKV:
			fkvs = vv[:newlen]
		case *[64]encStructFieldKV:
			fkvs = vv[:newlen]
		}
	}
	if fkvs == nil {
		fkvs = make([]encStructFieldKV, newlen)
	}
	// if toMap, use the sorted array. If toArray, use unsorted array (to match sequence in struct)
	if toMap {
		tisfi = fti.sfi
	}
	newlen = 0
	var kv encStructFieldKV
	for _, si := range tisfi {
		kv.v = si.field(rv, false)
		// if si.i != -1 {
		// 	rvals[newlen] = rv.Field(int(si.i))
		// } else {
		// 	rvals[newlen] = rv.FieldByIndex(si.is)
		// }
		if toMap {
			if si.omitEmpty && isEmptyValue(kv.v) {
				continue
			}
			kv.k = si.encName
		} else {
			// use the zero value.
			// if a reference or struct, set to nil (so you do not output too much)
			if si.omitEmpty && isEmptyValue(kv.v) {
				switch kv.v.Kind() {
				case reflect.Struct, reflect.Interface, reflect.Ptr, reflect.Array,
					reflect.Map, reflect.Slice:
					kv.v = reflect.Value{} //encode as nil
				}
			}
		}
		fkvs[newlen] = kv
		newlen++
	}

	// debugf(">>>> kStruct: newlen: %v", newlen)
	sep := !e.be
	ee := f.ee //don't dereference everytime
	if sep {
		if toMap {
			ee.EncodeMapStart(newlen)
			// asSymbols := e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
			asSymbols := e.h.AsSymbols == AsSymbolDefault || e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
			for j := 0; j < newlen; j++ {
				kv = fkvs[j]
				if j > 0 {
					ee.EncodeMapEntrySeparator()
				}
				if asSymbols {
					ee.EncodeSymbol(kv.k)
				} else {
					ee.EncodeString(c_UTF8, kv.k)
				}
				ee.EncodeMapKVSeparator()
				e.encodeValue(kv.v, encFn{})
			}
			ee.EncodeMapEnd()
		} else {
			ee.EncodeArrayStart(newlen)
			for j := 0; j < newlen; j++ {
				kv = fkvs[j]
				if j > 0 {
					ee.EncodeArrayEntrySeparator()
				}
				e.encodeValue(kv.v, encFn{})
			}
			ee.EncodeArrayEnd()
		}
	} else {
		if toMap {
			ee.EncodeMapStart(newlen)
			// asSymbols := e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
			asSymbols := e.h.AsSymbols == AsSymbolDefault || e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
			for j := 0; j < newlen; j++ {
				kv = fkvs[j]
				if asSymbols {
					ee.EncodeSymbol(kv.k)
				} else {
					ee.EncodeString(c_UTF8, kv.k)
				}
				e.encodeValue(kv.v, encFn{})
			}
		} else {
			ee.EncodeArrayStart(newlen)
			for j := 0; j < newlen; j++ {
				kv = fkvs[j]
				e.encodeValue(kv.v, encFn{})
			}
		}
	}

	// do not use defer. Instead, use explicit pool return at end of function.
	// defer has a cost we are trying to avoid.
	// If there is a panic and these slices are not returned, it is ok.
	if pool != nil {
		pool.Put(poolv)
	}
}

// func (f encFnInfo) kPtr(rv reflect.Value) {
// 	debugf(">>>>>>> ??? encode kPtr called - shouldn't get called")
// 	if rv.IsNil() {
// 		f.ee.encodeNil()
// 		return
// 	}
// 	f.e.encodeValue(rv.Elem())
// }

func (f encFnInfo) kInterface(rv reflect.Value) {
	if rv.IsNil() {
		f.ee.EncodeNil()
		return
	}
	f.e.encodeValue(rv.Elem(), encFn{})
}

func (f encFnInfo) kMap(rv reflect.Value) {
	if rv.IsNil() {
		f.ee.EncodeNil()
		return
	}

	l := rv.Len()
	f.ee.EncodeMapStart(l)
	e := f.e
	sep := !e.be
	if l == 0 {
		if sep {
			f.ee.EncodeMapEnd()
		}
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
	var keyFn, valFn encFn
	ti := f.ti
	rtkey := ti.rt.Key()
	rtval := ti.rt.Elem()
	rtkeyid := reflect.ValueOf(rtkey).Pointer()
	// keyTypeIsString := f.ti.rt.Key().Kind() == reflect.String
	var keyTypeIsString = rtkeyid == stringTypId
	if keyTypeIsString {
		asSymbols = e.h.AsSymbols&AsSymbolMapStringKeysFlag != 0
	} else {
		for rtkey.Kind() == reflect.Ptr {
			rtkey = rtkey.Elem()
		}
		if rtkey.Kind() != reflect.Interface {
			rtkeyid = reflect.ValueOf(rtkey).Pointer()
			keyFn = e.getEncFn(rtkeyid, rtkey, true, true)
		}
	}
	for rtval.Kind() == reflect.Ptr {
		rtval = rtval.Elem()
	}
	if rtval.Kind() != reflect.Interface {
		rtvalid := reflect.ValueOf(rtval).Pointer()
		valFn = e.getEncFn(rtvalid, rtval, true, true)
	}
	mks := rv.MapKeys()
	// for j, lmks := 0, len(mks); j < lmks; j++ {
	ee := f.ee //don't dereference everytime
	if e.h.Canonical {
		// first encode each key to a []byte first, then sort them, then record
		// println(">>>>>>>> CANONICAL <<<<<<<<")
		var mksv []byte // temporary byte slice for the encoding
		e2 := NewEncoderBytes(&mksv, e.hh)
		mksbv := make([]encStructFieldBytesV, len(mks))
		for i, k := range mks {
			l := len(mksv)
			e2.MustEncode(k)
			mksbv[i].v = k
			mksbv[i].b = mksv[l:]
		}
		sort.Sort(encStructFieldBytesVslice(mksbv))
		for j := range mksbv {
			if j > 0 {
				ee.EncodeMapEntrySeparator()
			}
			e.w.writeb(mksbv[j].b)
			ee.EncodeMapKVSeparator()
			e.encodeValue(rv.MapIndex(mksbv[j].v), valFn)
		}
		ee.EncodeMapEnd()
	} else if sep {
		for j := range mks {
			if j > 0 {
				ee.EncodeMapEntrySeparator()
			}
			if keyTypeIsString {
				if asSymbols {
					ee.EncodeSymbol(mks[j].String())
				} else {
					ee.EncodeString(c_UTF8, mks[j].String())
				}
			} else {
				e.encodeValue(mks[j], keyFn)
			}
			ee.EncodeMapKVSeparator()
			e.encodeValue(rv.MapIndex(mks[j]), valFn)
		}
		ee.EncodeMapEnd()
	} else {
		for j := range mks {
			if keyTypeIsString {
				if asSymbols {
					ee.EncodeSymbol(mks[j].String())
				} else {
					ee.EncodeString(c_UTF8, mks[j].String())
				}
			} else {
				e.encodeValue(mks[j], keyFn)
			}
			e.encodeValue(rv.MapIndex(mks[j]), valFn)
		}
	}
}

// --------------------------------------------------

// encFn encapsulates the captured variables and the encode function.
// This way, we only do some calculations one times, and pass to the
// code block that should be called (encapsulated in a function)
// instead of executing the checks every time.
type encFn struct {
	i encFnInfo
	f func(encFnInfo, reflect.Value)
}

// --------------------------------------------------

type rtidEncFn struct {
	rtid uintptr
	fn   encFn
}

// An Encoder writes an object to an output stream in the codec format.
type Encoder struct {
	// hopefully, reduce derefencing cost by laying the encWriter inside the Encoder
	e  encDriver
	w  encWriter
	s  []rtidEncFn
	be bool // is binary encoding

	wi ioEncWriter
	wb bytesEncWriter
	h  *BasicHandle

	hh Handle
	f  map[uintptr]encFn
	b  [scratchByteArrayLen]byte
}

// NewEncoder returns an Encoder for encoding into an io.Writer.
//
// For efficiency, Users are encouraged to pass in a memory buffered writer
// (eg bufio.Writer, bytes.Buffer).
func NewEncoder(w io.Writer, h Handle) *Encoder {
	e := &Encoder{hh: h, h: h.getBasicHandle(), be: h.isBinary()}
	ww, ok := w.(ioEncWriterWriter)
	if !ok {
		sww := simpleIoEncWriterWriter{w: w}
		sww.bw, _ = w.(io.ByteWriter)
		sww.sw, _ = w.(ioEncStringWriter)
		ww = &sww
		//ww = bufio.NewWriterSize(w, defEncByteBufSize)
	}
	e.wi.w = ww
	e.w = &e.wi
	e.e = h.newEncDriver(e)
	return e
}

// NewEncoderBytes returns an encoder for encoding directly and efficiently
// into a byte slice, using zero-copying to temporary slices.
//
// It will potentially replace the output byte slice pointed to.
// After encoding, the out parameter contains the encoded contents.
func NewEncoderBytes(out *[]byte, h Handle) *Encoder {
	e := &Encoder{hh: h, h: h.getBasicHandle(), be: h.isBinary()}
	in := *out
	if in == nil {
		in = make([]byte, defEncByteBufSize)
	}
	e.wb.b, e.wb.out = in, out
	e.w = &e.wb
	e.e = h.newEncDriver(e)
	return e
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
//      // NOTE: 'json:' can be used as struct tag key, in place 'codec:' below.
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

// MustEncode is like Encode, but panics if unable to Encode.
// This provides insight to the code location that triggered the error.
func (e *Encoder) MustEncode(v interface{}) {
	e.encode(v)
	e.w.atEndOfEncode()
}

// comment out these (Must)Write methods. They were only put there to support cbor.
// However, users already have access to the streams, and can write directly.
//
// // Write allows users write to the Encoder stream directly.
// func (e *Encoder) Write(bs []byte) (err error) {
// 	defer panicToErr(&err)
// 	e.w.writeb(bs)
// 	return
// }
// // MustWrite is like write, but panics if unable to Write.
// func (e *Encoder) MustWrite(bs []byte) {
// 	e.w.writeb(bs)
// }

func (e *Encoder) encode(iv interface{}) {
	// if ics, ok := iv.(Selfer); ok {
	// 	ics.CodecEncodeSelf(e)
	// 	return
	// }

	switch v := iv.(type) {
	case nil:
		e.e.EncodeNil()
	case Selfer:
		v.CodecEncodeSelf(e)

	case reflect.Value:
		e.encodeValue(v, encFn{})

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
	case *float32:
		e.e.EncodeFloat32(*v)
	case *float64:
		e.e.EncodeFloat64(*v)

	case *[]uint8:
		e.e.EncodeStringBytes(c_RAW, *v)

	default:
		// canonical mode is not supported for fastpath of maps (but is fine for slices)
		if e.h.Canonical {
			if !fastpathEncodeTypeSwitchSlice(iv, e) {
				e.encodeI(iv, false, false)
			}
		} else if !fastpathEncodeTypeSwitch(iv, e) {
			e.encodeI(iv, false, false)
		}
	}
}

func (e *Encoder) encodeI(iv interface{}, checkFastpath, checkCodecSelfer bool) {
	if rv, proceed := e.preEncodeValue(reflect.ValueOf(iv)); proceed {
		rt := rv.Type()
		rtid := reflect.ValueOf(rt).Pointer()
		fn := e.getEncFn(rtid, rt, checkFastpath, checkCodecSelfer)
		fn.f(fn.i, rv)
	}
}

func (e *Encoder) preEncodeValue(rv reflect.Value) (rv2 reflect.Value, proceed bool) {
LOOP:
	for {
		switch rv.Kind() {
		case reflect.Ptr, reflect.Interface:
			if rv.IsNil() {
				e.e.EncodeNil()
				return
			}
			rv = rv.Elem()
			continue LOOP
		case reflect.Slice, reflect.Map:
			if rv.IsNil() {
				e.e.EncodeNil()
				return
			}
		case reflect.Invalid, reflect.Func:
			e.e.EncodeNil()
			return
		}
		break
	}

	return rv, true
}

func (e *Encoder) encodeValue(rv reflect.Value, fn encFn) {
	// if a valid fn is passed, it MUST BE for the dereferenced type of rv
	if rv, proceed := e.preEncodeValue(rv); proceed {
		if fn.f == nil {
			rt := rv.Type()
			rtid := reflect.ValueOf(rt).Pointer()
			fn = e.getEncFn(rtid, rt, true, true)
		}
		fn.f(fn.i, rv)
	}
}

func (e *Encoder) getEncFn(rtid uintptr, rt reflect.Type, checkFastpath, checkCodecSelfer bool) (fn encFn) {
	// rtid := reflect.ValueOf(rt).Pointer()
	var ok bool
	if useMapForCodecCache {
		fn, ok = e.f[rtid]
	} else {
		for _, v := range e.s {
			if v.rtid == rtid {
				fn, ok = v.fn, true
				break
			}
		}
	}
	if ok {
		return
	}
	// fi.encFnInfoX = new(encFnInfoX)
	ti := getTypeInfo(rtid, rt)
	var fi encFnInfo
	fi.ee = e.e

	if checkCodecSelfer && ti.cs {
		fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
		fn.f = (encFnInfo).selferMarshal
	} else if rtid == rawExtTypId {
		fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
		fn.f = (encFnInfo).rawExt
	} else if e.e.IsBuiltinType(rtid) {
		fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
		fn.f = (encFnInfo).builtin
	} else if xfFn := e.h.getExt(rtid); xfFn != nil {
		// fi.encFnInfoX = new(encFnInfoX)
		fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
		fi.xfTag, fi.xfFn = xfFn.tag, xfFn.ext
		fn.f = (encFnInfo).ext
	} else if supportMarshalInterfaces && e.be && ti.bm {
		fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
		fn.f = (encFnInfo).binaryMarshal
	} else if supportMarshalInterfaces && !e.be && ti.tm {
		fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
		fn.f = (encFnInfo).textMarshal
	} else {
		rk := rt.Kind()
		if fastpathEnabled && checkFastpath && (rk == reflect.Map || rk == reflect.Slice) {
			if rt.PkgPath() == "" {
				if idx := fastpathAV.index(rtid); idx != -1 {
					fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
					fn.f = fastpathAV[idx].encfn
				}
			} else {
				ok = false
				// use mapping for underlying type if there
				var rtu reflect.Type
				if rk == reflect.Map {
					rtu = reflect.MapOf(rt.Key(), rt.Elem())
				} else {
					rtu = reflect.SliceOf(rt.Elem())
				}
				rtuid := reflect.ValueOf(rtu).Pointer()
				if idx := fastpathAV.index(rtuid); idx != -1 {
					xfnf := fastpathAV[idx].encfn
					xrt := fastpathAV[idx].rt
					fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
					fn.f = func(xf encFnInfo, xrv reflect.Value) {
						xfnf(xf, xrv.Convert(xrt))
					}
				}
			}
		}
		if fn.f == nil {
			switch rk {
			case reflect.Bool:
				fn.f = (encFnInfo).kBool
			case reflect.String:
				fn.f = (encFnInfo).kString
			case reflect.Float64:
				fn.f = (encFnInfo).kFloat64
			case reflect.Float32:
				fn.f = (encFnInfo).kFloat32
			case reflect.Int, reflect.Int8, reflect.Int64, reflect.Int32, reflect.Int16:
				fn.f = (encFnInfo).kInt
			case reflect.Uint8, reflect.Uint64, reflect.Uint, reflect.Uint32, reflect.Uint16:
				fn.f = (encFnInfo).kUint
			case reflect.Invalid:
				fn.f = (encFnInfo).kInvalid
			case reflect.Chan:
				fi.encFnInfoX = &encFnInfoX{e: e, ti: ti, seq: seqTypeChan}
				fn.f = (encFnInfo).kSlice
			case reflect.Slice:
				fi.encFnInfoX = &encFnInfoX{e: e, ti: ti, seq: seqTypeSlice}
				fn.f = (encFnInfo).kSlice
			case reflect.Array:
				fi.encFnInfoX = &encFnInfoX{e: e, ti: ti, seq: seqTypeArray}
				fn.f = (encFnInfo).kSlice
			case reflect.Struct:
				fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
				fn.f = (encFnInfo).kStruct
				// case reflect.Ptr:
				// 	fn.f = (encFnInfo).kPtr
			case reflect.Interface:
				fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
				fn.f = (encFnInfo).kInterface
			case reflect.Map:
				fi.encFnInfoX = &encFnInfoX{e: e, ti: ti}
				fn.f = (encFnInfo).kMap
			default:
				fn.f = (encFnInfo).kErr
			}
		}
	}
	fn.i = fi

	if useMapForCodecCache {
		if e.f == nil {
			e.f = make(map[uintptr]encFn, 32)
		}
		e.f[rtid] = fn
	} else {
		if e.s == nil {
			e.s = make([]rtidEncFn, 0, 32)
		}
		e.s = append(e.s, rtidEncFn{rtid, fn})
	}
	return
}

func (e *Encoder) errorf(format string, params ...interface{}) {
	err := fmt.Errorf(format, params...)
	panic(err)
}

// ----------------------------------------

type encStructFieldKV struct {
	k string
	v reflect.Value
}

const encStructPoolLen = 4

// encStructPool is an array of sync.Pool.
// Each element of the array pools one of encStructPool(8|16|32|64).
// It allows the re-use of slices up to 64 in length.
// A performance cost of encoding structs was collecting
// which values were empty and should be omitted.
// We needed slices of reflect.Value and string to collect them.
// This shared pool reduces the amount of unnecessary creation we do.
// The cost is that of locking sometimes, but sync.Pool is efficient
// enough to reduce thread contention.
var encStructPool [encStructPoolLen]sync.Pool

func init() {
	encStructPool[0].New = func() interface{} { return new([8]encStructFieldKV) }
	encStructPool[1].New = func() interface{} { return new([16]encStructFieldKV) }
	encStructPool[2].New = func() interface{} { return new([32]encStructFieldKV) }
	encStructPool[3].New = func() interface{} { return new([64]encStructFieldKV) }
}

// ----------------------------------------

// func encErr(format string, params ...interface{}) {
// 	doPanic(msgTagEnc, format, params...)
// }
