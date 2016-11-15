// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"encoding"
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
	EncodeMapStart(length int)
	EncodeString(c charEncoding, v string)
	EncodeSymbol(v string)
	EncodeStringBytes(c charEncoding, v []byte)
	//TODO
	//encBignum(f *big.Int)
	//encStringRunes(c charEncoding, v []rune)

	reset()
}

type encDriverAsis interface {
	EncodeAsis(v []byte)
}

type encNoSeparator struct{}

func (_ encNoSeparator) EncodeEnd() {}

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
	s simpleIoEncWriterWriter
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
			// appendslice logic (if cap < 1024, *2, else *1.25): more expensive. many copy calls.
			// bytes.Buffer model (2*cap + n): much better
			// bs := make([]byte, 2*cap(z.b)+n)
			bs := make([]byte, growCap(cap(z.b), 1, n))
			copy(bs, z.b[:oldcursor])
			z.b = bs
		} else {
			z.b = z.b[:cap(z.b)]
		}
	}
	return
}

// ---------------------------------------------

type encFnInfo struct {
	e     *Encoder
	ti    *typeInfo
	xfFn  Ext
	xfTag uint64
	seq   seqType
}

func (f *encFnInfo) builtin(rv reflect.Value) {
	f.e.e.EncodeBuiltin(f.ti.rtid, rv.Interface())
}

func (f *encFnInfo) rawExt(rv reflect.Value) {
	// rev := rv.Interface().(RawExt)
	// f.e.e.EncodeRawExt(&rev, f.e)
	var re *RawExt
	if rv.CanAddr() {
		re = rv.Addr().Interface().(*RawExt)
	} else {
		rev := rv.Interface().(RawExt)
		re = &rev
	}
	f.e.e.EncodeRawExt(re, f.e)
}

func (f *encFnInfo) ext(rv reflect.Value) {
	// if this is a struct|array and it was addressable, then pass the address directly (not the value)
	if k := rv.Kind(); (k == reflect.Struct || k == reflect.Array) && rv.CanAddr() {
		rv = rv.Addr()
	}
	f.e.e.EncodeExt(rv.Interface(), f.xfTag, f.xfFn, f.e)
}

func (f *encFnInfo) getValueForMarshalInterface(rv reflect.Value, indir int8) (v interface{}, proceed bool) {
	if indir == 0 {
		v = rv.Interface()
	} else if indir == -1 {
		// If a non-pointer was passed to Encode(), then that value is not addressable.
		// Take addr if addresable, else copy value to an addressable value.
		if rv.CanAddr() {
			v = rv.Addr().Interface()
		} else {
			rv2 := reflect.New(rv.Type())
			rv2.Elem().Set(rv)
			v = rv2.Interface()
			// fmt.Printf("rv.Type: %v, rv2.Type: %v, v: %v\n", rv.Type(), rv2.Type(), v)
		}
	} else {
		for j := int8(0); j < indir; j++ {
			if rv.IsNil() {
				f.e.e.EncodeNil()
				return
			}
			rv = rv.Elem()
		}
		v = rv.Interface()
	}
	return v, true
}

func (f *encFnInfo) selferMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.csIndir); proceed {
		v.(Selfer).CodecEncodeSelf(f.e)
	}
}

func (f *encFnInfo) binaryMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.bmIndir); proceed {
		bs, fnerr := v.(encoding.BinaryMarshaler).MarshalBinary()
		f.e.marshal(bs, fnerr, false, c_RAW)
	}
}

func (f *encFnInfo) textMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.tmIndir); proceed {
		// debugf(">>>> encoding.TextMarshaler: %T", rv.Interface())
		bs, fnerr := v.(encoding.TextMarshaler).MarshalText()
		f.e.marshal(bs, fnerr, false, c_UTF8)
	}
}

func (f *encFnInfo) jsonMarshal(rv reflect.Value) {
	if v, proceed := f.getValueForMarshalInterface(rv, f.ti.jmIndir); proceed {
		bs, fnerr := v.(jsonMarshaler).MarshalJSON()
		f.e.marshal(bs, fnerr, true, c_UTF8)
	}
}

func (f *encFnInfo) kBool(rv reflect.Value) {
	f.e.e.EncodeBool(rv.Bool())
}

func (f *encFnInfo) kString(rv reflect.Value) {
	f.e.e.EncodeString(c_UTF8, rv.String())
}

func (f *encFnInfo) kFloat64(rv reflect.Value) {
	f.e.e.EncodeFloat64(rv.Float())
}

func (f *encFnInfo) kFloat32(rv reflect.Value) {
	f.e.e.EncodeFloat32(float32(rv.Float()))
}

func (f *encFnInfo) kInt(rv reflect.Value) {
	f.e.e.EncodeInt(rv.Int())
}

func (f *encFnInfo) kUint(rv reflect.Value) {
	f.e.e.EncodeUint(rv.Uint())
}

func (f *encFnInfo) kInvalid(rv reflect.Value) {
	f.e.e.EncodeNil()
}

func (f *encFnInfo) kErr(rv reflect.Value) {
	f.e.errorf("unsupported kind %s, for %#v", rv.Kind(), rv)
}

func (f *encFnInfo) kSlice(rv reflect.Value) {
	ti := f.ti
	// array may be non-addressable, so we have to manage with care
	//   (don't call rv.Bytes, rv.Slice, etc).
	// E.g. type struct S{B [2]byte};
	//   Encode(S{}) will bomb on "panic: slice of unaddressable array".
	e := f.e
	if f.seq != seqTypeArray {
		if rv.IsNil() {
			e.e.EncodeNil()
			return
		}
		// If in this method, then there was no extension function defined.
		// So it's okay to treat as []byte.
		if ti.rtid == uint8SliceTypId {
			e.e.EncodeStringBytes(c_RAW, rv.Bytes())
			return
		}
	}
	cr := e.cr
	rtelem := ti.rt.Elem()
	l := rv.Len()
	if ti.rtid == uint8SliceTypId || rtelem.Kind() == reflect.Uint8 {
		switch f.seq {
		case seqTypeArray:
			// if l == 0 { e.e.encodeStringBytes(c_RAW, nil) } else
			if rv.CanAddr() {
				e.e.EncodeStringBytes(c_RAW, rv.Slice(0, l).Bytes())
			} else {
				var bs []byte
				if l <= cap(e.b) {
					bs = e.b[:l]
				} else {
					bs = make([]byte, l)
				}
				reflect.Copy(reflect.ValueOf(bs), rv)
				// TODO: Test that reflect.Copy works instead of manual one-by-one
				// for i := 0; i < l; i++ {
				// 	bs[i] = byte(rv.Index(i).Uint())
				// }
				e.e.EncodeStringBytes(c_RAW, bs)
			}
		case seqTypeSlice:
			e.e.EncodeStringBytes(c_RAW, rv.Bytes())
		case seqTypeChan:
			bs := e.b[:0]
			// do not use range, so that the number of elements encoded
			// does not change, and encoding does not hang waiting on someone to close chan.
			// for b := range rv.Interface().(<-chan byte) {
			// 	bs = append(bs, b)
			// }
			ch := rv.Interface().(<-chan byte)
			for i := 0; i < l; i++ {
				bs = append(bs, <-ch)
			}
			e.e.EncodeStringBytes(c_RAW, bs)
		}
		return
	}

	if ti.mbs {
		if l%2 == 1 {
			e.errorf("mapBySlice requires even slice length, but got %v", l)
			return
		}
		e.e.EncodeMapStart(l / 2)
	} else {
		e.e.EncodeArrayStart(l)
	}

	if l > 0 {
		for rtelem.Kind() == reflect.Ptr {
			rtelem = rtelem.Elem()
		}
		// if kind is reflect.Interface, do not pre-determine the
		// encoding type, because preEncodeValue may break it down to
		// a concrete type and kInterface will bomb.
		var fn *encFn
		if rtelem.Kind() != reflect.Interface {
			rtelemid := reflect.ValueOf(rtelem).Pointer()
			fn = e.getEncFn(rtelemid, rtelem, true, true)
		}
		// TODO: Consider perf implication of encoding odd index values as symbols if type is string
		for j := 0; j < l; j++ {
			if cr != nil {
				if ti.mbs {
					if l%2 == 0 {
						cr.sendContainerState(containerMapKey)
					} else {
						cr.sendContainerState(containerMapValue)
					}
				} else {
					cr.sendContainerState(containerArrayElem)
				}
			}
			if f.seq == seqTypeChan {
				if rv2, ok2 := rv.Recv(); ok2 {
					e.encodeValue(rv2, fn)
				} else {
					e.encode(nil) // WE HAVE TO DO SOMETHING, so nil if nothing received.
				}
			} else {
				e.encodeValue(rv.Index(j), fn)
			}
		}
	}

	if cr != nil {
		if ti.mbs {
			cr.sendContainerState(containerMapEnd)
		} else {
			cr.sendContainerState(containerArrayEnd)
		}
	}
}

func (f *encFnInfo) kStruct(rv reflect.Value) {
	fti := f.ti
	e := f.e
	cr := e.cr
	tisfi := fti.sfip
	toMap := !(fti.toArray || e.h.StructToArray)
	newlen := len(fti.sfi)

	// Use sync.Pool to reduce allocating slices unnecessarily.
	// The cost of the occasional locking is less than the cost of new allocation.
	pool, poolv, fkvs := encStructPoolGet(newlen)

	// if toMap, use the sorted array. If toArray, use unsorted array (to match sequence in struct)
	if toMap {
		tisfi = fti.sfi
	}
	newlen = 0
	var kv stringRv
	for _, si := range tisfi {
		kv.r = si.field(rv, false)
		// if si.i != -1 {
		// 	rvals[newlen] = rv.Field(int(si.i))
		// } else {
		// 	rvals[newlen] = rv.FieldByIndex(si.is)
		// }
		if toMap {
			if si.omitEmpty && isEmptyValue(kv.r) {
				continue
			}
			kv.v = si.encName
		} else {
			// use the zero value.
			// if a reference or struct, set to nil (so you do not output too much)
			if si.omitEmpty && isEmptyValue(kv.r) {
				switch kv.r.Kind() {
				case reflect.Struct, reflect.Interface, reflect.Ptr, reflect.Array,
					reflect.Map, reflect.Slice:
					kv.r = reflect.Value{} //encode as nil
				}
			}
		}
		fkvs[newlen] = kv
		newlen++
	}

	// debugf(">>>> kStruct: newlen: %v", newlen)
	// sep := !e.be
	ee := e.e //don't dereference everytime

	if toMap {
		ee.EncodeMapStart(newlen)
		// asSymbols := e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		asSymbols := e.h.AsSymbols == AsSymbolDefault || e.h.AsSymbols&AsSymbolStructFieldNameFlag != 0
		for j := 0; j < newlen; j++ {
			kv = fkvs[j]
			if cr != nil {
				cr.sendContainerState(containerMapKey)
			}
			if asSymbols {
				ee.EncodeSymbol(kv.v)
			} else {
				ee.EncodeString(c_UTF8, kv.v)
			}
			if cr != nil {
				cr.sendContainerState(containerMapValue)
			}
			e.encodeValue(kv.r, nil)
		}
		if cr != nil {
			cr.sendContainerState(containerMapEnd)
		}
	} else {
		ee.EncodeArrayStart(newlen)
		for j := 0; j < newlen; j++ {
			kv = fkvs[j]
			if cr != nil {
				cr.sendContainerState(containerArrayElem)
			}
			e.encodeValue(kv.r, nil)
		}
		if cr != nil {
			cr.sendContainerState(containerArrayEnd)
		}
	}

	// do not use defer. Instead, use explicit pool return at end of function.
	// defer has a cost we are trying to avoid.
	// If there is a panic and these slices are not returned, it is ok.
	if pool != nil {
		pool.Put(poolv)
	}
}

// func (f *encFnInfo) kPtr(rv reflect.Value) {
// 	debugf(">>>>>>> ??? encode kPtr called - shouldn't get called")
// 	if rv.IsNil() {
// 		f.e.e.encodeNil()
// 		return
// 	}
// 	f.e.encodeValue(rv.Elem())
// }

func (f *encFnInfo) kInterface(rv reflect.Value) {
	if rv.IsNil() {
		f.e.e.EncodeNil()
		return
	}
	f.e.encodeValue(rv.Elem(), nil)
}

func (f *encFnInfo) kMap(rv reflect.Value) {
	ee := f.e.e
	if rv.IsNil() {
		ee.EncodeNil()
		return
	}

	l := rv.Len()
	ee.EncodeMapStart(l)
	e := f.e
	cr := e.cr
	if l == 0 {
		if cr != nil {
			cr.sendContainerState(containerMapEnd)
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
	var keyFn, valFn *encFn
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

	if e.h.Canonical {
		e.kMapCanonical(rtkeyid, rtkey, rv, mks, valFn, asSymbols)
	} else {
		for j := range mks {
			if cr != nil {
				cr.sendContainerState(containerMapKey)
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
			if cr != nil {
				cr.sendContainerState(containerMapValue)
			}
			e.encodeValue(rv.MapIndex(mks[j]), valFn)
		}
	}
	if cr != nil {
		cr.sendContainerState(containerMapEnd)
	}
}

func (e *Encoder) kMapCanonical(rtkeyid uintptr, rtkey reflect.Type, rv reflect.Value, mks []reflect.Value, valFn *encFn, asSymbols bool) {
	ee := e.e
	cr := e.cr
	// we previously did out-of-band if an extension was registered.
	// This is not necessary, as the natural kind is sufficient for ordering.

	if rtkeyid == uint8SliceTypId {
		mksv := make([]bytesRv, len(mks))
		for i, k := range mks {
			v := &mksv[i]
			v.r = k
			v.v = k.Bytes()
		}
		sort.Sort(bytesRvSlice(mksv))
		for i := range mksv {
			if cr != nil {
				cr.sendContainerState(containerMapKey)
			}
			ee.EncodeStringBytes(c_RAW, mksv[i].v)
			if cr != nil {
				cr.sendContainerState(containerMapValue)
			}
			e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
		}
	} else {
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
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				ee.EncodeBool(mksv[i].v)
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
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
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				if asSymbols {
					ee.EncodeSymbol(mksv[i].v)
				} else {
					ee.EncodeString(c_UTF8, mksv[i].v)
				}
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
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
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				ee.EncodeUint(mksv[i].v)
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
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
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				ee.EncodeInt(mksv[i].v)
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
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
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				ee.EncodeFloat32(float32(mksv[i].v))
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
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
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				ee.EncodeFloat64(mksv[i].v)
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksv[i].r), valFn)
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
				// fmt.Printf(">>>>> %s\n", mksv[l:])
			}
			sort.Sort(bytesRvSlice(mksbv))
			for j := range mksbv {
				if cr != nil {
					cr.sendContainerState(containerMapKey)
				}
				e.asis(mksbv[j].v)
				if cr != nil {
					cr.sendContainerState(containerMapValue)
				}
				e.encodeValue(rv.MapIndex(mksbv[j].r), valFn)
			}
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
	f func(*encFnInfo, reflect.Value)
}

// --------------------------------------------------

type encRtidFn struct {
	rtid uintptr
	fn   encFn
}

// An Encoder writes an object to an output stream in the codec format.
type Encoder struct {
	// hopefully, reduce derefencing cost by laying the encWriter inside the Encoder
	e encDriver
	// NOTE: Encoder shouldn't call it's write methods,
	// as the handler MAY need to do some coordination.
	w  encWriter
	s  []encRtidFn
	be bool // is binary encoding
	js bool // is json handle

	wi ioEncWriter
	wb bytesEncWriter

	h  *BasicHandle
	hh Handle

	cr containerStateRecv
	as encDriverAsis

	f map[uintptr]*encFn
	b [scratchByteArrayLen]byte
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
	e := &Encoder{hh: h, h: h.getBasicHandle(), be: h.isBinary()}
	_, e.js = h.(*JsonHandle)
	e.e = h.newEncDriver(e)
	e.as, _ = e.e.(encDriverAsis)
	e.cr, _ = e.e.(containerStateRecv)
	return e
}

// Reset the Encoder with a new output stream.
//
// This accomodates using the state of the Encoder,
// where it has "cached" information about sub-engines.
func (e *Encoder) Reset(w io.Writer) {
	ww, ok := w.(ioEncWriterWriter)
	if ok {
		e.wi.w = ww
	} else {
		sww := &e.wi.s
		sww.w = w
		sww.bw, _ = w.(io.ByteWriter)
		sww.sw, _ = w.(ioEncStringWriter)
		e.wi.w = sww
		//ww = bufio.NewWriterSize(w, defEncByteBufSize)
	}
	e.w = &e.wi
	e.e.reset()
}

func (e *Encoder) ResetBytes(out *[]byte) {
	in := *out
	if in == nil {
		in = make([]byte, defEncByteBufSize)
	}
	e.wb.b, e.wb.out, e.wb.c = in, out, 0
	e.w = &e.wb
	e.e.reset()
}

// func (e *Encoder) sendContainerState(c containerState) {
// 	if e.cr != nil {
// 		e.cr.sendContainerState(c)
// 	}
// }

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
//          _struct bool    `codec:",omitempty,toarray"`   //set omitempty for every field
//                                                         //and encode struct as an array
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
		e.encodeValue(v, nil)

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
		const checkCodecSelfer1 = true // in case T is passed, where *T is a Selfer, still checkCodecSelfer
		if !fastpathEncodeTypeSwitch(iv, e) {
			e.encodeI(iv, false, checkCodecSelfer1)
		}
	}
}

func (e *Encoder) encodeI(iv interface{}, checkFastpath, checkCodecSelfer bool) {
	if rv, proceed := e.preEncodeValue(reflect.ValueOf(iv)); proceed {
		rt := rv.Type()
		rtid := reflect.ValueOf(rt).Pointer()
		fn := e.getEncFn(rtid, rt, checkFastpath, checkCodecSelfer)
		fn.f(&fn.i, rv)
	}
}

func (e *Encoder) preEncodeValue(rv reflect.Value) (rv2 reflect.Value, proceed bool) {
	// use a goto statement instead of a recursive function for ptr/interface.
TOP:
	switch rv.Kind() {
	case reflect.Ptr, reflect.Interface:
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

	return rv, true
}

func (e *Encoder) encodeValue(rv reflect.Value, fn *encFn) {
	// if a valid fn is passed, it MUST BE for the dereferenced type of rv
	if rv, proceed := e.preEncodeValue(rv); proceed {
		if fn == nil {
			rt := rv.Type()
			rtid := reflect.ValueOf(rt).Pointer()
			fn = e.getEncFn(rtid, rt, true, true)
		}
		fn.f(&fn.i, rv)
	}
}

func (e *Encoder) getEncFn(rtid uintptr, rt reflect.Type, checkFastpath, checkCodecSelfer bool) (fn *encFn) {
	// rtid := reflect.ValueOf(rt).Pointer()
	var ok bool
	if useMapForCodecCache {
		fn, ok = e.f[rtid]
	} else {
		for i := range e.s {
			v := &(e.s[i])
			if v.rtid == rtid {
				fn, ok = &(v.fn), true
				break
			}
		}
	}
	if ok {
		return
	}

	if useMapForCodecCache {
		if e.f == nil {
			e.f = make(map[uintptr]*encFn, initCollectionCap)
		}
		fn = new(encFn)
		e.f[rtid] = fn
	} else {
		if e.s == nil {
			e.s = make([]encRtidFn, 0, initCollectionCap)
		}
		e.s = append(e.s, encRtidFn{rtid: rtid})
		fn = &(e.s[len(e.s)-1]).fn
	}

	ti := e.h.getTypeInfo(rtid, rt)
	fi := &(fn.i)
	fi.e = e
	fi.ti = ti

	if checkCodecSelfer && ti.cs {
		fn.f = (*encFnInfo).selferMarshal
	} else if rtid == rawExtTypId {
		fn.f = (*encFnInfo).rawExt
	} else if e.e.IsBuiltinType(rtid) {
		fn.f = (*encFnInfo).builtin
	} else if xfFn := e.h.getExt(rtid); xfFn != nil {
		fi.xfTag, fi.xfFn = xfFn.tag, xfFn.ext
		fn.f = (*encFnInfo).ext
	} else if supportMarshalInterfaces && e.be && ti.bm {
		fn.f = (*encFnInfo).binaryMarshal
	} else if supportMarshalInterfaces && !e.be && e.js && ti.jm {
		//If JSON, we should check JSONMarshal before textMarshal
		fn.f = (*encFnInfo).jsonMarshal
	} else if supportMarshalInterfaces && !e.be && ti.tm {
		fn.f = (*encFnInfo).textMarshal
	} else {
		rk := rt.Kind()
		if fastpathEnabled && checkFastpath && (rk == reflect.Map || rk == reflect.Slice) {
			if rt.PkgPath() == "" {
				if idx := fastpathAV.index(rtid); idx != -1 {
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
					fn.f = func(xf *encFnInfo, xrv reflect.Value) {
						xfnf(xf, xrv.Convert(xrt))
					}
				}
			}
		}
		if fn.f == nil {
			switch rk {
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
			case reflect.Uint8, reflect.Uint64, reflect.Uint, reflect.Uint32, reflect.Uint16, reflect.Uintptr:
				fn.f = (*encFnInfo).kUint
			case reflect.Invalid:
				fn.f = (*encFnInfo).kInvalid
			case reflect.Chan:
				fi.seq = seqTypeChan
				fn.f = (*encFnInfo).kSlice
			case reflect.Slice:
				fi.seq = seqTypeSlice
				fn.f = (*encFnInfo).kSlice
			case reflect.Array:
				fi.seq = seqTypeArray
				fn.f = (*encFnInfo).kSlice
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
	}

	return
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

func (e *Encoder) errorf(format string, params ...interface{}) {
	err := fmt.Errorf(format, params...)
	panic(err)
}

// ----------------------------------------

const encStructPoolLen = 5

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
	encStructPool[0].New = func() interface{} { return new([8]stringRv) }
	encStructPool[1].New = func() interface{} { return new([16]stringRv) }
	encStructPool[2].New = func() interface{} { return new([32]stringRv) }
	encStructPool[3].New = func() interface{} { return new([64]stringRv) }
	encStructPool[4].New = func() interface{} { return new([128]stringRv) }
}

func encStructPoolGet(newlen int) (p *sync.Pool, v interface{}, s []stringRv) {
	// if encStructPoolLen != 5 { // constant chec, so removed at build time.
	// 	panic(errors.New("encStructPoolLen must be equal to 4")) // defensive, in case it is changed
	// }
	// idxpool := newlen / 8

	// if pool == nil {
	// 	fkvs = make([]stringRv, newlen)
	// } else {
	// 	poolv = pool.Get()
	// 	switch vv := poolv.(type) {
	// 	case *[8]stringRv:
	// 		fkvs = vv[:newlen]
	// 	case *[16]stringRv:
	// 		fkvs = vv[:newlen]
	// 	case *[32]stringRv:
	// 		fkvs = vv[:newlen]
	// 	case *[64]stringRv:
	// 		fkvs = vv[:newlen]
	// 	case *[128]stringRv:
	// 		fkvs = vv[:newlen]
	// 	}
	// }

	if newlen <= 8 {
		p = &encStructPool[0]
		v = p.Get()
		s = v.(*[8]stringRv)[:newlen]
	} else if newlen <= 16 {
		p = &encStructPool[1]
		v = p.Get()
		s = v.(*[16]stringRv)[:newlen]
	} else if newlen <= 32 {
		p = &encStructPool[2]
		v = p.Get()
		s = v.(*[32]stringRv)[:newlen]
	} else if newlen <= 64 {
		p = &encStructPool[3]
		v = p.Get()
		s = v.(*[64]stringRv)[:newlen]
	} else if newlen <= 128 {
		p = &encStructPool[4]
		v = p.Get()
		s = v.(*[128]stringRv)[:newlen]
	} else {
		s = make([]stringRv, newlen)
	}
	return
}

// ----------------------------------------

// func encErr(format string, params ...interface{}) {
// 	doPanic(msgTagEnc, format, params...)
// }
