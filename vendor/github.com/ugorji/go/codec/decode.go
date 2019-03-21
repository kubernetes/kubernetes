// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"encoding"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sync"
	"time"
)

// Some tagging information for error messages.
const (
	msgBadDesc            = "Unrecognized descriptor byte"
	msgDecCannotExpandArr = "cannot expand go array from %v to stream length: %v"
)

var (
	onlyMapOrArrayCanDecodeIntoStructErr = errors.New("only encoded map or array can be decoded into a struct")
	cannotDecodeIntoNilErr               = errors.New("cannot decode into nil")

	decUnreadByteNothingToReadErr   = errors.New("cannot unread - nothing has been read")
	decUnreadByteLastByteNotReadErr = errors.New("cannot unread - last byte has not been read")
	decUnreadByteUnknownErr         = errors.New("cannot unread - reason unknown")
)

// decReader abstracts the reading source, allowing implementations that can
// read from an io.Reader or directly off a byte slice with zero-copying.
type decReader interface {
	unreadn1()

	// readx will use the implementation scratch buffer if possible i.e. n < len(scratchbuf), OR
	// just return a view of the []byte being decoded from.
	// Ensure you call detachZeroCopyBytes later if this needs to be sent outside codec control.
	readx(n int) []byte
	readb([]byte)
	readn1() uint8
	readn3() (uint8, uint8, uint8)
	readn4() (uint8, uint8, uint8, uint8)
	// readn1eof() (v uint8, eof bool)
	numread() int // number of bytes read
	track()
	stopTrack() []byte

	// skip will skip any byte that matches, and return the first non-matching byte
	skip(accept *bitset256) (token byte)
	// readTo will read any byte that matches, stopping once no-longer matching.
	readTo(in []byte, accept *bitset256) (out []byte)
	// readUntil will read, only stopping once it matches the 'stop' byte.
	readUntil(in []byte, stop byte) (out []byte)
}

// type decReaderByteScanner interface {
// 	io.Reader
// 	io.ByteScanner
// }

type decDriver interface {
	// this will check if the next token is a break.
	CheckBreak() bool
	// Note: TryDecodeAsNil should be careful not to share any temporary []byte with
	// the rest of the decDriver. This is because sometimes, we optimize by holding onto
	// a transient []byte, and ensuring the only other call we make to the decDriver
	// during that time is maybe a TryDecodeAsNil() call.
	TryDecodeAsNil() bool
	// vt is one of: Bytes, String, Nil, Slice or Map. Return unSet if not known.
	ContainerType() (vt valueType)
	// IsBuiltinType(rt uintptr) bool
	DecodeBuiltin(rt uintptr, v interface{})

	// DecodeNaked will decode primitives (number, bool, string, []byte) and RawExt.
	// For maps and arrays, it will not do the decoding in-band, but will signal
	// the decoder, so that is done later, by setting the decNaked.valueType field.
	//
	// Note: Numbers are decoded as int64, uint64, float64 only (no smaller sized number types).
	// for extensions, DecodeNaked must read the tag and the []byte if it exists.
	// if the []byte is not read, then kInterfaceNaked will treat it as a Handle
	// that stores the subsequent value in-band, and complete reading the RawExt.
	//
	// extensions should also use readx to decode them, for efficiency.
	// kInterface will extract the detached byte slice if it has to pass it outside its realm.
	DecodeNaked()
	DecodeInt(bitsize uint8) (i int64)
	DecodeUint(bitsize uint8) (ui uint64)
	DecodeFloat(chkOverflow32 bool) (f float64)
	DecodeBool() (b bool)
	// DecodeString can also decode symbols.
	// It looks redundant as DecodeBytes is available.
	// However, some codecs (e.g. binc) support symbols and can
	// return a pre-stored string value, meaning that it can bypass
	// the cost of []byte->string conversion.
	DecodeString() (s string)
	DecodeStringAsBytes() (v []byte)

	// DecodeBytes may be called directly, without going through reflection.
	// Consequently, it must be designed to handle possible nil.
	DecodeBytes(bs []byte, zerocopy bool) (bsOut []byte)
	// DecodeBytes(bs []byte, isstring, zerocopy bool) (bsOut []byte)

	// decodeExt will decode into a *RawExt or into an extension.
	DecodeExt(v interface{}, xtag uint64, ext Ext) (realxtag uint64)
	// decodeExt(verifyTag bool, tag byte) (xtag byte, xbs []byte)
	ReadArrayStart() int
	ReadArrayElem()
	ReadArrayEnd()
	ReadMapStart() int
	ReadMapElemKey()
	ReadMapElemValue()
	ReadMapEnd()

	reset()
	uncacheRead()
}

// type decNoSeparator struct {}
// func (_ decNoSeparator) ReadEnd() {}

type decDriverNoopContainerReader struct{}

func (_ decDriverNoopContainerReader) ReadArrayStart() (v int) { return }
func (_ decDriverNoopContainerReader) ReadArrayElem()          {}
func (_ decDriverNoopContainerReader) ReadArrayEnd()           {}
func (_ decDriverNoopContainerReader) ReadMapStart() (v int)   { return }
func (_ decDriverNoopContainerReader) ReadMapElemKey()         {}
func (_ decDriverNoopContainerReader) ReadMapElemValue()       {}
func (_ decDriverNoopContainerReader) ReadMapEnd()             {}
func (_ decDriverNoopContainerReader) CheckBreak() (v bool)    { return }

// func (_ decNoSeparator) uncacheRead() {}

type DecodeOptions struct {
	// MapType specifies type to use during schema-less decoding of a map in the stream.
	// If nil, we use map[interface{}]interface{}
	MapType reflect.Type

	// SliceType specifies type to use during schema-less decoding of an array in the stream.
	// If nil, we use []interface{}
	SliceType reflect.Type

	// MaxInitLen defines the maxinum initial length that we "make" a collection (string, slice, map, chan).
	// If 0 or negative, we default to a sensible value based on the size of an element in the collection.
	//
	// For example, when decoding, a stream may say that it has 2^64 elements.
	// We should not auto-matically provision a slice of that length, to prevent Out-Of-Memory crash.
	// Instead, we provision up to MaxInitLen, fill that up, and start appending after that.
	MaxInitLen int

	// If ErrorIfNoField, return an error when decoding a map
	// from a codec stream into a struct, and no matching struct field is found.
	ErrorIfNoField bool

	// If ErrorIfNoArrayExpand, return an error when decoding a slice/array that cannot be expanded.
	// For example, the stream contains an array of 8 items, but you are decoding into a [4]T array,
	// or you are decoding into a slice of length 4 which is non-addressable (and so cannot be set).
	ErrorIfNoArrayExpand bool

	// If SignedInteger, use the int64 during schema-less decoding of unsigned values (not uint64).
	SignedInteger bool

	// MapValueReset controls how we decode into a map value.
	//
	// By default, we MAY retrieve the mapping for a key, and then decode into that.
	// However, especially with big maps, that retrieval may be expensive and unnecessary
	// if the stream already contains all that is necessary to recreate the value.
	//
	// If true, we will never retrieve the previous mapping,
	// but rather decode into a new value and set that in the map.
	//
	// If false, we will retrieve the previous mapping if necessary e.g.
	// the previous mapping is a pointer, or is a struct or array with pre-set state,
	// or is an interface.
	MapValueReset bool

	// SliceElementReset: on decoding a slice, reset the element to a zero value first.
	//
	// concern: if the slice already contained some garbage, we will decode into that garbage.
	SliceElementReset bool

	// InterfaceReset controls how we decode into an interface.
	//
	// By default, when we see a field that is an interface{...},
	// or a map with interface{...} value, we will attempt decoding into the
	// "contained" value.
	//
	// However, this prevents us from reading a string into an interface{}
	// that formerly contained a number.
	//
	// If true, we will decode into a new "blank" value, and set that in the interface.
	// If false, we will decode into whatever is contained in the interface.
	InterfaceReset bool

	// InternString controls interning of strings during decoding.
	//
	// Some handles, e.g. json, typically will read map keys as strings.
	// If the set of keys are finite, it may help reduce allocation to
	// look them up from a map (than to allocate them afresh).
	//
	// Note: Handles will be smart when using the intern functionality.
	// Every string should not be interned.
	// An excellent use-case for interning is struct field names,
	// or map keys where key type is string.
	InternString bool

	// PreferArrayOverSlice controls whether to decode to an array or a slice.
	//
	// This only impacts decoding into a nil interface{}.
	// Consequently, it has no effect on codecgen.
	//
	// *Note*: This only applies if using go1.5 and above,
	// as it requires reflect.ArrayOf support which was absent before go1.5.
	PreferArrayOverSlice bool

	// DeleteOnNilMapValue controls how to decode a nil value in the stream.
	//
	// If true, we will delete the mapping of the key.
	// Else, just set the mapping to the zero value of the type.
	DeleteOnNilMapValue bool

	// ReaderBufferSize is the size of the buffer used when reading.
	//
	// if > 0, we use a smart buffer internally for performance purposes.
	ReaderBufferSize int
}

// ------------------------------------

type bufioDecReader struct {
	buf []byte
	r   io.Reader

	c   int // cursor
	n   int // num read
	err error

	trb bool
	tr  []byte

	b [8]byte
}

func (z *bufioDecReader) reset(r io.Reader) {
	z.r, z.c, z.n, z.err, z.trb = r, 0, 0, nil, false
	if z.tr != nil {
		z.tr = z.tr[:0]
	}
}

func (z *bufioDecReader) Read(p []byte) (n int, err error) {
	if z.err != nil {
		return 0, z.err
	}
	p0 := p
	n = copy(p, z.buf[z.c:])
	z.c += n
	if z.c == len(z.buf) {
		z.c = 0
	}
	z.n += n
	if len(p) == n {
		if z.c == 0 {
			z.buf = z.buf[:1]
			z.buf[0] = p[len(p)-1]
			z.c = 1
		}
		if z.trb {
			z.tr = append(z.tr, p0[:n]...)
		}
		return
	}
	p = p[n:]
	var n2 int
	// if we are here, then z.buf is all read
	if len(p) > len(z.buf) {
		n2, err = decReadFull(z.r, p)
		n += n2
		z.n += n2
		z.err = err
		// don't return EOF if some bytes were read. keep for next time.
		if n > 0 && err == io.EOF {
			err = nil
		}
		// always keep last byte in z.buf
		z.buf = z.buf[:1]
		z.buf[0] = p[len(p)-1]
		z.c = 1
		if z.trb {
			z.tr = append(z.tr, p0[:n]...)
		}
		return
	}
	// z.c is now 0, and len(p) <= len(z.buf)
	for len(p) > 0 && z.err == nil {
		// println("len(p) loop starting ... ")
		z.c = 0
		z.buf = z.buf[0:cap(z.buf)]
		n2, err = z.r.Read(z.buf)
		if n2 > 0 {
			if err == io.EOF {
				err = nil
			}
			z.buf = z.buf[:n2]
			n2 = copy(p, z.buf)
			z.c = n2
			n += n2
			z.n += n2
			p = p[n2:]
		}
		z.err = err
		// println("... len(p) loop done")
	}
	if z.c == 0 {
		z.buf = z.buf[:1]
		z.buf[0] = p[len(p)-1]
		z.c = 1
	}
	if z.trb {
		z.tr = append(z.tr, p0[:n]...)
	}
	return
}

func (z *bufioDecReader) ReadByte() (b byte, err error) {
	z.b[0] = 0
	_, err = z.Read(z.b[:1])
	b = z.b[0]
	return
}

func (z *bufioDecReader) UnreadByte() (err error) {
	if z.err != nil {
		return z.err
	}
	if z.c > 0 {
		z.c--
		z.n--
		if z.trb {
			z.tr = z.tr[:len(z.tr)-1]
		}
		return
	}
	return decUnreadByteNothingToReadErr
}

func (z *bufioDecReader) numread() int {
	return z.n
}

func (z *bufioDecReader) readx(n int) (bs []byte) {
	if n <= 0 || z.err != nil {
		return
	}
	if z.c+n <= len(z.buf) {
		bs = z.buf[z.c : z.c+n]
		z.n += n
		z.c += n
		if z.trb {
			z.tr = append(z.tr, bs...)
		}
		return
	}
	bs = make([]byte, n)
	_, err := z.Read(bs)
	if err != nil {
		panic(err)
	}
	return
}

func (z *bufioDecReader) readb(bs []byte) {
	_, err := z.Read(bs)
	if err != nil {
		panic(err)
	}
}

// func (z *bufioDecReader) readn1eof() (b uint8, eof bool) {
// 	b, err := z.ReadByte()
// 	if err != nil {
// 		if err == io.EOF {
// 			eof = true
// 		} else {
// 			panic(err)
// 		}
// 	}
// 	return
// }

func (z *bufioDecReader) readn1() (b uint8) {
	b, err := z.ReadByte()
	if err != nil {
		panic(err)
	}
	return
}

func (z *bufioDecReader) readn3() (b1, b2, b3 uint8) {
	z.readb(z.b[:3])
	return z.b[0], z.b[1], z.b[2]
}

func (z *bufioDecReader) readn4() (b1, b2, b3, b4 uint8) {
	z.readb(z.b[:4])
	return z.b[0], z.b[1], z.b[2], z.b[3]
}

func (z *bufioDecReader) search(in []byte, accept *bitset256, stop, flag uint8) (token byte, out []byte) {
	// flag: 1 (skip), 2 (readTo), 4 (readUntil)
	if flag == 4 {
		for i := z.c; i < len(z.buf); i++ {
			if z.buf[i] == stop {
				token = z.buf[i]
				z.n = z.n + (i - z.c) - 1
				i++
				out = z.buf[z.c:i]
				if z.trb {
					z.tr = append(z.tr, z.buf[z.c:i]...)
				}
				z.c = i
				return
			}
		}
	} else {
		for i := z.c; i < len(z.buf); i++ {
			if !accept.isset(z.buf[i]) {
				token = z.buf[i]
				z.n = z.n + (i - z.c) - 1
				if flag == 1 {
					i++
				} else {
					out = z.buf[z.c:i]
				}
				if z.trb {
					z.tr = append(z.tr, z.buf[z.c:i]...)
				}
				z.c = i
				return
			}
		}
	}
	z.n += len(z.buf) - z.c
	if flag != 1 {
		out = append(in, z.buf[z.c:]...)
	}
	if z.trb {
		z.tr = append(z.tr, z.buf[z.c:]...)
	}
	var n2 int
	if z.err != nil {
		return
	}
	for {
		z.c = 0
		z.buf = z.buf[0:cap(z.buf)]
		n2, z.err = z.r.Read(z.buf)
		if n2 > 0 && z.err != nil {
			z.err = nil
		}
		z.buf = z.buf[:n2]
		if flag == 4 {
			for i := 0; i < n2; i++ {
				if z.buf[i] == stop {
					token = z.buf[i]
					z.n += i - 1
					i++
					out = append(out, z.buf[z.c:i]...)
					if z.trb {
						z.tr = append(z.tr, z.buf[z.c:i]...)
					}
					z.c = i
					return
				}
			}
		} else {
			for i := 0; i < n2; i++ {
				if !accept.isset(z.buf[i]) {
					token = z.buf[i]
					z.n += i - 1
					if flag == 1 {
						i++
					}
					if flag != 1 {
						out = append(out, z.buf[z.c:i]...)
					}
					if z.trb {
						z.tr = append(z.tr, z.buf[z.c:i]...)
					}
					z.c = i
					return
				}
			}
		}
		if flag != 1 {
			out = append(out, z.buf[:n2]...)
		}
		z.n += n2
		if z.err != nil {
			return
		}
		if z.trb {
			z.tr = append(z.tr, z.buf[:n2]...)
		}
	}
}

func (z *bufioDecReader) skip(accept *bitset256) (token byte) {
	token, _ = z.search(nil, accept, 0, 1)
	return
}

func (z *bufioDecReader) readTo(in []byte, accept *bitset256) (out []byte) {
	_, out = z.search(in, accept, 0, 2)
	return
}

func (z *bufioDecReader) readUntil(in []byte, stop byte) (out []byte) {
	_, out = z.search(in, nil, stop, 4)
	return
}

func (z *bufioDecReader) unreadn1() {
	err := z.UnreadByte()
	if err != nil {
		panic(err)
	}
}

func (z *bufioDecReader) track() {
	if z.tr != nil {
		z.tr = z.tr[:0]
	}
	z.trb = true
}

func (z *bufioDecReader) stopTrack() (bs []byte) {
	z.trb = false
	return z.tr
}

// ioDecReader is a decReader that reads off an io.Reader.
//
// It also has a fallback implementation of ByteScanner if needed.
type ioDecReader struct {
	r io.Reader // the reader passed in

	rr io.Reader
	br io.ByteScanner

	l   byte    // last byte
	ls  byte    // last byte status. 0: init-canDoNothing, 1: canRead, 2: canUnread
	b   [4]byte // tiny buffer for reading single bytes
	trb bool    // tracking bytes turned on

	// temp byte array re-used internally for efficiency during read.
	// shares buffer with Decoder, so we keep size of struct within 8 words.
	x  *[scratchByteArrayLen]byte
	n  int    // num read
	tr []byte // tracking bytes read
}

func (z *ioDecReader) reset(r io.Reader) {
	z.r = r
	z.rr = r
	z.l, z.ls, z.n, z.trb = 0, 0, 0, false
	if z.tr != nil {
		z.tr = z.tr[:0]
	}
	var ok bool
	if z.br, ok = r.(io.ByteScanner); !ok {
		z.br = z
		z.rr = z
	}
}

func (z *ioDecReader) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return
	}
	var firstByte bool
	if z.ls == 1 {
		z.ls = 2
		p[0] = z.l
		if len(p) == 1 {
			n = 1
			return
		}
		firstByte = true
		p = p[1:]
	}
	n, err = z.r.Read(p)
	if n > 0 {
		if err == io.EOF && n == len(p) {
			err = nil // read was successful, so postpone EOF (till next time)
		}
		z.l = p[n-1]
		z.ls = 2
	}
	if firstByte {
		n++
	}
	return
}

func (z *ioDecReader) ReadByte() (c byte, err error) {
	n, err := z.Read(z.b[:1])
	if n == 1 {
		c = z.b[0]
		if err == io.EOF {
			err = nil // read was successful, so postpone EOF (till next time)
		}
	}
	return
}

func (z *ioDecReader) UnreadByte() (err error) {
	switch z.ls {
	case 2:
		z.ls = 1
	case 0:
		err = decUnreadByteNothingToReadErr
	case 1:
		err = decUnreadByteLastByteNotReadErr
	default:
		err = decUnreadByteUnknownErr
	}
	return
}

func (z *ioDecReader) numread() int {
	return z.n
}

func (z *ioDecReader) readx(n int) (bs []byte) {
	if n <= 0 {
		return
	}
	if n < len(z.x) {
		bs = z.x[:n]
	} else {
		bs = make([]byte, n)
	}
	if _, err := decReadFull(z.rr, bs); err != nil {
		panic(err)
	}
	z.n += len(bs)
	if z.trb {
		z.tr = append(z.tr, bs...)
	}
	return
}

func (z *ioDecReader) readb(bs []byte) {
	// if len(bs) == 0 {
	// 	return
	// }
	if _, err := decReadFull(z.rr, bs); err != nil {
		panic(err)
	}
	z.n += len(bs)
	if z.trb {
		z.tr = append(z.tr, bs...)
	}
}

func (z *ioDecReader) readn1eof() (b uint8, eof bool) {
	b, err := z.br.ReadByte()
	if err == nil {
		z.n++
		if z.trb {
			z.tr = append(z.tr, b)
		}
	} else if err == io.EOF {
		eof = true
	} else {
		panic(err)
	}
	return
}

func (z *ioDecReader) readn1() (b uint8) {
	var err error
	if b, err = z.br.ReadByte(); err == nil {
		z.n++
		if z.trb {
			z.tr = append(z.tr, b)
		}
		return
	}
	panic(err)
}

func (z *ioDecReader) readn3() (b1, b2, b3 uint8) {
	z.readb(z.b[:3])
	return z.b[0], z.b[1], z.b[2]
}

func (z *ioDecReader) readn4() (b1, b2, b3, b4 uint8) {
	z.readb(z.b[:4])
	return z.b[0], z.b[1], z.b[2], z.b[3]
}

func (z *ioDecReader) skip(accept *bitset256) (token byte) {
	for {
		var eof bool
		token, eof = z.readn1eof()
		if eof {
			return
		}
		if accept.isset(token) {
			continue
		}
		return
	}
}

func (z *ioDecReader) readTo(in []byte, accept *bitset256) (out []byte) {
	out = in
	for {
		token, eof := z.readn1eof()
		if eof {
			return
		}
		if accept.isset(token) {
			out = append(out, token)
		} else {
			z.unreadn1()
			return
		}
	}
}

func (z *ioDecReader) readUntil(in []byte, stop byte) (out []byte) {
	out = in
	for {
		token, eof := z.readn1eof()
		if eof {
			panic(io.EOF)
		}
		out = append(out, token)
		if token == stop {
			return
		}
	}
}

func (z *ioDecReader) unreadn1() {
	err := z.br.UnreadByte()
	if err != nil {
		panic(err)
	}
	z.n--
	if z.trb {
		if l := len(z.tr) - 1; l >= 0 {
			z.tr = z.tr[:l]
		}
	}
}

func (z *ioDecReader) track() {
	if z.tr != nil {
		z.tr = z.tr[:0]
	}
	z.trb = true
}

func (z *ioDecReader) stopTrack() (bs []byte) {
	z.trb = false
	return z.tr
}

// ------------------------------------

var bytesDecReaderCannotUnreadErr = errors.New("cannot unread last byte read")

// bytesDecReader is a decReader that reads off a byte slice with zero copying
type bytesDecReader struct {
	b []byte // data
	c int    // cursor
	a int    // available
	t int    // track start
}

func (z *bytesDecReader) reset(in []byte) {
	z.b = in
	z.a = len(in)
	z.c = 0
	z.t = 0
}

func (z *bytesDecReader) numread() int {
	return z.c
}

func (z *bytesDecReader) unreadn1() {
	if z.c == 0 || len(z.b) == 0 {
		panic(bytesDecReaderCannotUnreadErr)
	}
	z.c--
	z.a++
	return
}

func (z *bytesDecReader) readx(n int) (bs []byte) {
	// slicing from a non-constant start position is more expensive,
	// as more computation is required to decipher the pointer start position.
	// However, we do it only once, and it's better than reslicing both z.b and return value.

	if n <= 0 {
	} else if z.a == 0 {
		panic(io.EOF)
	} else if n > z.a {
		panic(io.ErrUnexpectedEOF)
	} else {
		c0 := z.c
		z.c = c0 + n
		z.a = z.a - n
		bs = z.b[c0:z.c]
	}
	return
}

func (z *bytesDecReader) readb(bs []byte) {
	copy(bs, z.readx(len(bs)))
}

func (z *bytesDecReader) readn1() (v uint8) {
	if z.a == 0 {
		panic(io.EOF)
	}
	v = z.b[z.c]
	z.c++
	z.a--
	return
}

func (z *bytesDecReader) readn3() (b1, b2, b3 uint8) {
	if 3 > z.a {
		panic(io.ErrUnexpectedEOF)
	}
	b3 = z.b[z.c+2]
	b2 = z.b[z.c+1]
	b1 = z.b[z.c]
	z.c += 3
	z.a -= 3
	return
}

func (z *bytesDecReader) readn4() (b1, b2, b3, b4 uint8) {
	if 4 > z.a {
		panic(io.ErrUnexpectedEOF)
	}
	b4 = z.b[z.c+3]
	b3 = z.b[z.c+2]
	b2 = z.b[z.c+1]
	b1 = z.b[z.c]
	z.c += 4
	z.a -= 4
	return
}

// func (z *bytesDecReader) readn1eof() (v uint8, eof bool) {
// 	if z.a == 0 {
// 		eof = true
// 		return
// 	}
// 	v = z.b[z.c]
// 	z.c++
// 	z.a--
// 	return
// }

func (z *bytesDecReader) skip(accept *bitset256) (token byte) {
	if z.a == 0 {
		return
	}
	blen := len(z.b)
	for i := z.c; i < blen; i++ {
		if !accept.isset(z.b[i]) {
			token = z.b[i]
			i++
			z.a -= (i - z.c)
			z.c = i
			return
		}
	}
	z.a, z.c = 0, blen
	return
}

func (z *bytesDecReader) readTo(_ []byte, accept *bitset256) (out []byte) {
	if z.a == 0 {
		return
	}
	blen := len(z.b)
	for i := z.c; i < blen; i++ {
		if !accept.isset(z.b[i]) {
			out = z.b[z.c:i]
			z.a -= (i - z.c)
			z.c = i
			return
		}
	}
	out = z.b[z.c:]
	z.a, z.c = 0, blen
	return
}

func (z *bytesDecReader) readUntil(_ []byte, stop byte) (out []byte) {
	if z.a == 0 {
		panic(io.EOF)
	}
	blen := len(z.b)
	for i := z.c; i < blen; i++ {
		if z.b[i] == stop {
			i++
			out = z.b[z.c:i]
			z.a -= (i - z.c)
			z.c = i
			return
		}
	}
	z.a, z.c = 0, blen
	panic(io.EOF)
}

func (z *bytesDecReader) track() {
	z.t = z.c
}

func (z *bytesDecReader) stopTrack() (bs []byte) {
	return z.b[z.t:z.c]
}

// ----------------------------------------

func (d *Decoder) builtin(f *codecFnInfo, rv reflect.Value) {
	d.d.DecodeBuiltin(f.ti.rtid, rv2i(rv))
}

func (d *Decoder) rawExt(f *codecFnInfo, rv reflect.Value) {
	d.d.DecodeExt(rv2i(rv), 0, nil)
}

func (d *Decoder) ext(f *codecFnInfo, rv reflect.Value) {
	d.d.DecodeExt(rv2i(rv), f.xfTag, f.xfFn)
}

func (d *Decoder) getValueForUnmarshalInterface(rv reflect.Value, indir int8) (v interface{}) {
	if indir == -1 {
		v = rv2i(rv.Addr())
	} else if indir == 0 {
		v = rv2i(rv)
	} else {
		for j := int8(0); j < indir; j++ {
			if rv.IsNil() {
				rv.Set(reflect.New(rv.Type().Elem()))
			}
			rv = rv.Elem()
		}
		v = rv2i(rv)
	}
	return
}

func (d *Decoder) selferUnmarshal(f *codecFnInfo, rv reflect.Value) {
	d.getValueForUnmarshalInterface(rv, f.ti.csIndir).(Selfer).CodecDecodeSelf(d)
}

func (d *Decoder) binaryUnmarshal(f *codecFnInfo, rv reflect.Value) {
	bm := d.getValueForUnmarshalInterface(rv, f.ti.bunmIndir).(encoding.BinaryUnmarshaler)
	xbs := d.d.DecodeBytes(nil, true)
	if fnerr := bm.UnmarshalBinary(xbs); fnerr != nil {
		panic(fnerr)
	}
}

func (d *Decoder) textUnmarshal(f *codecFnInfo, rv reflect.Value) {
	tm := d.getValueForUnmarshalInterface(rv, f.ti.tunmIndir).(encoding.TextUnmarshaler)
	fnerr := tm.UnmarshalText(d.d.DecodeStringAsBytes())
	if fnerr != nil {
		panic(fnerr)
	}
}

func (d *Decoder) jsonUnmarshal(f *codecFnInfo, rv reflect.Value) {
	tm := d.getValueForUnmarshalInterface(rv, f.ti.junmIndir).(jsonUnmarshaler)
	// bs := d.d.DecodeBytes(d.b[:], true, true)
	// grab the bytes to be read, as UnmarshalJSON needs the full JSON so as to unmarshal it itself.
	fnerr := tm.UnmarshalJSON(d.nextValueBytes())
	if fnerr != nil {
		panic(fnerr)
	}
}

func (d *Decoder) kErr(f *codecFnInfo, rv reflect.Value) {
	d.errorf("no decoding function defined for kind %v", rv.Kind())
}

// var kIntfCtr uint64

func (d *Decoder) kInterfaceNaked(f *codecFnInfo) (rvn reflect.Value) {
	// nil interface:
	// use some hieristics to decode it appropriately
	// based on the detected next value in the stream.
	n := d.naked()
	d.d.DecodeNaked()
	if n.v == valueTypeNil {
		return
	}
	// We cannot decode non-nil stream value into nil interface with methods (e.g. io.Reader).
	// if num := f.ti.rt.NumMethod(); num > 0 {
	if f.ti.numMeth > 0 {
		d.errorf("cannot decode non-nil codec value into nil %v (%v methods)", f.ti.rt, f.ti.numMeth)
		return
	}
	// var useRvn bool
	switch n.v {
	case valueTypeMap:
		if d.mtid == 0 || d.mtid == mapIntfIntfTypId {
			if n.lm < arrayCacheLen {
				n.ma[n.lm] = nil
				rvn = n.rr[decNakedMapIntfIntfIdx*arrayCacheLen+n.lm]
				n.lm++
				d.decode(&n.ma[n.lm-1])
				n.lm--
			} else {
				var v2 map[interface{}]interface{}
				d.decode(&v2)
				rvn = reflect.ValueOf(&v2).Elem()
			}
		} else if d.mtid == mapStrIntfTypId { // for json performance
			if n.ln < arrayCacheLen {
				n.na[n.ln] = nil
				rvn = n.rr[decNakedMapStrIntfIdx*arrayCacheLen+n.ln]
				n.ln++
				d.decode(&n.na[n.ln-1])
				n.ln--
			} else {
				var v2 map[string]interface{}
				d.decode(&v2)
				rvn = reflect.ValueOf(&v2).Elem()
			}
		} else {
			rvn = reflect.New(d.h.MapType)
			if useLookupRecognizedTypes && d.mtr { // isRecognizedRtid(d.mtid) {
				d.decode(rv2i(rvn))
				rvn = rvn.Elem()
			} else {
				rvn = rvn.Elem()
				d.decodeValue(rvn, nil, false, true)
			}
		}
	case valueTypeArray:
		if d.stid == 0 || d.stid == intfSliceTypId {
			if n.ls < arrayCacheLen {
				n.sa[n.ls] = nil
				rvn = n.rr[decNakedSliceIntfIdx*arrayCacheLen+n.ls]
				n.ls++
				d.decode(&n.sa[n.ls-1])
				n.ls--
			} else {
				var v2 []interface{}
				d.decode(&v2)
				rvn = reflect.ValueOf(&v2).Elem()
			}
			if reflectArrayOfSupported && d.stid == 0 && d.h.PreferArrayOverSlice {
				rvn2 := reflect.New(reflectArrayOf(rvn.Len(), intfTyp)).Elem()
				reflect.Copy(rvn2, rvn)
				rvn = rvn2
			}
		} else {
			rvn = reflect.New(d.h.SliceType)
			if useLookupRecognizedTypes && d.str { // isRecognizedRtid(d.stid) {
				d.decode(rv2i(rvn))
				rvn = rvn.Elem()
			} else {
				rvn = rvn.Elem()
				d.decodeValue(rvn, nil, false, true)
			}
		}
	case valueTypeExt:
		var v interface{}
		tag, bytes := n.u, n.l // calling decode below might taint the values
		if bytes == nil {
			if n.li < arrayCacheLen {
				n.ia[n.li] = nil
				n.li++
				d.decode(&n.ia[n.li-1])
				// v = *(&n.ia[l])
				n.li--
				v = n.ia[n.li]
				n.ia[n.li] = nil
			} else {
				d.decode(&v)
			}
		}
		bfn := d.h.getExtForTag(tag)
		if bfn == nil {
			var re RawExt
			re.Tag = tag
			re.Data = detachZeroCopyBytes(d.bytes, nil, bytes)
			re.Value = v
			rvn = reflect.ValueOf(&re).Elem()
		} else {
			rvnA := reflect.New(bfn.rt)
			if bytes != nil {
				bfn.ext.ReadExt(rv2i(rvnA), bytes)
			} else {
				bfn.ext.UpdateExt(rv2i(rvnA), v)
			}
			rvn = rvnA.Elem()
		}
	case valueTypeNil:
		// no-op
	case valueTypeInt:
		rvn = n.rr[decNakedIntIdx] // d.np.get(&n.i)
	case valueTypeUint:
		rvn = n.rr[decNakedUintIdx] // d.np.get(&n.u)
	case valueTypeFloat:
		rvn = n.rr[decNakedFloatIdx] // d.np.get(&n.f)
	case valueTypeBool:
		rvn = n.rr[decNakedBoolIdx] // d.np.get(&n.b)
	case valueTypeString, valueTypeSymbol:
		rvn = n.rr[decNakedStringIdx] // d.np.get(&n.s)
	case valueTypeBytes:
		rvn = n.rr[decNakedBytesIdx] // d.np.get(&n.l)
	case valueTypeTimestamp:
		rvn = n.rr[decNakedTimeIdx] // d.np.get(&n.t)
	default:
		panic(fmt.Errorf("kInterfaceNaked: unexpected valueType: %d", n.v))
	}
	return
}

func (d *Decoder) kInterface(f *codecFnInfo, rv reflect.Value) {
	// Note:
	// A consequence of how kInterface works, is that
	// if an interface already contains something, we try
	// to decode into what was there before.
	// We do not replace with a generic value (as got from decodeNaked).

	// every interface passed here MUST be settable.
	var rvn reflect.Value
	if rv.IsNil() {
		if rvn = d.kInterfaceNaked(f); rvn.IsValid() {
			rv.Set(rvn)
		}
		return
	}
	if d.h.InterfaceReset {
		if rvn = d.kInterfaceNaked(f); rvn.IsValid() {
			rv.Set(rvn)
		} else {
			// reset to zero value based on current type in there.
			rv.Set(reflect.Zero(rv.Elem().Type()))
		}
		return
	}

	// now we have a non-nil interface value, meaning it contains a type
	rvn = rv.Elem()
	if d.d.TryDecodeAsNil() {
		rv.Set(reflect.Zero(rvn.Type()))
		return
	}

	// Note: interface{} is settable, but underlying type may not be.
	// Consequently, we MAY have to create a decodable value out of the underlying value,
	// decode into it, and reset the interface itself.
	// fmt.Printf(">>>> kInterface: rvn type: %v, rv type: %v\n", rvn.Type(), rv.Type())

	rvn2, canDecode := isDecodeable(rvn)
	if canDecode {
		d.decodeValue(rvn2, nil, true, true)
		return
	}

	rvn2 = reflect.New(rvn.Type()).Elem()
	rvn2.Set(rvn)
	d.decodeValue(rvn2, nil, true, true)
	rv.Set(rvn2)
}

func (d *Decoder) kStruct(f *codecFnInfo, rv reflect.Value) {
	// checking if recognized within kstruct is too expensive.
	// only check where you can determine if valid outside the loop
	// ie on homogenous collections: slices, arrays and maps.
	//
	// if true, we don't create too many decFn's.
	// It's a delicate balance.
	const checkRecognized bool = false // false: TODO

	fti := f.ti
	dd := d.d
	elemsep := d.hh.hasElemSeparators()
	sfn := structFieldNode{v: rv, update: true}
	ctyp := dd.ContainerType()
	if ctyp == valueTypeMap {
		containerLen := dd.ReadMapStart()
		if containerLen == 0 {
			dd.ReadMapEnd()
			return
		}
		tisfi := fti.sfi
		hasLen := containerLen >= 0

		for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
			// rvkencname := dd.DecodeString()
			if elemsep {
				dd.ReadMapElemKey()
			}
			rvkencnameB := dd.DecodeStringAsBytes()
			rvkencname := stringView(rvkencnameB)
			// rvksi := ti.getForEncName(rvkencname)
			if elemsep {
				dd.ReadMapElemValue()
			}
			if k := fti.indexForEncName(rvkencname); k > -1 {
				si := tisfi[k]
				if dd.TryDecodeAsNil() {
					si.setToZeroValue(rv)
				} else {
					d.decodeValue(sfn.field(si), nil, checkRecognized, true)
				}
			} else {
				d.structFieldNotFound(-1, rvkencname)
			}
			// keepAlive4StringView(rvkencnameB) // maintain ref 4 stringView // not needed, as reference is outside loop
		}
		dd.ReadMapEnd()
	} else if ctyp == valueTypeArray {
		containerLen := dd.ReadArrayStart()
		if containerLen == 0 {
			dd.ReadArrayEnd()
			return
		}
		// Not much gain from doing it two ways for array.
		// Arrays are not used as much for structs.
		hasLen := containerLen >= 0
		for j, si := range fti.sfip {
			if (hasLen && j == containerLen) || (!hasLen && dd.CheckBreak()) {
				break
			}
			if elemsep {
				dd.ReadArrayElem()
			}
			if dd.TryDecodeAsNil() {
				si.setToZeroValue(rv)
			} else {
				d.decodeValue(sfn.field(si), nil, checkRecognized, true)
			}
		}
		if containerLen > len(fti.sfip) {
			// read remaining values and throw away
			for j := len(fti.sfip); j < containerLen; j++ {
				if elemsep {
					dd.ReadArrayElem()
				}
				d.structFieldNotFound(j, "")
			}
		}
		dd.ReadArrayEnd()
	} else {
		d.error(onlyMapOrArrayCanDecodeIntoStructErr)
		return
	}
}

func (d *Decoder) kSlice(f *codecFnInfo, rv reflect.Value) {
	// A slice can be set from a map or array in stream.
	// This way, the order can be kept (as order is lost with map).
	ti := f.ti
	dd := d.d
	rtelem0 := ti.rt.Elem()
	ctyp := dd.ContainerType()
	if ctyp == valueTypeBytes || ctyp == valueTypeString {
		// you can only decode bytes or string in the stream into a slice or array of bytes
		if !(ti.rtid == uint8SliceTypId || rtelem0.Kind() == reflect.Uint8) {
			d.errorf("bytes or string in the stream must be decoded into a slice or array of bytes, not %v", ti.rt)
		}
		if f.seq == seqTypeChan {
			bs2 := dd.DecodeBytes(nil, true)
			ch := rv2i(rv).(chan<- byte)
			for _, b := range bs2 {
				ch <- b
			}
		} else {
			rvbs := rv.Bytes()
			bs2 := dd.DecodeBytes(rvbs, false)
			if rvbs == nil && bs2 != nil || rvbs != nil && bs2 == nil || len(bs2) != len(rvbs) {
				if rv.CanSet() {
					rv.SetBytes(bs2)
				} else {
					copy(rvbs, bs2)
				}
			}
		}
		return
	}

	// array := f.seq == seqTypeChan

	slh, containerLenS := d.decSliceHelperStart() // only expects valueType(Array|Map)

	// an array can never return a nil slice. so no need to check f.array here.
	if containerLenS == 0 {
		if rv.CanSet() {
			if f.seq == seqTypeSlice {
				if rv.IsNil() {
					rv.Set(reflect.MakeSlice(ti.rt, 0, 0))
				} else {
					rv.SetLen(0)
				}
			} else if f.seq == seqTypeChan {
				if rv.IsNil() {
					rv.Set(reflect.MakeChan(ti.rt, 0))
				}
			}
		}
		slh.End()
		return
	}

	rtelem0Size := int(rtelem0.Size())
	rtElem0Kind := rtelem0.Kind()
	rtElem0Id := rt2id(rtelem0)
	rtelem0Mut := !isImmutableKind(rtElem0Kind)
	rtelem := rtelem0
	rtelemkind := rtelem.Kind()
	for rtelemkind == reflect.Ptr {
		rtelem = rtelem.Elem()
		rtelemkind = rtelem.Kind()
	}

	var fn *codecFn

	var rv0, rv9 reflect.Value
	rv0 = rv
	rvChanged := false

	rvlen := rv.Len()
	rvcap := rv.Cap()
	hasLen := containerLenS > 0
	if hasLen && f.seq == seqTypeSlice {
		if containerLenS > rvcap {
			oldRvlenGtZero := rvlen > 0
			rvlen = decInferLen(containerLenS, d.h.MaxInitLen, int(rtelem0.Size()))
			if rvlen <= rvcap {
				if rv.CanSet() {
					rv.SetLen(rvlen)
				} else {
					rv = rv.Slice(0, rvlen)
					rvChanged = true
				}
			} else {
				rv = reflect.MakeSlice(ti.rt, rvlen, rvlen)
				rvcap = rvlen
				rvChanged = true
			}
			if rvChanged && oldRvlenGtZero && !isImmutableKind(rtelem0.Kind()) {
				reflect.Copy(rv, rv0) // only copy up to length NOT cap i.e. rv0.Slice(0, rvcap)
			}
		} else if containerLenS != rvlen {
			rvlen = containerLenS
			if rv.CanSet() {
				rv.SetLen(rvlen)
			} else {
				rv = rv.Slice(0, rvlen)
				rvChanged = true
			}
		}
	}

	var recognizedRtid, recognizedRtidPtr bool
	if useLookupRecognizedTypes {
		recognizedRtid = isRecognizedRtid(rtElem0Id)
		recognizedRtidPtr = isRecognizedRtidPtr(rtElem0Id)
	}

	// consider creating new element once, and just decoding into it.
	var rtelem0Zero reflect.Value
	var rtelem0ZeroValid bool
	var decodeAsNil bool
	var j int
	for ; (hasLen && j < containerLenS) || !(hasLen || dd.CheckBreak()); j++ {
		if j == 0 && (f.seq == seqTypeSlice || f.seq == seqTypeChan) && rv.IsNil() {
			if hasLen {
				rvlen = decInferLen(containerLenS, d.h.MaxInitLen, rtelem0Size)
			} else {
				rvlen = 8
			}
			if f.seq == seqTypeSlice {
				rv = reflect.MakeSlice(ti.rt, rvlen, rvlen)
				rvChanged = true
			} else if f.seq == seqTypeChan {
				rv.Set(reflect.MakeChan(ti.rt, rvlen))
			}
		}
		slh.ElemContainerState(j)
		decodeAsNil = dd.TryDecodeAsNil()
		if f.seq == seqTypeChan {
			if decodeAsNil {
				rv.Send(reflect.Zero(rtelem0))
				continue
			}
			if rtelem0Mut || !rv9.IsValid() { // || (rtElem0Kind == reflect.Ptr && rv9.IsNil()) {
				rv9 = reflect.New(rtelem0).Elem()
			}
			if useLookupRecognizedTypes && (recognizedRtid || recognizedRtidPtr) {
				d.decode(rv2i(rv9.Addr()))
			} else {
				if fn == nil {
					fn = d.cf.get(rtelem, true, true)
				}
				d.decodeValue(rv9, fn, false, true)
			}
			rv.Send(rv9)
		} else {
			// if indefinite, etc, then expand the slice if necessary
			var decodeIntoBlank bool
			if j >= rvlen {
				if f.seq == seqTypeArray {
					d.arrayCannotExpand(rvlen, j+1)
					decodeIntoBlank = true
				} else { // if f.seq == seqTypeSlice
					// rv = reflect.Append(rv, reflect.Zero(rtelem0)) // uses append logic, plus varargs
					var rvcap2 int
					rv9, rvcap2, rvChanged = decExpandSliceRV(rv, ti.rt, rtelem0Size, 1, rvlen, rvcap)
					rvlen++
					if rvChanged {
						rv = rv9
						rvcap = rvcap2
					}
				}
			}
			if decodeIntoBlank {
				if !decodeAsNil {
					d.swallow()
				}
			} else {
				rv9 = rv.Index(j)
				if d.h.SliceElementReset || decodeAsNil {
					if !rtelem0ZeroValid {
						rtelem0ZeroValid = true
						rtelem0Zero = reflect.Zero(rtelem0)
					}
					rv9.Set(rtelem0Zero)
				}
				if decodeAsNil {
					continue
				}

				if useLookupRecognizedTypes && recognizedRtid {
					d.decode(rv2i(rv9.Addr()))
				} else if useLookupRecognizedTypes && recognizedRtidPtr { // && !rv9.IsNil() {
					if rv9.IsNil() {
						rv9.Set(reflect.New(rtelem))
					}
					d.decode(rv2i(rv9))
				} else {
					if fn == nil {
						fn = d.cf.get(rtelem, true, true)
					}
					d.decodeValue(rv9, fn, false, true)
				}
			}
		}
	}
	if f.seq == seqTypeSlice {
		if j < rvlen {
			if rv.CanSet() {
				rv.SetLen(j)
			} else {
				rv = rv.Slice(0, j)
				rvChanged = true
			}
			rvlen = j
		} else if j == 0 && rv.IsNil() {
			rv = reflect.MakeSlice(ti.rt, 0, 0)
			rvChanged = true
		}
	}
	slh.End()

	if rvChanged {
		rv0.Set(rv)
	}
}

// func (d *Decoder) kArray(f *codecFnInfo, rv reflect.Value) {
// 	// d.decodeValueFn(rv.Slice(0, rv.Len()))
// 	f.kSlice(rv.Slice(0, rv.Len()))
// }

func (d *Decoder) kMap(f *codecFnInfo, rv reflect.Value) {
	dd := d.d
	containerLen := dd.ReadMapStart()
	elemsep := d.hh.hasElemSeparators()
	ti := f.ti
	if rv.IsNil() {
		rv.Set(makeMapReflect(ti.rt, containerLen))
	}

	if containerLen == 0 {
		dd.ReadMapEnd()
		return
	}

	ktype, vtype := ti.rt.Key(), ti.rt.Elem()
	ktypeId := rt2id(ktype)
	vtypeId := rt2id(vtype)
	vtypeKind := vtype.Kind()
	var recognizedKtyp, recognizedVtyp, recognizedPtrKtyp, recognizedPtrVtyp bool
	if useLookupRecognizedTypes {
		recognizedKtyp = isRecognizedRtid(ktypeId)
		recognizedVtyp = isRecognizedRtid(vtypeId)
		recognizedPtrKtyp = isRecognizedRtidPtr(ktypeId)
		recognizedPtrVtyp = isRecognizedRtidPtr(vtypeId)
	}

	var keyFn, valFn *codecFn
	var ktypeLo, vtypeLo reflect.Type
	for ktypeLo = ktype; ktypeLo.Kind() == reflect.Ptr; ktypeLo = ktypeLo.Elem() {
	}

	for vtypeLo = vtype; vtypeLo.Kind() == reflect.Ptr; vtypeLo = vtypeLo.Elem() {
	}

	var mapGet, mapSet bool
	rvvImmut := isImmutableKind(vtypeKind)
	if !d.h.MapValueReset {
		// if pointer, mapGet = true
		// if interface, mapGet = true if !DecodeNakedAlways (else false)
		// if builtin, mapGet = false
		// else mapGet = true
		if vtypeKind == reflect.Ptr {
			mapGet = true
		} else if vtypeKind == reflect.Interface {
			if !d.h.InterfaceReset {
				mapGet = true
			}
		} else if !rvvImmut {
			mapGet = true
		}
	}

	var rvk, rvkp, rvv, rvz reflect.Value
	rvkMut := !isImmutableKind(ktype.Kind()) // if ktype is immutable, then re-use the same rvk.
	ktypeIsString := ktypeId == stringTypId
	ktypeIsIntf := ktypeId == intfTypId
	hasLen := containerLen > 0
	var kstrbs []byte
	for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
		if elemsep {
			dd.ReadMapElemKey()
		}
		// if a nil key, just ignore the mapped value and continue
		if dd.TryDecodeAsNil() {
			if elemsep {
				dd.ReadMapElemValue()
			}
			d.swallow()
			continue
		}
		if rvkMut || !rvkp.IsValid() {
			rvkp = reflect.New(ktype)
			rvk = rvkp.Elem()
		}
		if ktypeIsString {
			kstrbs = dd.DecodeStringAsBytes()
			rvk.SetString(stringView(kstrbs))
			// NOTE: if doing an insert, you MUST use a real string (not stringview)
		} else if useLookupRecognizedTypes && recognizedKtyp {
			d.decode(rv2i(rvkp))
			// rvk = rvkp.Elem() //TODO: remove, unnecessary
		} else if useLookupRecognizedTypes && recognizedPtrKtyp {
			if rvk.IsNil() {
				rvk = reflect.New(ktypeLo)
			}
			d.decode(rv2i(rvk))
		} else {
			if keyFn == nil {
				keyFn = d.cf.get(ktypeLo, true, true)
			}
			d.decodeValue(rvk, keyFn, false, true)
		}
		// special case if a byte array.
		if ktypeIsIntf {
			rvk = rvk.Elem()
			if rvk.Type() == uint8SliceTyp {
				rvk = reflect.ValueOf(d.string(rvk.Bytes()))
			}
		}

		if elemsep {
			dd.ReadMapElemValue()
		}

		// Brittle, but OK per TryDecodeAsNil() contract.
		// i.e. TryDecodeAsNil never shares slices with other decDriver procedures
		if dd.TryDecodeAsNil() {
			if ktypeIsString {
				rvk.SetString(d.string(kstrbs))
			}
			if d.h.DeleteOnNilMapValue {
				rv.SetMapIndex(rvk, reflect.Value{})
			} else {
				rv.SetMapIndex(rvk, reflect.Zero(vtype))
			}
			continue
		}

		mapSet = true // set to false if u do a get, and its a non-nil pointer
		if mapGet {
			// mapGet true only in case where kind=Ptr|Interface or kind is otherwise mutable.
			rvv = rv.MapIndex(rvk)
			if !rvv.IsValid() {
				rvv = reflect.New(vtype).Elem()
			} else if vtypeKind == reflect.Ptr {
				if rvv.IsNil() {
					rvv = reflect.New(vtype).Elem()
				} else {
					mapSet = false
				}
			} else if vtypeKind == reflect.Interface {
				// not addressable, and thus not settable.
				// e MUST create a settable/addressable variant
				rvv2 := reflect.New(rvv.Type()).Elem()
				if !rvv.IsNil() {
					rvv2.Set(rvv)
				}
				rvv = rvv2
			}
			// else it is ~mutable, and we can just decode into it directly
		} else if rvvImmut {
			if !rvz.IsValid() {
				rvz = reflect.New(vtype).Elem()
			}
			rvv = rvz
		} else {
			rvv = reflect.New(vtype).Elem()
		}

		// We MUST be done with the stringview of the key, before decoding the value
		// so that we don't bastardize the reused byte array.
		if mapSet && ktypeIsString {
			rvk.SetString(d.string(kstrbs))
		}
		if useLookupRecognizedTypes && recognizedVtyp && rvv.CanAddr() {
			d.decode(rv2i(rvv.Addr()))
		} else if useLookupRecognizedTypes && recognizedPtrVtyp {
			if rvv.IsNil() {
				rvv = reflect.New(vtypeLo)
				mapSet = true
			}
			d.decode(rv2i(rvv))
		} else {
			if valFn == nil {
				valFn = d.cf.get(vtypeLo, true, true)
			}
			d.decodeValue(rvv, valFn, false, true)
			// d.decodeValueFn(rvv, valFn)
		}
		if mapSet {
			rv.SetMapIndex(rvk, rvv)
		}
		// if ktypeIsString {
		// 	// keepAlive4StringView(kstrbs) // not needed, as reference is outside loop
		// }
	}

	dd.ReadMapEnd()
}

// decNaked is used to keep track of the primitives decoded.
// Without it, we would have to decode each primitive and wrap it
// in an interface{}, causing an allocation.
// In this model, the primitives are decoded in a "pseudo-atomic" fashion,
// so we can rest assured that no other decoding happens while these
// primitives are being decoded.
//
// maps and arrays are not handled by this mechanism.
// However, RawExt is, and we accommodate for extensions that decode
// RawExt from DecodeNaked, but need to decode the value subsequently.
// kInterfaceNaked and swallow, which call DecodeNaked, handle this caveat.
//
// However, decNaked also keeps some arrays of default maps and slices
// used in DecodeNaked. This way, we can get a pointer to it
// without causing a new heap allocation.
//
// kInterfaceNaked will ensure that there is no allocation for the common
// uses.
type decNaked struct {
	// r RawExt // used for RawExt, uint, []byte.
	u uint64
	i int64
	f float64
	l []byte
	s string
	t time.Time

	b bool

	inited bool

	v valueType

	li, lm, ln, ls int8

	// array/stacks for reducing allocation
	// keep arrays at the bottom? Chance is that they are not used much.
	ia [arrayCacheLen]interface{}
	ma [arrayCacheLen]map[interface{}]interface{}
	na [arrayCacheLen]map[string]interface{}
	sa [arrayCacheLen][]interface{}
	// ra [2]RawExt

	rr [5 * arrayCacheLen]reflect.Value
}

const (
	decNakedUintIdx = iota
	decNakedIntIdx
	decNakedFloatIdx
	decNakedBytesIdx
	decNakedStringIdx
	decNakedTimeIdx
	decNakedBoolIdx
)
const (
	_ = iota // maps to the scalars above
	decNakedIntfIdx
	decNakedMapIntfIntfIdx
	decNakedMapStrIntfIdx
	decNakedSliceIntfIdx
)

func (n *decNaked) init() {
	if n.inited {
		return
	}
	// n.ms = n.ma[:0]
	// n.is = n.ia[:0]
	// n.ns = n.na[:0]
	// n.ss = n.sa[:0]

	n.rr[decNakedUintIdx] = reflect.ValueOf(&n.u).Elem()
	n.rr[decNakedIntIdx] = reflect.ValueOf(&n.i).Elem()
	n.rr[decNakedFloatIdx] = reflect.ValueOf(&n.f).Elem()
	n.rr[decNakedBytesIdx] = reflect.ValueOf(&n.l).Elem()
	n.rr[decNakedStringIdx] = reflect.ValueOf(&n.s).Elem()
	n.rr[decNakedTimeIdx] = reflect.ValueOf(&n.t).Elem()
	n.rr[decNakedBoolIdx] = reflect.ValueOf(&n.b).Elem()

	for i := range [arrayCacheLen]struct{}{} {
		n.rr[decNakedIntfIdx*arrayCacheLen+i] = reflect.ValueOf(&(n.ia[i])).Elem()
		n.rr[decNakedMapIntfIntfIdx*arrayCacheLen+i] = reflect.ValueOf(&(n.ma[i])).Elem()
		n.rr[decNakedMapStrIntfIdx*arrayCacheLen+i] = reflect.ValueOf(&(n.na[i])).Elem()
		n.rr[decNakedSliceIntfIdx*arrayCacheLen+i] = reflect.ValueOf(&(n.sa[i])).Elem()
	}
	n.inited = true
	// n.rr[] = reflect.ValueOf(&n.)
}

func (n *decNaked) reset() {
	if n == nil {
		return
	}
	n.li, n.lm, n.ln, n.ls = 0, 0, 0, 0
}

type rtid2rv struct {
	rtid uintptr
	rv   reflect.Value
}

// A Decoder reads and decodes an object from an input stream in the codec format.
type Decoder struct {
	// hopefully, reduce derefencing cost by laying the decReader inside the Decoder.
	// Try to put things that go together to fit within a cache line (8 words).

	d decDriver
	// NOTE: Decoder shouldn't call it's read methods,
	// as the handler MAY need to do some coordination.
	r  decReader
	hh Handle
	h  *BasicHandle

	mtr, mtrp, str, strp bool //

	be    bool // is binary encoding
	bytes bool // is bytes reader
	js    bool // is json handle

	// ---- cpu cache line boundary?

	rb bytesDecReader
	ri ioDecReader
	bi bufioDecReader

	// cr containerStateRecv

	n   *decNaked
	nsp *sync.Pool

	// ---- cpu cache line boundary?

	is map[string]string // used for interning strings

	// cache the mapTypeId and sliceTypeId for faster comparisons
	mtid uintptr
	stid uintptr

	b [scratchByteArrayLen]byte
	// _  uintptr // for alignment purposes, so next one starts from a cache line

	err error
	// ---- cpu cache line boundary?

	cf codecFner
	// _ [64]byte // force alignment???
}

// NewDecoder returns a Decoder for decoding a stream of bytes from an io.Reader.
//
// For efficiency, Users are encouraged to pass in a memory buffered reader
// (eg bufio.Reader, bytes.Buffer).
func NewDecoder(r io.Reader, h Handle) *Decoder {
	d := newDecoder(h)
	d.Reset(r)
	return d
}

// NewDecoderBytes returns a Decoder which efficiently decodes directly
// from a byte slice with zero copying.
func NewDecoderBytes(in []byte, h Handle) *Decoder {
	d := newDecoder(h)
	d.ResetBytes(in)
	return d
}

var defaultDecNaked decNaked

func newDecoder(h Handle) *Decoder {
	d := &Decoder{hh: h, h: h.getBasicHandle(), be: h.isBinary()}

	// NOTE: do not initialize d.n here. It is lazily initialized in d.naked()

	_, d.js = h.(*JsonHandle)
	if d.h.InternString {
		d.is = make(map[string]string, 32)
	}
	d.d = h.newDecDriver(d)
	// d.cr, _ = d.d.(containerStateRecv)
	return d
}

// naked must be called before each call to .DecodeNaked,
// as they will use it.
func (d *Decoder) naked() *decNaked {
	if d.n == nil {
		// consider one of:
		//   - get from sync.Pool  (if GC is frequent, there's no value here)
		//   - new alloc           (safest. only init'ed if it a naked decode will be done)
		//   - field in Decoder    (makes the Decoder struct very big)
		// To support using a decoder where a DecodeNaked is not needed,
		// we prefer #1 or #2.
		// d.n = new(decNaked) // &d.nv // new(decNaked) // grab from a sync.Pool
		// d.n.init()
		var v interface{}
		d.nsp, v = pool.decNaked()
		d.n = v.(*decNaked)
	}
	return d.n
}

func (d *Decoder) resetCommon() {
	d.n.reset()
	d.d.reset()
	d.cf.reset(d.hh)
	d.err = nil
	// reset all things which were cached from the Handle,
	// but could be changed.
	d.mtid, d.stid = 0, 0
	d.mtr, d.mtrp, d.str, d.strp = false, false, false, false
	if d.h.MapType != nil {
		d.mtid = rt2id(d.h.MapType)
		if useLookupRecognizedTypes {
			d.mtr = isRecognizedRtid(d.mtid)
			d.mtrp = isRecognizedRtidPtr(d.mtid)
		}
	}
	if d.h.SliceType != nil {
		d.stid = rt2id(d.h.SliceType)
		if useLookupRecognizedTypes {
			d.str = isRecognizedRtid(d.stid)
			d.strp = isRecognizedRtidPtr(d.stid)
		}
	}
}

func (d *Decoder) Reset(r io.Reader) {
	if d.h.ReaderBufferSize > 0 {
		d.bi.buf = make([]byte, 0, d.h.ReaderBufferSize)
		d.bi.reset(r)
		d.r = &d.bi
	} else {
		d.ri.x = &d.b
		// d.s = d.sa[:0]
		d.ri.reset(r)
		d.r = &d.ri
	}
	d.resetCommon()
}

func (d *Decoder) ResetBytes(in []byte) {
	d.bytes = true
	d.rb.reset(in)
	d.r = &d.rb
	d.resetCommon()
}

// Decode decodes the stream from reader and stores the result in the
// value pointed to by v. v cannot be a nil pointer. v can also be
// a reflect.Value of a pointer.
//
// Note that a pointer to a nil interface is not a nil pointer.
// If you do not know what type of stream it is, pass in a pointer to a nil interface.
// We will decode and store a value in that nil interface.
//
// Sample usages:
//   // Decoding into a non-nil typed value
//   var f float32
//   err = codec.NewDecoder(r, handle).Decode(&f)
//
//   // Decoding into nil interface
//   var v interface{}
//   dec := codec.NewDecoder(r, handle)
//   err = dec.Decode(&v)
//
// When decoding into a nil interface{}, we will decode into an appropriate value based
// on the contents of the stream:
//   - Numbers are decoded as float64, int64 or uint64.
//   - Other values are decoded appropriately depending on the type:
//     bool, string, []byte, time.Time, etc
//   - Extensions are decoded as RawExt (if no ext function registered for the tag)
// Configurations exist on the Handle to override defaults
// (e.g. for MapType, SliceType and how to decode raw bytes).
//
// When decoding into a non-nil interface{} value, the mode of encoding is based on the
// type of the value. When a value is seen:
//   - If an extension is registered for it, call that extension function
//   - If it implements BinaryUnmarshaler, call its UnmarshalBinary(data []byte) error
//   - Else decode it based on its reflect.Kind
//
// There are some special rules when decoding into containers (slice/array/map/struct).
// Decode will typically use the stream contents to UPDATE the container.
//   - A map can be decoded from a stream map, by updating matching keys.
//   - A slice can be decoded from a stream array,
//     by updating the first n elements, where n is length of the stream.
//   - A slice can be decoded from a stream map, by decoding as if
//     it contains a sequence of key-value pairs.
//   - A struct can be decoded from a stream map, by updating matching fields.
//   - A struct can be decoded from a stream array,
//     by updating fields as they occur in the struct (by index).
//
// When decoding a stream map or array with length of 0 into a nil map or slice,
// we reset the destination map or slice to a zero-length value.
//
// However, when decoding a stream nil, we reset the destination container
// to its "zero" value (e.g. nil for slice/map, etc).
//
func (d *Decoder) Decode(v interface{}) (err error) {
	defer panicToErrs2(&d.err, &err)
	d.MustDecode(v)
	return
}

// MustDecode is like Decode, but panics if unable to Decode.
// This provides insight to the code location that triggered the error.
func (d *Decoder) MustDecode(v interface{}) {
	// TODO: Top-level: ensure that v is a pointer and not nil.
	if d.err != nil {
		panic(d.err)
	}
	if d.d.TryDecodeAsNil() {
		d.setZero(v)
	} else {
		d.decode(v)
	}
	if d.nsp != nil {
		if d.n != nil {
			d.nsp.Put(d.n)
			d.n = nil
		}
		d.nsp = nil
	}
	d.n = nil
	// xprintf(">>>>>>>> >>>>>>>> num decFns: %v\n", d.cf.sn)
}

// // this is not a smart swallow, as it allocates objects and does unnecessary work.
// func (d *Decoder) swallowViaHammer() {
// 	var blank interface{}
// 	d.decodeValueNoFn(reflect.ValueOf(&blank).Elem())
// }

func (d *Decoder) swallow() {
	// smarter decode that just swallows the content
	dd := d.d
	if dd.TryDecodeAsNil() {
		return
	}
	elemsep := d.hh.hasElemSeparators()
	switch dd.ContainerType() {
	case valueTypeMap:
		containerLen := dd.ReadMapStart()
		hasLen := containerLen >= 0
		for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
			// if clenGtEqualZero {if j >= containerLen {break} } else if dd.CheckBreak() {break}
			if elemsep {
				dd.ReadMapElemKey()
			}
			d.swallow()
			if elemsep {
				dd.ReadMapElemValue()
			}
			d.swallow()
		}
		dd.ReadMapEnd()
	case valueTypeArray:
		containerLen := dd.ReadArrayStart()
		hasLen := containerLen >= 0
		for j := 0; (hasLen && j < containerLen) || !(hasLen || dd.CheckBreak()); j++ {
			if elemsep {
				dd.ReadArrayElem()
			}
			d.swallow()
		}
		dd.ReadArrayEnd()
	case valueTypeBytes:
		dd.DecodeBytes(d.b[:], true)
	case valueTypeString:
		dd.DecodeStringAsBytes()
	default:
		// these are all primitives, which we can get from decodeNaked
		// if RawExt using Value, complete the processing.
		n := d.naked()
		dd.DecodeNaked()
		if n.v == valueTypeExt && n.l == nil {
			if n.li < arrayCacheLen {
				n.ia[n.li] = nil
				n.li++
				d.decode(&n.ia[n.li-1])
				n.ia[n.li-1] = nil
				n.li--
			} else {
				var v2 interface{}
				d.decode(&v2)
			}
		}
	}
}

func (d *Decoder) setZero(iv interface{}) {
	if iv == nil || definitelyNil(iv) {
		return
	}
	switch v := iv.(type) {
	case *string:
		*v = ""
	case *bool:
		*v = false
	case *int:
		*v = 0
	case *int8:
		*v = 0
	case *int16:
		*v = 0
	case *int32:
		*v = 0
	case *int64:
		*v = 0
	case *uint:
		*v = 0
	case *uint8:
		*v = 0
	case *uint16:
		*v = 0
	case *uint32:
		*v = 0
	case *uint64:
		*v = 0
	case *float32:
		*v = 0
	case *float64:
		*v = 0
	case *[]uint8:
		*v = nil
	case *Raw:
		*v = nil
	case reflect.Value:
		v = d.ensureDecodeable(v)
		if v.CanSet() {
			v.Set(reflect.Zero(v.Type()))
		} // TODO: else drain if chan, clear if map, set all to nil if slice???
	default:
		if !fastpathDecodeSetZeroTypeSwitch(iv, d) {
			v := reflect.ValueOf(iv)
			v = d.ensureDecodeable(v)
			if v.CanSet() {
				v.Set(reflect.Zero(v.Type()))
			} // TODO: else drain if chan, clear if map, set all to nil if slice???
		}
	}
}

func (d *Decoder) decode(iv interface{}) {
	// check nil and interfaces explicitly,
	// so that type switches just have a run of constant non-interface types.
	if iv == nil {
		d.error(cannotDecodeIntoNilErr)
		return
	}
	if v, ok := iv.(Selfer); ok {
		v.CodecDecodeSelf(d)
		return
	}

	switch v := iv.(type) {
	// case nil:
	// case Selfer:

	case reflect.Value:
		v = d.ensureDecodeable(v)
		d.decodeValue(v, nil, false, true) // TODO: maybe ask to recognize ...

	case *string:
		*v = d.d.DecodeString()
	case *bool:
		*v = d.d.DecodeBool()
	case *int:
		*v = int(d.d.DecodeInt(intBitsize))
	case *int8:
		*v = int8(d.d.DecodeInt(8))
	case *int16:
		*v = int16(d.d.DecodeInt(16))
	case *int32:
		*v = int32(d.d.DecodeInt(32))
	case *int64:
		*v = d.d.DecodeInt(64)
	case *uint:
		*v = uint(d.d.DecodeUint(uintBitsize))
	case *uint8:
		*v = uint8(d.d.DecodeUint(8))
	case *uint16:
		*v = uint16(d.d.DecodeUint(16))
	case *uint32:
		*v = uint32(d.d.DecodeUint(32))
	case *uint64:
		*v = d.d.DecodeUint(64)
	case *float32:
		*v = float32(d.d.DecodeFloat(true))
	case *float64:
		*v = d.d.DecodeFloat(false)
	case *[]uint8:
		*v = d.d.DecodeBytes(*v, false)

	case *Raw:
		*v = d.rawBytes()

	case *interface{}:
		d.decodeValue(reflect.ValueOf(iv).Elem(), nil, false, true) // TODO: consider recognize here
		// d.decodeValueNotNil(reflect.ValueOf(iv).Elem())

	default:
		if !fastpathDecodeTypeSwitch(iv, d) {
			v := reflect.ValueOf(iv)
			v = d.ensureDecodeable(v)
			d.decodeValue(v, nil, false, false)
			// d.decodeValueFallback(v)
		}
	}
}

func (d *Decoder) decodeValue(rv reflect.Value, fn *codecFn, tryRecognized, chkAll bool) {
	// If stream is not containing a nil value, then we can deref to the base
	// non-pointer value, and decode into that.
	var rvp reflect.Value
	var rvpValid bool
	if rv.Kind() == reflect.Ptr {
		rvpValid = true
		for {
			if rv.IsNil() {
				rv.Set(reflect.New(rv.Type().Elem()))
			}
			rvp = rv
			rv = rv.Elem()
			if rv.Kind() != reflect.Ptr {
				break
			}
		}
	}

	if useLookupRecognizedTypes && tryRecognized && isRecognizedRtid(rv2rtid(rv)) {
		if rvpValid {
			d.decode(rv2i(rvp))
			return
		} else if rv.CanAddr() {
			d.decode(rv2i(rv.Addr()))
			return
		}
	}

	if fn == nil {
		// always pass checkCodecSelfer=true, in case T or ****T is passed, where *T is a Selfer
		fn = d.cf.get(rv.Type(), chkAll, true) // chkAll, chkAll)
	}
	if fn.i.addr {
		if rvpValid {
			fn.fd(d, &fn.i, rvp)
		} else if rv.CanAddr() {
			fn.fd(d, &fn.i, rv.Addr())
		} else {
			fn.fd(d, &fn.i, rv)
		}
	} else {
		fn.fd(d, &fn.i, rv)
	}
	// return rv
}

func (d *Decoder) structFieldNotFound(index int, rvkencname string) {
	// NOTE: rvkencname may be a stringView, so don't pass it to another function.
	if d.h.ErrorIfNoField {
		if index >= 0 {
			d.errorf("no matching struct field found when decoding stream array at index %v", index)
			return
		} else if rvkencname != "" {
			d.errorf("no matching struct field found when decoding stream map with key " + rvkencname)
			return
		}
	}
	d.swallow()
}

func (d *Decoder) arrayCannotExpand(sliceLen, streamLen int) {
	if d.h.ErrorIfNoArrayExpand {
		d.errorf("cannot expand array len during decode from %v to %v", sliceLen, streamLen)
	}
}

func isDecodeable(rv reflect.Value) (rv2 reflect.Value, canDecode bool) {
	switch rv.Kind() {
	case reflect.Array:
		return rv, true
	case reflect.Ptr:
		if !rv.IsNil() {
			return rv.Elem(), true
		}
	case reflect.Slice, reflect.Chan, reflect.Map:
		if !rv.IsNil() {
			return rv, true
		}
	}
	return
}

func (d *Decoder) ensureDecodeable(rv reflect.Value) (rv2 reflect.Value) {
	// decode can take any reflect.Value that is a inherently addressable i.e.
	//   - array
	//   - non-nil chan    (we will SEND to it)
	//   - non-nil slice   (we will set its elements)
	//   - non-nil map     (we will put into it)
	//   - non-nil pointer (we can "update" it)
	rv2, canDecode := isDecodeable(rv)
	if canDecode {
		return
	}
	if !rv.IsValid() {
		d.error(cannotDecodeIntoNilErr)
		return
	}
	if !rv.CanInterface() {
		d.errorf("cannot decode into a value without an interface: %v", rv)
		return
	}
	rvi := rv2i(rv)
	d.errorf("cannot decode into value of kind: %v, type: %T, %v", rv.Kind(), rvi, rvi)
	return
}

// func (d *Decoder) chkPtrValue(rv reflect.Value) {
// 	// We can only decode into a non-nil pointer
// 	if rv.Kind() == reflect.Ptr && !rv.IsNil() {
// 		return
// 	}
// 	d.errNotValidPtrValue(rv)
// }

// func (d *Decoder) errNotValidPtrValue(rv reflect.Value) {
// 	if !rv.IsValid() {
// 		d.error(cannotDecodeIntoNilErr)
// 		return
// 	}
// 	if !rv.CanInterface() {
// 		d.errorf("cannot decode into a value without an interface: %v", rv)
// 		return
// 	}
// 	rvi := rv2i(rv)
// 	d.errorf("cannot decode into non-pointer or nil pointer. Got: %v, %T, %v", rv.Kind(), rvi, rvi)
// }

func (d *Decoder) error(err error) {
	panic(err)
}

func (d *Decoder) errorf(format string, params ...interface{}) {
	params2 := make([]interface{}, len(params)+1)
	params2[0] = d.r.numread()
	copy(params2[1:], params)
	err := fmt.Errorf("[pos %d]: "+format, params2...)
	panic(err)
}

// Possibly get an interned version of a string
//
// This should mostly be used for map keys, where the key type is string.
// This is because keys of a map/struct are typically reused across many objects.
func (d *Decoder) string(v []byte) (s string) {
	if d.is == nil {
		return string(v) // don't return stringView, as we need a real string here.
	}
	s, ok := d.is[string(v)] // no allocation here, per go implementation
	if !ok {
		s = string(v) // new allocation here
		d.is[s] = s
	}
	return s
}

// nextValueBytes returns the next value in the stream as a set of bytes.
func (d *Decoder) nextValueBytes() (bs []byte) {
	d.d.uncacheRead()
	d.r.track()
	d.swallow()
	bs = d.r.stopTrack()
	return
}

func (d *Decoder) rawBytes() []byte {
	// ensure that this is not a view into the bytes
	// i.e. make new copy always.
	bs := d.nextValueBytes()
	bs2 := make([]byte, len(bs))
	copy(bs2, bs)
	return bs2
}

// --------------------------------------------------

// decSliceHelper assists when decoding into a slice, from a map or an array in the stream.
// A slice can be set from a map or array in stream. This supports the MapBySlice interface.
type decSliceHelper struct {
	d *Decoder
	// ct valueType
	array bool
}

func (d *Decoder) decSliceHelperStart() (x decSliceHelper, clen int) {
	dd := d.d
	ctyp := dd.ContainerType()
	if ctyp == valueTypeArray {
		x.array = true
		clen = dd.ReadArrayStart()
	} else if ctyp == valueTypeMap {
		clen = dd.ReadMapStart() * 2
	} else {
		d.errorf("only encoded map or array can be decoded into a slice (%d)", ctyp)
	}
	// x.ct = ctyp
	x.d = d
	return
}

func (x decSliceHelper) End() {
	if x.array {
		x.d.d.ReadArrayEnd()
	} else {
		x.d.d.ReadMapEnd()
	}
}

func (x decSliceHelper) ElemContainerState(index int) {
	if x.array {
		x.d.d.ReadArrayElem()
	} else {
		if index%2 == 0 {
			x.d.d.ReadMapElemKey()
		} else {
			x.d.d.ReadMapElemValue()
		}
	}
}

func decByteSlice(r decReader, clen, maxInitLen int, bs []byte) (bsOut []byte) {
	if clen == 0 {
		return zeroByteSlice
	}
	if len(bs) == clen {
		bsOut = bs
		r.readb(bsOut)
	} else if cap(bs) >= clen {
		bsOut = bs[:clen]
		r.readb(bsOut)
	} else {
		// bsOut = make([]byte, clen)
		len2 := decInferLen(clen, maxInitLen, 1)
		bsOut = make([]byte, len2)
		r.readb(bsOut)
		for len2 < clen {
			len3 := decInferLen(clen-len2, maxInitLen, 1)
			bs3 := bsOut
			bsOut = make([]byte, len2+len3)
			copy(bsOut, bs3)
			r.readb(bsOut[len2:])
			len2 += len3
		}
	}
	return
}

func detachZeroCopyBytes(isBytesReader bool, dest []byte, in []byte) (out []byte) {
	if xlen := len(in); xlen > 0 {
		if isBytesReader || xlen <= scratchByteArrayLen {
			if cap(dest) >= xlen {
				out = dest[:xlen]
			} else {
				out = make([]byte, xlen)
			}
			copy(out, in)
			return
		}
	}
	return in
}

// decInferLen will infer a sensible length, given the following:
//    - clen: length wanted.
//    - maxlen: max length to be returned.
//      if <= 0, it is unset, and we infer it based on the unit size
//    - unit: number of bytes for each element of the collection
func decInferLen(clen, maxlen, unit int) (rvlen int) {
	// handle when maxlen is not set i.e. <= 0
	if clen <= 0 {
		return
	}
	if maxlen <= 0 {
		// no maxlen defined. Use maximum of 256K memory, with a floor of 4K items.
		// maxlen = 256 * 1024 / unit
		// if maxlen < (4 * 1024) {
		// 	maxlen = 4 * 1024
		// }
		if unit < (256 / 4) {
			maxlen = 256 * 1024 / unit
		} else {
			maxlen = 4 * 1024
		}
	}
	if clen > maxlen {
		rvlen = maxlen
	} else {
		rvlen = clen
	}
	return
}

func decExpandSliceRV(s reflect.Value, st reflect.Type, stElemSize, num, slen, scap int) (
	s2 reflect.Value, scap2 int, changed bool) {
	l1 := slen + num // new slice length
	if l1 < slen {
		panic("expandSlice: slice overflow")
	}
	if l1 <= scap {
		if s.CanSet() {
			s.SetLen(l1)
		} else {
			s2 = s.Slice(0, l1)
			scap2 = scap
			changed = true
		}
		return
	}
	scap2 = growCap(scap, stElemSize, num)
	s2 = reflect.MakeSlice(st, l1, scap2)
	changed = true
	reflect.Copy(s2, s)
	return
}

func decReadFull(r io.Reader, bs []byte) (n int, err error) {
	var nn int
	for n < len(bs) && err == nil {
		nn, err = r.Read(bs[n:])
		if nn > 0 {
			if err == io.EOF {
				// leave EOF for next time
				err = nil
			}
			n += nn
		}
	}

	// do not do this - it serves no purpose
	// if n != len(bs) && err == io.EOF { err = io.ErrUnexpectedEOF }
	return
}
