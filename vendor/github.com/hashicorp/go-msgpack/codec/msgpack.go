// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

/*
MSGPACK

Msgpack-c implementation powers the c, c++, python, ruby, etc libraries.
We need to maintain compatibility with it and how it encodes integer values
without caring about the type.

For compatibility with behaviour of msgpack-c reference implementation:
  - Go intX (>0) and uintX
       IS ENCODED AS
    msgpack +ve fixnum, unsigned
  - Go intX (<0)
       IS ENCODED AS
    msgpack -ve fixnum, signed

*/
package codec

import (
	"fmt"
	"io"
	"math"
	"net/rpc"
)

const (
	mpPosFixNumMin byte = 0x00
	mpPosFixNumMax      = 0x7f
	mpFixMapMin         = 0x80
	mpFixMapMax         = 0x8f
	mpFixArrayMin       = 0x90
	mpFixArrayMax       = 0x9f
	mpFixStrMin         = 0xa0
	mpFixStrMax         = 0xbf
	mpNil               = 0xc0
	_                   = 0xc1
	mpFalse             = 0xc2
	mpTrue              = 0xc3
	mpFloat             = 0xca
	mpDouble            = 0xcb
	mpUint8             = 0xcc
	mpUint16            = 0xcd
	mpUint32            = 0xce
	mpUint64            = 0xcf
	mpInt8              = 0xd0
	mpInt16             = 0xd1
	mpInt32             = 0xd2
	mpInt64             = 0xd3

	// extensions below
	mpBin8     = 0xc4
	mpBin16    = 0xc5
	mpBin32    = 0xc6
	mpExt8     = 0xc7
	mpExt16    = 0xc8
	mpExt32    = 0xc9
	mpFixExt1  = 0xd4
	mpFixExt2  = 0xd5
	mpFixExt4  = 0xd6
	mpFixExt8  = 0xd7
	mpFixExt16 = 0xd8

	mpStr8  = 0xd9 // new
	mpStr16 = 0xda
	mpStr32 = 0xdb

	mpArray16 = 0xdc
	mpArray32 = 0xdd

	mpMap16 = 0xde
	mpMap32 = 0xdf

	mpNegFixNumMin = 0xe0
	mpNegFixNumMax = 0xff
)

// MsgpackSpecRpcMultiArgs is a special type which signifies to the MsgpackSpecRpcCodec
// that the backend RPC service takes multiple arguments, which have been arranged
// in sequence in the slice.
//
// The Codec then passes it AS-IS to the rpc service (without wrapping it in an
// array of 1 element).
type MsgpackSpecRpcMultiArgs []interface{}

// A MsgpackContainer type specifies the different types of msgpackContainers.
type msgpackContainerType struct {
	fixCutoff                   int
	bFixMin, b8, b16, b32       byte
	hasFixMin, has8, has8Always bool
}

var (
	msgpackContainerStr  = msgpackContainerType{32, mpFixStrMin, mpStr8, mpStr16, mpStr32, true, true, false}
	msgpackContainerBin  = msgpackContainerType{0, 0, mpBin8, mpBin16, mpBin32, false, true, true}
	msgpackContainerList = msgpackContainerType{16, mpFixArrayMin, 0, mpArray16, mpArray32, true, false, false}
	msgpackContainerMap  = msgpackContainerType{16, mpFixMapMin, 0, mpMap16, mpMap32, true, false, false}
)

//---------------------------------------------

type msgpackEncDriver struct {
	w encWriter
	h *MsgpackHandle
}

func (e *msgpackEncDriver) isBuiltinType(rt uintptr) bool {
	//no builtin types. All encodings are based on kinds. Types supported as extensions.
	return false
}

func (e *msgpackEncDriver) encodeBuiltin(rt uintptr, v interface{}) {}

func (e *msgpackEncDriver) encodeNil() {
	e.w.writen1(mpNil)
}

func (e *msgpackEncDriver) encodeInt(i int64) {

	switch {
	case i >= 0:
		e.encodeUint(uint64(i))
	case i >= -32:
		e.w.writen1(byte(i))
	case i >= math.MinInt8:
		e.w.writen2(mpInt8, byte(i))
	case i >= math.MinInt16:
		e.w.writen1(mpInt16)
		e.w.writeUint16(uint16(i))
	case i >= math.MinInt32:
		e.w.writen1(mpInt32)
		e.w.writeUint32(uint32(i))
	default:
		e.w.writen1(mpInt64)
		e.w.writeUint64(uint64(i))
	}
}

func (e *msgpackEncDriver) encodeUint(i uint64) {
	switch {
	case i <= math.MaxInt8:
		e.w.writen1(byte(i))
	case i <= math.MaxUint8:
		e.w.writen2(mpUint8, byte(i))
	case i <= math.MaxUint16:
		e.w.writen1(mpUint16)
		e.w.writeUint16(uint16(i))
	case i <= math.MaxUint32:
		e.w.writen1(mpUint32)
		e.w.writeUint32(uint32(i))
	default:
		e.w.writen1(mpUint64)
		e.w.writeUint64(uint64(i))
	}
}

func (e *msgpackEncDriver) encodeBool(b bool) {
	if b {
		e.w.writen1(mpTrue)
	} else {
		e.w.writen1(mpFalse)
	}
}

func (e *msgpackEncDriver) encodeFloat32(f float32) {
	e.w.writen1(mpFloat)
	e.w.writeUint32(math.Float32bits(f))
}

func (e *msgpackEncDriver) encodeFloat64(f float64) {
	e.w.writen1(mpDouble)
	e.w.writeUint64(math.Float64bits(f))
}

func (e *msgpackEncDriver) encodeExtPreamble(xtag byte, l int) {
	switch {
	case l == 1:
		e.w.writen2(mpFixExt1, xtag)
	case l == 2:
		e.w.writen2(mpFixExt2, xtag)
	case l == 4:
		e.w.writen2(mpFixExt4, xtag)
	case l == 8:
		e.w.writen2(mpFixExt8, xtag)
	case l == 16:
		e.w.writen2(mpFixExt16, xtag)
	case l < 256:
		e.w.writen2(mpExt8, byte(l))
		e.w.writen1(xtag)
	case l < 65536:
		e.w.writen1(mpExt16)
		e.w.writeUint16(uint16(l))
		e.w.writen1(xtag)
	default:
		e.w.writen1(mpExt32)
		e.w.writeUint32(uint32(l))
		e.w.writen1(xtag)
	}
}

func (e *msgpackEncDriver) encodeArrayPreamble(length int) {
	e.writeContainerLen(msgpackContainerList, length)
}

func (e *msgpackEncDriver) encodeMapPreamble(length int) {
	e.writeContainerLen(msgpackContainerMap, length)
}

func (e *msgpackEncDriver) encodeString(c charEncoding, s string) {
	if c == c_RAW && e.h.WriteExt {
		e.writeContainerLen(msgpackContainerBin, len(s))
	} else {
		e.writeContainerLen(msgpackContainerStr, len(s))
	}
	if len(s) > 0 {
		e.w.writestr(s)
	}
}

func (e *msgpackEncDriver) encodeSymbol(v string) {
	e.encodeString(c_UTF8, v)
}

func (e *msgpackEncDriver) encodeStringBytes(c charEncoding, bs []byte) {
	if c == c_RAW && e.h.WriteExt {
		e.writeContainerLen(msgpackContainerBin, len(bs))
	} else {
		e.writeContainerLen(msgpackContainerStr, len(bs))
	}
	if len(bs) > 0 {
		e.w.writeb(bs)
	}
}

func (e *msgpackEncDriver) writeContainerLen(ct msgpackContainerType, l int) {
	switch {
	case ct.hasFixMin && l < ct.fixCutoff:
		e.w.writen1(ct.bFixMin | byte(l))
	case ct.has8 && l < 256 && (ct.has8Always || e.h.WriteExt):
		e.w.writen2(ct.b8, uint8(l))
	case l < 65536:
		e.w.writen1(ct.b16)
		e.w.writeUint16(uint16(l))
	default:
		e.w.writen1(ct.b32)
		e.w.writeUint32(uint32(l))
	}
}

//---------------------------------------------

type msgpackDecDriver struct {
	r      decReader
	h      *MsgpackHandle
	bd     byte
	bdRead bool
	bdType valueType
}

func (d *msgpackDecDriver) isBuiltinType(rt uintptr) bool {
	//no builtin types. All encodings are based on kinds. Types supported as extensions.
	return false
}

func (d *msgpackDecDriver) decodeBuiltin(rt uintptr, v interface{}) {}

// Note: This returns either a primitive (int, bool, etc) for non-containers,
// or a containerType, or a specific type denoting nil or extension.
// It is called when a nil interface{} is passed, leaving it up to the DecDriver
// to introspect the stream and decide how best to decode.
// It deciphers the value by looking at the stream first.
func (d *msgpackDecDriver) decodeNaked() (v interface{}, vt valueType, decodeFurther bool) {
	d.initReadNext()
	bd := d.bd

	switch bd {
	case mpNil:
		vt = valueTypeNil
		d.bdRead = false
	case mpFalse:
		vt = valueTypeBool
		v = false
	case mpTrue:
		vt = valueTypeBool
		v = true

	case mpFloat:
		vt = valueTypeFloat
		v = float64(math.Float32frombits(d.r.readUint32()))
	case mpDouble:
		vt = valueTypeFloat
		v = math.Float64frombits(d.r.readUint64())

	case mpUint8:
		vt = valueTypeUint
		v = uint64(d.r.readn1())
	case mpUint16:
		vt = valueTypeUint
		v = uint64(d.r.readUint16())
	case mpUint32:
		vt = valueTypeUint
		v = uint64(d.r.readUint32())
	case mpUint64:
		vt = valueTypeUint
		v = uint64(d.r.readUint64())

	case mpInt8:
		vt = valueTypeInt
		v = int64(int8(d.r.readn1()))
	case mpInt16:
		vt = valueTypeInt
		v = int64(int16(d.r.readUint16()))
	case mpInt32:
		vt = valueTypeInt
		v = int64(int32(d.r.readUint32()))
	case mpInt64:
		vt = valueTypeInt
		v = int64(int64(d.r.readUint64()))

	default:
		switch {
		case bd >= mpPosFixNumMin && bd <= mpPosFixNumMax:
			// positive fixnum (always signed)
			vt = valueTypeInt
			v = int64(int8(bd))
		case bd >= mpNegFixNumMin && bd <= mpNegFixNumMax:
			// negative fixnum
			vt = valueTypeInt
			v = int64(int8(bd))
		case bd == mpStr8, bd == mpStr16, bd == mpStr32, bd >= mpFixStrMin && bd <= mpFixStrMax:
			if d.h.RawToString {
				var rvm string
				vt = valueTypeString
				v = &rvm
			} else {
				var rvm = []byte{}
				vt = valueTypeBytes
				v = &rvm
			}
			decodeFurther = true
		case bd == mpBin8, bd == mpBin16, bd == mpBin32:
			var rvm = []byte{}
			vt = valueTypeBytes
			v = &rvm
			decodeFurther = true
		case bd == mpArray16, bd == mpArray32, bd >= mpFixArrayMin && bd <= mpFixArrayMax:
			vt = valueTypeArray
			decodeFurther = true
		case bd == mpMap16, bd == mpMap32, bd >= mpFixMapMin && bd <= mpFixMapMax:
			vt = valueTypeMap
			decodeFurther = true
		case bd >= mpFixExt1 && bd <= mpFixExt16, bd >= mpExt8 && bd <= mpExt32:
			clen := d.readExtLen()
			var re RawExt
			re.Tag = d.r.readn1()
			re.Data = d.r.readn(clen)
			v = &re
			vt = valueTypeExt
		default:
			decErr("Nil-Deciphered DecodeValue: %s: hex: %x, dec: %d", msgBadDesc, bd, bd)
		}
	}
	if !decodeFurther {
		d.bdRead = false
	}
	return
}

// int can be decoded from msgpack type: intXXX or uintXXX
func (d *msgpackDecDriver) decodeInt(bitsize uint8) (i int64) {
	switch d.bd {
	case mpUint8:
		i = int64(uint64(d.r.readn1()))
	case mpUint16:
		i = int64(uint64(d.r.readUint16()))
	case mpUint32:
		i = int64(uint64(d.r.readUint32()))
	case mpUint64:
		i = int64(d.r.readUint64())
	case mpInt8:
		i = int64(int8(d.r.readn1()))
	case mpInt16:
		i = int64(int16(d.r.readUint16()))
	case mpInt32:
		i = int64(int32(d.r.readUint32()))
	case mpInt64:
		i = int64(d.r.readUint64())
	default:
		switch {
		case d.bd >= mpPosFixNumMin && d.bd <= mpPosFixNumMax:
			i = int64(int8(d.bd))
		case d.bd >= mpNegFixNumMin && d.bd <= mpNegFixNumMax:
			i = int64(int8(d.bd))
		default:
			decErr("Unhandled single-byte unsigned integer value: %s: %x", msgBadDesc, d.bd)
		}
	}
	// check overflow (logic adapted from std pkg reflect/value.go OverflowUint()
	if bitsize > 0 {
		if trunc := (i << (64 - bitsize)) >> (64 - bitsize); i != trunc {
			decErr("Overflow int value: %v", i)
		}
	}
	d.bdRead = false
	return
}

// uint can be decoded from msgpack type: intXXX or uintXXX
func (d *msgpackDecDriver) decodeUint(bitsize uint8) (ui uint64) {
	switch d.bd {
	case mpUint8:
		ui = uint64(d.r.readn1())
	case mpUint16:
		ui = uint64(d.r.readUint16())
	case mpUint32:
		ui = uint64(d.r.readUint32())
	case mpUint64:
		ui = d.r.readUint64()
	case mpInt8:
		if i := int64(int8(d.r.readn1())); i >= 0 {
			ui = uint64(i)
		} else {
			decErr("Assigning negative signed value: %v, to unsigned type", i)
		}
	case mpInt16:
		if i := int64(int16(d.r.readUint16())); i >= 0 {
			ui = uint64(i)
		} else {
			decErr("Assigning negative signed value: %v, to unsigned type", i)
		}
	case mpInt32:
		if i := int64(int32(d.r.readUint32())); i >= 0 {
			ui = uint64(i)
		} else {
			decErr("Assigning negative signed value: %v, to unsigned type", i)
		}
	case mpInt64:
		if i := int64(d.r.readUint64()); i >= 0 {
			ui = uint64(i)
		} else {
			decErr("Assigning negative signed value: %v, to unsigned type", i)
		}
	default:
		switch {
		case d.bd >= mpPosFixNumMin && d.bd <= mpPosFixNumMax:
			ui = uint64(d.bd)
		case d.bd >= mpNegFixNumMin && d.bd <= mpNegFixNumMax:
			decErr("Assigning negative signed value: %v, to unsigned type", int(d.bd))
		default:
			decErr("Unhandled single-byte unsigned integer value: %s: %x", msgBadDesc, d.bd)
		}
	}
	// check overflow (logic adapted from std pkg reflect/value.go OverflowUint()
	if bitsize > 0 {
		if trunc := (ui << (64 - bitsize)) >> (64 - bitsize); ui != trunc {
			decErr("Overflow uint value: %v", ui)
		}
	}
	d.bdRead = false
	return
}

// float can either be decoded from msgpack type: float, double or intX
func (d *msgpackDecDriver) decodeFloat(chkOverflow32 bool) (f float64) {
	switch d.bd {
	case mpFloat:
		f = float64(math.Float32frombits(d.r.readUint32()))
	case mpDouble:
		f = math.Float64frombits(d.r.readUint64())
	default:
		f = float64(d.decodeInt(0))
	}
	checkOverflowFloat32(f, chkOverflow32)
	d.bdRead = false
	return
}

// bool can be decoded from bool, fixnum 0 or 1.
func (d *msgpackDecDriver) decodeBool() (b bool) {
	switch d.bd {
	case mpFalse, 0:
		// b = false
	case mpTrue, 1:
		b = true
	default:
		decErr("Invalid single-byte value for bool: %s: %x", msgBadDesc, d.bd)
	}
	d.bdRead = false
	return
}

func (d *msgpackDecDriver) decodeString() (s string) {
	clen := d.readContainerLen(msgpackContainerStr)
	if clen > 0 {
		s = string(d.r.readn(clen))
	}
	d.bdRead = false
	return
}

// Callers must check if changed=true (to decide whether to replace the one they have)
func (d *msgpackDecDriver) decodeBytes(bs []byte) (bsOut []byte, changed bool) {
	// bytes can be decoded from msgpackContainerStr or msgpackContainerBin
	var clen int
	switch d.bd {
	case mpBin8, mpBin16, mpBin32:
		clen = d.readContainerLen(msgpackContainerBin)
	default:
		clen = d.readContainerLen(msgpackContainerStr)
	}
	// if clen < 0 {
	// 	changed = true
	// 	panic("length cannot be zero. this cannot be nil.")
	// }
	if clen > 0 {
		// if no contents in stream, don't update the passed byteslice
		if len(bs) != clen {
			// Return changed=true if length of passed slice diff from length of bytes in stream
			if len(bs) > clen {
				bs = bs[:clen]
			} else {
				bs = make([]byte, clen)
			}
			bsOut = bs
			changed = true
		}
		d.r.readb(bs)
	}
	d.bdRead = false
	return
}

// Every top-level decode funcs (i.e. decodeValue, decode) must call this first.
func (d *msgpackDecDriver) initReadNext() {
	if d.bdRead {
		return
	}
	d.bd = d.r.readn1()
	d.bdRead = true
	d.bdType = valueTypeUnset
}

func (d *msgpackDecDriver) currentEncodedType() valueType {
	if d.bdType == valueTypeUnset {
		bd := d.bd
		switch bd {
		case mpNil:
			d.bdType = valueTypeNil
		case mpFalse, mpTrue:
			d.bdType = valueTypeBool
		case mpFloat, mpDouble:
			d.bdType = valueTypeFloat
		case mpUint8, mpUint16, mpUint32, mpUint64:
			d.bdType = valueTypeUint
		case mpInt8, mpInt16, mpInt32, mpInt64:
			d.bdType = valueTypeInt
		default:
			switch {
			case bd >= mpPosFixNumMin && bd <= mpPosFixNumMax:
				d.bdType = valueTypeInt
			case bd >= mpNegFixNumMin && bd <= mpNegFixNumMax:
				d.bdType = valueTypeInt
			case bd == mpStr8, bd == mpStr16, bd == mpStr32, bd >= mpFixStrMin && bd <= mpFixStrMax:
				if d.h.RawToString {
					d.bdType = valueTypeString
				} else {
					d.bdType = valueTypeBytes
				}
			case bd == mpBin8, bd == mpBin16, bd == mpBin32:
				d.bdType = valueTypeBytes
			case bd == mpArray16, bd == mpArray32, bd >= mpFixArrayMin && bd <= mpFixArrayMax:
				d.bdType = valueTypeArray
			case bd == mpMap16, bd == mpMap32, bd >= mpFixMapMin && bd <= mpFixMapMax:
				d.bdType = valueTypeMap
			case bd >= mpFixExt1 && bd <= mpFixExt16, bd >= mpExt8 && bd <= mpExt32:
				d.bdType = valueTypeExt
			default:
				decErr("currentEncodedType: Undeciphered descriptor: %s: hex: %x, dec: %d", msgBadDesc, bd, bd)
			}
		}
	}
	return d.bdType
}

func (d *msgpackDecDriver) tryDecodeAsNil() bool {
	if d.bd == mpNil {
		d.bdRead = false
		return true
	}
	return false
}

func (d *msgpackDecDriver) readContainerLen(ct msgpackContainerType) (clen int) {
	bd := d.bd
	switch {
	case bd == mpNil:
		clen = -1 // to represent nil
	case bd == ct.b8:
		clen = int(d.r.readn1())
	case bd == ct.b16:
		clen = int(d.r.readUint16())
	case bd == ct.b32:
		clen = int(d.r.readUint32())
	case (ct.bFixMin & bd) == ct.bFixMin:
		clen = int(ct.bFixMin ^ bd)
	default:
		decErr("readContainerLen: %s: hex: %x, dec: %d", msgBadDesc, bd, bd)
	}
	d.bdRead = false
	return
}

func (d *msgpackDecDriver) readMapLen() int {
	return d.readContainerLen(msgpackContainerMap)
}

func (d *msgpackDecDriver) readArrayLen() int {
	return d.readContainerLen(msgpackContainerList)
}

func (d *msgpackDecDriver) readExtLen() (clen int) {
	switch d.bd {
	case mpNil:
		clen = -1 // to represent nil
	case mpFixExt1:
		clen = 1
	case mpFixExt2:
		clen = 2
	case mpFixExt4:
		clen = 4
	case mpFixExt8:
		clen = 8
	case mpFixExt16:
		clen = 16
	case mpExt8:
		clen = int(d.r.readn1())
	case mpExt16:
		clen = int(d.r.readUint16())
	case mpExt32:
		clen = int(d.r.readUint32())
	default:
		decErr("decoding ext bytes: found unexpected byte: %x", d.bd)
	}
	return
}

func (d *msgpackDecDriver) decodeExt(verifyTag bool, tag byte) (xtag byte, xbs []byte) {
	xbd := d.bd
	switch {
	case xbd == mpBin8, xbd == mpBin16, xbd == mpBin32:
		xbs, _ = d.decodeBytes(nil)
	case xbd == mpStr8, xbd == mpStr16, xbd == mpStr32,
		xbd >= mpFixStrMin && xbd <= mpFixStrMax:
		xbs = []byte(d.decodeString())
	default:
		clen := d.readExtLen()
		xtag = d.r.readn1()
		if verifyTag && xtag != tag {
			decErr("Wrong extension tag. Got %b. Expecting: %v", xtag, tag)
		}
		xbs = d.r.readn(clen)
	}
	d.bdRead = false
	return
}

//--------------------------------------------------

//MsgpackHandle is a Handle for the Msgpack Schema-Free Encoding Format.
type MsgpackHandle struct {
	BasicHandle

	// RawToString controls how raw bytes are decoded into a nil interface{}.
	RawToString bool
	// WriteExt flag supports encoding configured extensions with extension tags.
	// It also controls whether other elements of the new spec are encoded (ie Str8).
	//
	// With WriteExt=false, configured extensions are serialized as raw bytes
	// and Str8 is not encoded.
	//
	// A stream can still be decoded into a typed value, provided an appropriate value
	// is provided, but the type cannot be inferred from the stream. If no appropriate
	// type is provided (e.g. decoding into a nil interface{}), you get back
	// a []byte or string based on the setting of RawToString.
	WriteExt bool
}

func (h *MsgpackHandle) newEncDriver(w encWriter) encDriver {
	return &msgpackEncDriver{w: w, h: h}
}

func (h *MsgpackHandle) newDecDriver(r decReader) decDriver {
	return &msgpackDecDriver{r: r, h: h}
}

func (h *MsgpackHandle) writeExt() bool {
	return h.WriteExt
}

func (h *MsgpackHandle) getBasicHandle() *BasicHandle {
	return &h.BasicHandle
}

//--------------------------------------------------

type msgpackSpecRpcCodec struct {
	rpcCodec
}

// /////////////// Spec RPC Codec ///////////////////
func (c *msgpackSpecRpcCodec) WriteRequest(r *rpc.Request, body interface{}) error {
	// WriteRequest can write to both a Go service, and other services that do
	// not abide by the 1 argument rule of a Go service.
	// We discriminate based on if the body is a MsgpackSpecRpcMultiArgs
	var bodyArr []interface{}
	if m, ok := body.(MsgpackSpecRpcMultiArgs); ok {
		bodyArr = ([]interface{})(m)
	} else {
		bodyArr = []interface{}{body}
	}
	r2 := []interface{}{0, uint32(r.Seq), r.ServiceMethod, bodyArr}
	return c.write(r2, nil, false, true)
}

func (c *msgpackSpecRpcCodec) WriteResponse(r *rpc.Response, body interface{}) error {
	var moe interface{}
	if r.Error != "" {
		moe = r.Error
	}
	if moe != nil && body != nil {
		body = nil
	}
	r2 := []interface{}{1, uint32(r.Seq), moe, body}
	return c.write(r2, nil, false, true)
}

func (c *msgpackSpecRpcCodec) ReadResponseHeader(r *rpc.Response) error {
	return c.parseCustomHeader(1, &r.Seq, &r.Error)
}

func (c *msgpackSpecRpcCodec) ReadRequestHeader(r *rpc.Request) error {
	return c.parseCustomHeader(0, &r.Seq, &r.ServiceMethod)
}

func (c *msgpackSpecRpcCodec) ReadRequestBody(body interface{}) error {
	if body == nil { // read and discard
		return c.read(nil)
	}
	bodyArr := []interface{}{body}
	return c.read(&bodyArr)
}

func (c *msgpackSpecRpcCodec) parseCustomHeader(expectTypeByte byte, msgid *uint64, methodOrError *string) (err error) {

	if c.cls {
		return io.EOF
	}

	// We read the response header by hand
	// so that the body can be decoded on its own from the stream at a later time.

	const fia byte = 0x94 //four item array descriptor value
	// Not sure why the panic of EOF is swallowed above.
	// if bs1 := c.dec.r.readn1(); bs1 != fia {
	// 	err = fmt.Errorf("Unexpected value for array descriptor: Expecting %v. Received %v", fia, bs1)
	// 	return
	// }
	var b byte
	b, err = c.br.ReadByte()
	if err != nil {
		return
	}
	if b != fia {
		err = fmt.Errorf("Unexpected value for array descriptor: Expecting %v. Received %v", fia, b)
		return
	}

	if err = c.read(&b); err != nil {
		return
	}
	if b != expectTypeByte {
		err = fmt.Errorf("Unexpected byte descriptor in header. Expecting %v. Received %v", expectTypeByte, b)
		return
	}
	if err = c.read(msgid); err != nil {
		return
	}
	if err = c.read(methodOrError); err != nil {
		return
	}
	return
}

//--------------------------------------------------

// msgpackSpecRpc is the implementation of Rpc that uses custom communication protocol
// as defined in the msgpack spec at https://github.com/msgpack-rpc/msgpack-rpc/blob/master/spec.md
type msgpackSpecRpc struct{}

// MsgpackSpecRpc implements Rpc using the communication protocol defined in
// the msgpack spec at https://github.com/msgpack-rpc/msgpack-rpc/blob/master/spec.md .
// Its methods (ServerCodec and ClientCodec) return values that implement RpcCodecBuffered.
var MsgpackSpecRpc msgpackSpecRpc

func (x msgpackSpecRpc) ServerCodec(conn io.ReadWriteCloser, h Handle) rpc.ServerCodec {
	return &msgpackSpecRpcCodec{newRPCCodec(conn, h)}
}

func (x msgpackSpecRpc) ClientCodec(conn io.ReadWriteCloser, h Handle) rpc.ClientCodec {
	return &msgpackSpecRpcCodec{newRPCCodec(conn, h)}
}

var _ decDriver = (*msgpackDecDriver)(nil)
var _ encDriver = (*msgpackEncDriver)(nil)
