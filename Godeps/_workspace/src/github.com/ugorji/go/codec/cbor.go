// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import "math"

const (
	cborMajorUint byte = iota
	cborMajorNegInt
	cborMajorBytes
	cborMajorText
	cborMajorArray
	cborMajorMap
	cborMajorTag
	cborMajorOther
)

const (
	cborBdFalse byte = 0xf4 + iota
	cborBdTrue
	cborBdNil
	cborBdUndefined
	cborBdExt
	cborBdFloat16
	cborBdFloat32
	cborBdFloat64
)

const (
	cborBdIndefiniteBytes  byte = 0x5f
	cborBdIndefiniteString      = 0x7f
	cborBdIndefiniteArray       = 0x9f
	cborBdIndefiniteMap         = 0xbf
	cborBdBreak                 = 0xff
)

const (
	CborStreamBytes  byte = 0x5f
	CborStreamString      = 0x7f
	CborStreamArray       = 0x9f
	CborStreamMap         = 0xbf
	CborStreamBreak       = 0xff
)

const (
	cborBaseUint   byte = 0x00
	cborBaseNegInt      = 0x20
	cborBaseBytes       = 0x40
	cborBaseString      = 0x60
	cborBaseArray       = 0x80
	cborBaseMap         = 0xa0
	cborBaseTag         = 0xc0
	cborBaseSimple      = 0xe0
)

// -------------------

type cborEncDriver struct {
	e *Encoder
	w encWriter
	h *CborHandle
	noBuiltInTypes
	encNoSeparator
	x [8]byte
}

func (e *cborEncDriver) EncodeNil() {
	e.w.writen1(cborBdNil)
}

func (e *cborEncDriver) EncodeBool(b bool) {
	if b {
		e.w.writen1(cborBdTrue)
	} else {
		e.w.writen1(cborBdFalse)
	}
}

func (e *cborEncDriver) EncodeFloat32(f float32) {
	e.w.writen1(cborBdFloat32)
	bigenHelper{e.x[:4], e.w}.writeUint32(math.Float32bits(f))
}

func (e *cborEncDriver) EncodeFloat64(f float64) {
	e.w.writen1(cborBdFloat64)
	bigenHelper{e.x[:8], e.w}.writeUint64(math.Float64bits(f))
}

func (e *cborEncDriver) encUint(v uint64, bd byte) {
	if v <= 0x17 {
		e.w.writen1(byte(v) + bd)
	} else if v <= math.MaxUint8 {
		e.w.writen2(bd+0x18, uint8(v))
	} else if v <= math.MaxUint16 {
		e.w.writen1(bd + 0x19)
		bigenHelper{e.x[:2], e.w}.writeUint16(uint16(v))
	} else if v <= math.MaxUint32 {
		e.w.writen1(bd + 0x1a)
		bigenHelper{e.x[:4], e.w}.writeUint32(uint32(v))
	} else { // if v <= math.MaxUint64 {
		e.w.writen1(bd + 0x1b)
		bigenHelper{e.x[:8], e.w}.writeUint64(v)
	}
}

func (e *cborEncDriver) EncodeInt(v int64) {
	if v < 0 {
		e.encUint(uint64(-1-v), cborBaseNegInt)
	} else {
		e.encUint(uint64(v), cborBaseUint)
	}
}

func (e *cborEncDriver) EncodeUint(v uint64) {
	e.encUint(v, cborBaseUint)
}

func (e *cborEncDriver) encLen(bd byte, length int) {
	e.encUint(uint64(length), bd)
}

func (e *cborEncDriver) EncodeExt(rv interface{}, xtag uint64, ext Ext, en *Encoder) {
	e.encUint(uint64(xtag), cborBaseTag)
	if v := ext.ConvertExt(rv); v == nil {
		e.EncodeNil()
	} else {
		en.encode(v)
	}
}

func (e *cborEncDriver) EncodeRawExt(re *RawExt, en *Encoder) {
	e.encUint(uint64(re.Tag), cborBaseTag)
	if re.Data != nil {
		en.encode(re.Data)
	} else if re.Value == nil {
		e.EncodeNil()
	} else {
		en.encode(re.Value)
	}
}

func (e *cborEncDriver) EncodeArrayStart(length int) {
	e.encLen(cborBaseArray, length)
}

func (e *cborEncDriver) EncodeMapStart(length int) {
	e.encLen(cborBaseMap, length)
}

func (e *cborEncDriver) EncodeString(c charEncoding, v string) {
	e.encLen(cborBaseString, len(v))
	e.w.writestr(v)
}

func (e *cborEncDriver) EncodeSymbol(v string) {
	e.EncodeString(c_UTF8, v)
}

func (e *cborEncDriver) EncodeStringBytes(c charEncoding, v []byte) {
	e.encLen(cborBaseBytes, len(v))
	e.w.writeb(v)
}

// ----------------------

type cborDecDriver struct {
	d      *Decoder
	h      *CborHandle
	r      decReader
	br     bool // bytes reader
	bdRead bool
	bdType valueType
	bd     byte
	b      [scratchByteArrayLen]byte
	noBuiltInTypes
	decNoSeparator
}

func (d *cborDecDriver) readNextBd() {
	d.bd = d.r.readn1()
	d.bdRead = true
	d.bdType = valueTypeUnset
}

func (d *cborDecDriver) IsContainerType(vt valueType) (bv bool) {
	switch vt {
	case valueTypeNil:
		return d.bd == cborBdNil
	case valueTypeBytes:
		return d.bd == cborBdIndefiniteBytes || (d.bd >= cborBaseBytes && d.bd < cborBaseString)
	case valueTypeString:
		return d.bd == cborBdIndefiniteString || (d.bd >= cborBaseString && d.bd < cborBaseArray)
	case valueTypeArray:
		return d.bd == cborBdIndefiniteArray || (d.bd >= cborBaseArray && d.bd < cborBaseMap)
	case valueTypeMap:
		return d.bd == cborBdIndefiniteMap || (d.bd >= cborBaseMap && d.bd < cborBaseTag)
	}
	d.d.errorf("isContainerType: unsupported parameter: %v", vt)
	return // "unreachable"
}

func (d *cborDecDriver) TryDecodeAsNil() bool {
	if !d.bdRead {
		d.readNextBd()
	}
	// treat Nil and Undefined as nil values
	if d.bd == cborBdNil || d.bd == cborBdUndefined {
		d.bdRead = false
		return true
	}
	return false
}

func (d *cborDecDriver) CheckBreak() bool {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == cborBdBreak {
		d.bdRead = false
		return true
	}
	return false
}

func (d *cborDecDriver) decUint() (ui uint64) {
	v := d.bd & 0x1f
	if v <= 0x17 {
		ui = uint64(v)
	} else {
		if v == 0x18 {
			ui = uint64(d.r.readn1())
		} else if v == 0x19 {
			ui = uint64(bigen.Uint16(d.r.readx(2)))
		} else if v == 0x1a {
			ui = uint64(bigen.Uint32(d.r.readx(4)))
		} else if v == 0x1b {
			ui = uint64(bigen.Uint64(d.r.readx(8)))
		} else {
			d.d.errorf("decUint: Invalid descriptor: %v", d.bd)
			return
		}
	}
	return
}

func (d *cborDecDriver) decCheckInteger() (neg bool) {
	if !d.bdRead {
		d.readNextBd()
	}
	major := d.bd >> 5
	if major == cborMajorUint {
	} else if major == cborMajorNegInt {
		neg = true
	} else {
		d.d.errorf("invalid major: %v (bd: %v)", major, d.bd)
		return
	}
	return
}

func (d *cborDecDriver) DecodeInt(bitsize uint8) (i int64) {
	neg := d.decCheckInteger()
	ui := d.decUint()
	// check if this number can be converted to an int without overflow
	var overflow bool
	if neg {
		if i, overflow = chkOvf.SignedInt(ui + 1); overflow {
			d.d.errorf("cbor: overflow converting %v to signed integer", ui+1)
			return
		}
		i = -i
	} else {
		if i, overflow = chkOvf.SignedInt(ui); overflow {
			d.d.errorf("cbor: overflow converting %v to signed integer", ui)
			return
		}
	}
	if chkOvf.Int(i, bitsize) {
		d.d.errorf("cbor: overflow integer: %v", i)
		return
	}
	d.bdRead = false
	return
}

func (d *cborDecDriver) DecodeUint(bitsize uint8) (ui uint64) {
	if d.decCheckInteger() {
		d.d.errorf("Assigning negative signed value to unsigned type")
		return
	}
	ui = d.decUint()
	if chkOvf.Uint(ui, bitsize) {
		d.d.errorf("cbor: overflow integer: %v", ui)
		return
	}
	d.bdRead = false
	return
}

func (d *cborDecDriver) DecodeFloat(chkOverflow32 bool) (f float64) {
	if !d.bdRead {
		d.readNextBd()
	}
	if bd := d.bd; bd == cborBdFloat16 {
		f = float64(math.Float32frombits(halfFloatToFloatBits(bigen.Uint16(d.r.readx(2)))))
	} else if bd == cborBdFloat32 {
		f = float64(math.Float32frombits(bigen.Uint32(d.r.readx(4))))
	} else if bd == cborBdFloat64 {
		f = math.Float64frombits(bigen.Uint64(d.r.readx(8)))
	} else if bd >= cborBaseUint && bd < cborBaseBytes {
		f = float64(d.DecodeInt(64))
	} else {
		d.d.errorf("Float only valid from float16/32/64: Invalid descriptor: %v", bd)
		return
	}
	if chkOverflow32 && chkOvf.Float32(f) {
		d.d.errorf("cbor: float32 overflow: %v", f)
		return
	}
	d.bdRead = false
	return
}

// bool can be decoded from bool only (single byte).
func (d *cborDecDriver) DecodeBool() (b bool) {
	if !d.bdRead {
		d.readNextBd()
	}
	if bd := d.bd; bd == cborBdTrue {
		b = true
	} else if bd == cborBdFalse {
	} else {
		d.d.errorf("Invalid single-byte value for bool: %s: %x", msgBadDesc, d.bd)
		return
	}
	d.bdRead = false
	return
}

func (d *cborDecDriver) ReadMapStart() (length int) {
	d.bdRead = false
	if d.bd == cborBdIndefiniteMap {
		return -1
	}
	return d.decLen()
}

func (d *cborDecDriver) ReadArrayStart() (length int) {
	d.bdRead = false
	if d.bd == cborBdIndefiniteArray {
		return -1
	}
	return d.decLen()
}

func (d *cborDecDriver) decLen() int {
	return int(d.decUint())
}

func (d *cborDecDriver) decAppendIndefiniteBytes(bs []byte) []byte {
	d.bdRead = false
	for {
		if d.CheckBreak() {
			break
		}
		if major := d.bd >> 5; major != cborMajorBytes && major != cborMajorText {
			d.d.errorf("cbor: expect bytes or string major type in indefinite string/bytes; got: %v, byte: %v", major, d.bd)
			return nil
		}
		n := d.decLen()
		oldLen := len(bs)
		newLen := oldLen + n
		if newLen > cap(bs) {
			bs2 := make([]byte, newLen, 2*cap(bs)+n)
			copy(bs2, bs)
			bs = bs2
		} else {
			bs = bs[:newLen]
		}
		d.r.readb(bs[oldLen:newLen])
		// bs = append(bs, d.r.readn()...)
		d.bdRead = false
	}
	d.bdRead = false
	return bs
}

func (d *cborDecDriver) DecodeBytes(bs []byte, isstring, zerocopy bool) (bsOut []byte) {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == cborBdNil || d.bd == cborBdUndefined {
		d.bdRead = false
		return nil
	}
	if d.bd == cborBdIndefiniteBytes || d.bd == cborBdIndefiniteString {
		if bs == nil {
			return d.decAppendIndefiniteBytes(nil)
		}
		return d.decAppendIndefiniteBytes(bs[:0])
	}
	clen := d.decLen()
	d.bdRead = false
	if zerocopy {
		if d.br {
			return d.r.readx(clen)
		} else if len(bs) == 0 {
			bs = d.b[:]
		}
	}
	return decByteSlice(d.r, clen, bs)
}

func (d *cborDecDriver) DecodeString() (s string) {
	return string(d.DecodeBytes(d.b[:], true, true))
}

func (d *cborDecDriver) DecodeExt(rv interface{}, xtag uint64, ext Ext) (realxtag uint64) {
	if !d.bdRead {
		d.readNextBd()
	}
	u := d.decUint()
	d.bdRead = false
	realxtag = u
	if ext == nil {
		re := rv.(*RawExt)
		re.Tag = realxtag
		d.d.decode(&re.Value)
	} else if xtag != realxtag {
		d.d.errorf("Wrong extension tag. Got %b. Expecting: %v", realxtag, xtag)
		return
	} else {
		var v interface{}
		d.d.decode(&v)
		ext.UpdateExt(rv, v)
	}
	d.bdRead = false
	return
}

func (d *cborDecDriver) DecodeNaked() (v interface{}, vt valueType, decodeFurther bool) {
	if !d.bdRead {
		d.readNextBd()
	}

	switch d.bd {
	case cborBdNil:
		vt = valueTypeNil
	case cborBdFalse:
		vt = valueTypeBool
		v = false
	case cborBdTrue:
		vt = valueTypeBool
		v = true
	case cborBdFloat16, cborBdFloat32:
		vt = valueTypeFloat
		v = d.DecodeFloat(true)
	case cborBdFloat64:
		vt = valueTypeFloat
		v = d.DecodeFloat(false)
	case cborBdIndefiniteBytes:
		vt = valueTypeBytes
		v = d.DecodeBytes(nil, false, false)
	case cborBdIndefiniteString:
		vt = valueTypeString
		v = d.DecodeString()
	case cborBdIndefiniteArray:
		vt = valueTypeArray
		decodeFurther = true
	case cborBdIndefiniteMap:
		vt = valueTypeMap
		decodeFurther = true
	default:
		switch {
		case d.bd >= cborBaseUint && d.bd < cborBaseNegInt:
			if d.h.SignedInteger {
				vt = valueTypeInt
				v = d.DecodeInt(64)
			} else {
				vt = valueTypeUint
				v = d.DecodeUint(64)
			}
		case d.bd >= cborBaseNegInt && d.bd < cborBaseBytes:
			vt = valueTypeInt
			v = d.DecodeInt(64)
		case d.bd >= cborBaseBytes && d.bd < cborBaseString:
			vt = valueTypeBytes
			v = d.DecodeBytes(nil, false, false)
		case d.bd >= cborBaseString && d.bd < cborBaseArray:
			vt = valueTypeString
			v = d.DecodeString()
		case d.bd >= cborBaseArray && d.bd < cborBaseMap:
			vt = valueTypeArray
			decodeFurther = true
		case d.bd >= cborBaseMap && d.bd < cborBaseTag:
			vt = valueTypeMap
			decodeFurther = true
		case d.bd >= cborBaseTag && d.bd < cborBaseSimple:
			vt = valueTypeExt
			var re RawExt
			ui := d.decUint()
			d.bdRead = false
			re.Tag = ui
			d.d.decode(&re.Value)
			v = &re
			// decodeFurther = true
		default:
			d.d.errorf("decodeNaked: Unrecognized d.bd: 0x%x", d.bd)
			return
		}
	}

	if !decodeFurther {
		d.bdRead = false
	}
	return
}

// -------------------------

// CborHandle is a Handle for the CBOR encoding format,
// defined at http://tools.ietf.org/html/rfc7049 and documented further at http://cbor.io .
//
// CBOR is comprehensively supported, including support for:
//   - indefinite-length arrays/maps/bytes/strings
//   - (extension) tags in range 0..0xffff (0 .. 65535)
//   - half, single and double-precision floats
//   - all numbers (1, 2, 4 and 8-byte signed and unsigned integers)
//   - nil, true, false, ...
//   - arrays and maps, bytes and text strings
//
// None of the optional extensions (with tags) defined in the spec are supported out-of-the-box.
// Users can implement them as needed (using SetExt), including spec-documented ones:
//   - timestamp, BigNum, BigFloat, Decimals, Encoded Text (e.g. URL, regexp, base64, MIME Message), etc.
//
// To encode with indefinite lengths (streaming), users will use
// (Must)Encode methods of *Encoder, along with writing CborStreamXXX constants.
//
// For example, to encode "one-byte" as an indefinite length string:
//     var buf bytes.Buffer
//     e := NewEncoder(&buf, new(CborHandle))
//     buf.WriteByte(CborStreamString)
//     e.MustEncode("one-")
//     e.MustEncode("byte")
//     buf.WriteByte(CborStreamBreak)
//     encodedBytes := buf.Bytes()
//     var vv interface{}
//     NewDecoderBytes(buf.Bytes(), new(CborHandle)).MustDecode(&vv)
//     // Now, vv contains the same string "one-byte"
//
type CborHandle struct {
	BasicHandle
	binaryEncodingType
}

func (h *CborHandle) newEncDriver(e *Encoder) encDriver {
	return &cborEncDriver{e: e, w: e.w, h: h}
}

func (h *CborHandle) newDecDriver(d *Decoder) decDriver {
	return &cborDecDriver{d: d, r: d.r, h: h, br: d.bytes}
}

var _ decDriver = (*cborDecDriver)(nil)
var _ encDriver = (*cborEncDriver)(nil)
