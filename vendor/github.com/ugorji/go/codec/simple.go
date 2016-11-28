// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"math"
	"reflect"
)

const (
	_               uint8 = iota
	simpleVdNil           = 1
	simpleVdFalse         = 2
	simpleVdTrue          = 3
	simpleVdFloat32       = 4
	simpleVdFloat64       = 5

	// each lasts for 4 (ie n, n+1, n+2, n+3)
	simpleVdPosInt = 8
	simpleVdNegInt = 12

	// containers: each lasts for 4 (ie n, n+1, n+2, ... n+7)
	simpleVdString    = 216
	simpleVdByteArray = 224
	simpleVdArray     = 232
	simpleVdMap       = 240
	simpleVdExt       = 248
)

type simpleEncDriver struct {
	noBuiltInTypes
	encNoSeparator
	e *Encoder
	h *SimpleHandle
	w encWriter
	b [8]byte
}

func (e *simpleEncDriver) EncodeNil() {
	e.w.writen1(simpleVdNil)
}

func (e *simpleEncDriver) EncodeBool(b bool) {
	if b {
		e.w.writen1(simpleVdTrue)
	} else {
		e.w.writen1(simpleVdFalse)
	}
}

func (e *simpleEncDriver) EncodeFloat32(f float32) {
	e.w.writen1(simpleVdFloat32)
	bigenHelper{e.b[:4], e.w}.writeUint32(math.Float32bits(f))
}

func (e *simpleEncDriver) EncodeFloat64(f float64) {
	e.w.writen1(simpleVdFloat64)
	bigenHelper{e.b[:8], e.w}.writeUint64(math.Float64bits(f))
}

func (e *simpleEncDriver) EncodeInt(v int64) {
	if v < 0 {
		e.encUint(uint64(-v), simpleVdNegInt)
	} else {
		e.encUint(uint64(v), simpleVdPosInt)
	}
}

func (e *simpleEncDriver) EncodeUint(v uint64) {
	e.encUint(v, simpleVdPosInt)
}

func (e *simpleEncDriver) encUint(v uint64, bd uint8) {
	if v <= math.MaxUint8 {
		e.w.writen2(bd, uint8(v))
	} else if v <= math.MaxUint16 {
		e.w.writen1(bd + 1)
		bigenHelper{e.b[:2], e.w}.writeUint16(uint16(v))
	} else if v <= math.MaxUint32 {
		e.w.writen1(bd + 2)
		bigenHelper{e.b[:4], e.w}.writeUint32(uint32(v))
	} else { // if v <= math.MaxUint64 {
		e.w.writen1(bd + 3)
		bigenHelper{e.b[:8], e.w}.writeUint64(v)
	}
}

func (e *simpleEncDriver) encLen(bd byte, length int) {
	if length == 0 {
		e.w.writen1(bd)
	} else if length <= math.MaxUint8 {
		e.w.writen1(bd + 1)
		e.w.writen1(uint8(length))
	} else if length <= math.MaxUint16 {
		e.w.writen1(bd + 2)
		bigenHelper{e.b[:2], e.w}.writeUint16(uint16(length))
	} else if int64(length) <= math.MaxUint32 {
		e.w.writen1(bd + 3)
		bigenHelper{e.b[:4], e.w}.writeUint32(uint32(length))
	} else {
		e.w.writen1(bd + 4)
		bigenHelper{e.b[:8], e.w}.writeUint64(uint64(length))
	}
}

func (e *simpleEncDriver) EncodeExt(rv interface{}, xtag uint64, ext Ext, _ *Encoder) {
	bs := ext.WriteExt(rv)
	if bs == nil {
		e.EncodeNil()
		return
	}
	e.encodeExtPreamble(uint8(xtag), len(bs))
	e.w.writeb(bs)
}

func (e *simpleEncDriver) EncodeRawExt(re *RawExt, _ *Encoder) {
	e.encodeExtPreamble(uint8(re.Tag), len(re.Data))
	e.w.writeb(re.Data)
}

func (e *simpleEncDriver) encodeExtPreamble(xtag byte, length int) {
	e.encLen(simpleVdExt, length)
	e.w.writen1(xtag)
}

func (e *simpleEncDriver) EncodeArrayStart(length int) {
	e.encLen(simpleVdArray, length)
}

func (e *simpleEncDriver) EncodeMapStart(length int) {
	e.encLen(simpleVdMap, length)
}

func (e *simpleEncDriver) EncodeString(c charEncoding, v string) {
	e.encLen(simpleVdString, len(v))
	e.w.writestr(v)
}

func (e *simpleEncDriver) EncodeSymbol(v string) {
	e.EncodeString(c_UTF8, v)
}

func (e *simpleEncDriver) EncodeStringBytes(c charEncoding, v []byte) {
	e.encLen(simpleVdByteArray, len(v))
	e.w.writeb(v)
}

//------------------------------------

type simpleDecDriver struct {
	d      *Decoder
	h      *SimpleHandle
	r      decReader
	bdRead bool
	bd     byte
	br     bool // bytes reader
	noBuiltInTypes
	noStreamingCodec
	decNoSeparator
	b [scratchByteArrayLen]byte
}

func (d *simpleDecDriver) readNextBd() {
	d.bd = d.r.readn1()
	d.bdRead = true
}

func (d *simpleDecDriver) uncacheRead() {
	if d.bdRead {
		d.r.unreadn1()
		d.bdRead = false
	}
}

func (d *simpleDecDriver) ContainerType() (vt valueType) {
	if d.bd == simpleVdNil {
		return valueTypeNil
	} else if d.bd == simpleVdByteArray || d.bd == simpleVdByteArray+1 ||
		d.bd == simpleVdByteArray+2 || d.bd == simpleVdByteArray+3 || d.bd == simpleVdByteArray+4 {
		return valueTypeBytes
	} else if d.bd == simpleVdString || d.bd == simpleVdString+1 ||
		d.bd == simpleVdString+2 || d.bd == simpleVdString+3 || d.bd == simpleVdString+4 {
		return valueTypeString
	} else if d.bd == simpleVdArray || d.bd == simpleVdArray+1 ||
		d.bd == simpleVdArray+2 || d.bd == simpleVdArray+3 || d.bd == simpleVdArray+4 {
		return valueTypeArray
	} else if d.bd == simpleVdMap || d.bd == simpleVdMap+1 ||
		d.bd == simpleVdMap+2 || d.bd == simpleVdMap+3 || d.bd == simpleVdMap+4 {
		return valueTypeMap
	} else {
		// d.d.errorf("isContainerType: unsupported parameter: %v", vt)
	}
	return valueTypeUnset
}

func (d *simpleDecDriver) TryDecodeAsNil() bool {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == simpleVdNil {
		d.bdRead = false
		return true
	}
	return false
}

func (d *simpleDecDriver) decCheckInteger() (ui uint64, neg bool) {
	if !d.bdRead {
		d.readNextBd()
	}
	switch d.bd {
	case simpleVdPosInt:
		ui = uint64(d.r.readn1())
	case simpleVdPosInt + 1:
		ui = uint64(bigen.Uint16(d.r.readx(2)))
	case simpleVdPosInt + 2:
		ui = uint64(bigen.Uint32(d.r.readx(4)))
	case simpleVdPosInt + 3:
		ui = uint64(bigen.Uint64(d.r.readx(8)))
	case simpleVdNegInt:
		ui = uint64(d.r.readn1())
		neg = true
	case simpleVdNegInt + 1:
		ui = uint64(bigen.Uint16(d.r.readx(2)))
		neg = true
	case simpleVdNegInt + 2:
		ui = uint64(bigen.Uint32(d.r.readx(4)))
		neg = true
	case simpleVdNegInt + 3:
		ui = uint64(bigen.Uint64(d.r.readx(8)))
		neg = true
	default:
		d.d.errorf("decIntAny: Integer only valid from pos/neg integer1..8. Invalid descriptor: %v", d.bd)
		return
	}
	// don't do this check, because callers may only want the unsigned value.
	// if ui > math.MaxInt64 {
	// 	d.d.errorf("decIntAny: Integer out of range for signed int64: %v", ui)
	//		return
	// }
	return
}

func (d *simpleDecDriver) DecodeInt(bitsize uint8) (i int64) {
	ui, neg := d.decCheckInteger()
	i, overflow := chkOvf.SignedInt(ui)
	if overflow {
		d.d.errorf("simple: overflow converting %v to signed integer", ui)
		return
	}
	if neg {
		i = -i
	}
	if chkOvf.Int(i, bitsize) {
		d.d.errorf("simple: overflow integer: %v", i)
		return
	}
	d.bdRead = false
	return
}

func (d *simpleDecDriver) DecodeUint(bitsize uint8) (ui uint64) {
	ui, neg := d.decCheckInteger()
	if neg {
		d.d.errorf("Assigning negative signed value to unsigned type")
		return
	}
	if chkOvf.Uint(ui, bitsize) {
		d.d.errorf("simple: overflow integer: %v", ui)
		return
	}
	d.bdRead = false
	return
}

func (d *simpleDecDriver) DecodeFloat(chkOverflow32 bool) (f float64) {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == simpleVdFloat32 {
		f = float64(math.Float32frombits(bigen.Uint32(d.r.readx(4))))
	} else if d.bd == simpleVdFloat64 {
		f = math.Float64frombits(bigen.Uint64(d.r.readx(8)))
	} else {
		if d.bd >= simpleVdPosInt && d.bd <= simpleVdNegInt+3 {
			f = float64(d.DecodeInt(64))
		} else {
			d.d.errorf("Float only valid from float32/64: Invalid descriptor: %v", d.bd)
			return
		}
	}
	if chkOverflow32 && chkOvf.Float32(f) {
		d.d.errorf("msgpack: float32 overflow: %v", f)
		return
	}
	d.bdRead = false
	return
}

// bool can be decoded from bool only (single byte).
func (d *simpleDecDriver) DecodeBool() (b bool) {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == simpleVdTrue {
		b = true
	} else if d.bd == simpleVdFalse {
	} else {
		d.d.errorf("Invalid single-byte value for bool: %s: %x", msgBadDesc, d.bd)
		return
	}
	d.bdRead = false
	return
}

func (d *simpleDecDriver) ReadMapStart() (length int) {
	d.bdRead = false
	return d.decLen()
}

func (d *simpleDecDriver) ReadArrayStart() (length int) {
	d.bdRead = false
	return d.decLen()
}

func (d *simpleDecDriver) decLen() int {
	switch d.bd % 8 {
	case 0:
		return 0
	case 1:
		return int(d.r.readn1())
	case 2:
		return int(bigen.Uint16(d.r.readx(2)))
	case 3:
		ui := uint64(bigen.Uint32(d.r.readx(4)))
		if chkOvf.Uint(ui, intBitsize) {
			d.d.errorf("simple: overflow integer: %v", ui)
			return 0
		}
		return int(ui)
	case 4:
		ui := bigen.Uint64(d.r.readx(8))
		if chkOvf.Uint(ui, intBitsize) {
			d.d.errorf("simple: overflow integer: %v", ui)
			return 0
		}
		return int(ui)
	}
	d.d.errorf("decLen: Cannot read length: bd%%8 must be in range 0..4. Got: %d", d.bd%8)
	return -1
}

func (d *simpleDecDriver) DecodeString() (s string) {
	return string(d.DecodeBytes(d.b[:], true, true))
}

func (d *simpleDecDriver) DecodeBytes(bs []byte, isstring, zerocopy bool) (bsOut []byte) {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == simpleVdNil {
		d.bdRead = false
		return
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

func (d *simpleDecDriver) DecodeExt(rv interface{}, xtag uint64, ext Ext) (realxtag uint64) {
	if xtag > 0xff {
		d.d.errorf("decodeExt: tag must be <= 0xff; got: %v", xtag)
		return
	}
	realxtag1, xbs := d.decodeExtV(ext != nil, uint8(xtag))
	realxtag = uint64(realxtag1)
	if ext == nil {
		re := rv.(*RawExt)
		re.Tag = realxtag
		re.Data = detachZeroCopyBytes(d.br, re.Data, xbs)
	} else {
		ext.ReadExt(rv, xbs)
	}
	return
}

func (d *simpleDecDriver) decodeExtV(verifyTag bool, tag byte) (xtag byte, xbs []byte) {
	if !d.bdRead {
		d.readNextBd()
	}
	switch d.bd {
	case simpleVdExt, simpleVdExt + 1, simpleVdExt + 2, simpleVdExt + 3, simpleVdExt + 4:
		l := d.decLen()
		xtag = d.r.readn1()
		if verifyTag && xtag != tag {
			d.d.errorf("Wrong extension tag. Got %b. Expecting: %v", xtag, tag)
			return
		}
		xbs = d.r.readx(l)
	case simpleVdByteArray, simpleVdByteArray + 1, simpleVdByteArray + 2, simpleVdByteArray + 3, simpleVdByteArray + 4:
		xbs = d.DecodeBytes(nil, false, true)
	default:
		d.d.errorf("Invalid d.bd for extensions (Expecting extensions or byte array). Got: 0x%x", d.bd)
		return
	}
	d.bdRead = false
	return
}

func (d *simpleDecDriver) DecodeNaked() {
	if !d.bdRead {
		d.readNextBd()
	}

	n := &d.d.n
	var decodeFurther bool

	switch d.bd {
	case simpleVdNil:
		n.v = valueTypeNil
	case simpleVdFalse:
		n.v = valueTypeBool
		n.b = false
	case simpleVdTrue:
		n.v = valueTypeBool
		n.b = true
	case simpleVdPosInt, simpleVdPosInt + 1, simpleVdPosInt + 2, simpleVdPosInt + 3:
		if d.h.SignedInteger {
			n.v = valueTypeInt
			n.i = d.DecodeInt(64)
		} else {
			n.v = valueTypeUint
			n.u = d.DecodeUint(64)
		}
	case simpleVdNegInt, simpleVdNegInt + 1, simpleVdNegInt + 2, simpleVdNegInt + 3:
		n.v = valueTypeInt
		n.i = d.DecodeInt(64)
	case simpleVdFloat32:
		n.v = valueTypeFloat
		n.f = d.DecodeFloat(true)
	case simpleVdFloat64:
		n.v = valueTypeFloat
		n.f = d.DecodeFloat(false)
	case simpleVdString, simpleVdString + 1, simpleVdString + 2, simpleVdString + 3, simpleVdString + 4:
		n.v = valueTypeString
		n.s = d.DecodeString()
	case simpleVdByteArray, simpleVdByteArray + 1, simpleVdByteArray + 2, simpleVdByteArray + 3, simpleVdByteArray + 4:
		n.v = valueTypeBytes
		n.l = d.DecodeBytes(nil, false, false)
	case simpleVdExt, simpleVdExt + 1, simpleVdExt + 2, simpleVdExt + 3, simpleVdExt + 4:
		n.v = valueTypeExt
		l := d.decLen()
		n.u = uint64(d.r.readn1())
		n.l = d.r.readx(l)
	case simpleVdArray, simpleVdArray + 1, simpleVdArray + 2, simpleVdArray + 3, simpleVdArray + 4:
		n.v = valueTypeArray
		decodeFurther = true
	case simpleVdMap, simpleVdMap + 1, simpleVdMap + 2, simpleVdMap + 3, simpleVdMap + 4:
		n.v = valueTypeMap
		decodeFurther = true
	default:
		d.d.errorf("decodeNaked: Unrecognized d.bd: 0x%x", d.bd)
	}

	if !decodeFurther {
		d.bdRead = false
	}
	return
}

//------------------------------------

// SimpleHandle is a Handle for a very simple encoding format.
//
// simple is a simplistic codec similar to binc, but not as compact.
//   - Encoding of a value is always preceded by the descriptor byte (bd)
//   - True, false, nil are encoded fully in 1 byte (the descriptor)
//   - Integers (intXXX, uintXXX) are encoded in 1, 2, 4 or 8 bytes (plus a descriptor byte).
//     There are positive (uintXXX and intXXX >= 0) and negative (intXXX < 0) integers.
//   - Floats are encoded in 4 or 8 bytes (plus a descriptor byte)
//   - Lenght of containers (strings, bytes, array, map, extensions)
//     are encoded in 0, 1, 2, 4 or 8 bytes.
//     Zero-length containers have no length encoded.
//     For others, the number of bytes is given by pow(2, bd%3)
//   - maps are encoded as [bd] [length] [[key][value]]...
//   - arrays are encoded as [bd] [length] [value]...
//   - extensions are encoded as [bd] [length] [tag] [byte]...
//   - strings/bytearrays are encoded as [bd] [length] [byte]...
//
// The full spec will be published soon.
type SimpleHandle struct {
	BasicHandle
	binaryEncodingType
}

func (h *SimpleHandle) SetBytesExt(rt reflect.Type, tag uint64, ext BytesExt) (err error) {
	return h.SetExt(rt, tag, &setExtWrapper{b: ext})
}

func (h *SimpleHandle) newEncDriver(e *Encoder) encDriver {
	return &simpleEncDriver{e: e, w: e.w, h: h}
}

func (h *SimpleHandle) newDecDriver(d *Decoder) decDriver {
	return &simpleDecDriver{d: d, r: d.r, h: h, br: d.bytes}
}

func (e *simpleEncDriver) reset() {
	e.w = e.e.w
}

func (d *simpleDecDriver) reset() {
	d.r = d.d.r
	d.bd, d.bdRead = 0, false
}

var _ decDriver = (*simpleDecDriver)(nil)
var _ encDriver = (*simpleEncDriver)(nil)
