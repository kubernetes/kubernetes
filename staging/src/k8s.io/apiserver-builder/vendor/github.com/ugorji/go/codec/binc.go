// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"math"
	"reflect"
	"time"
)

const bincDoPrune = true // No longer needed. Needed before as C lib did not support pruning.

// vd as low 4 bits (there are 16 slots)
const (
	bincVdSpecial byte = iota
	bincVdPosInt
	bincVdNegInt
	bincVdFloat

	bincVdString
	bincVdByteArray
	bincVdArray
	bincVdMap

	bincVdTimestamp
	bincVdSmallInt
	bincVdUnicodeOther
	bincVdSymbol

	bincVdDecimal
	_               // open slot
	_               // open slot
	bincVdCustomExt = 0x0f
)

const (
	bincSpNil byte = iota
	bincSpFalse
	bincSpTrue
	bincSpNan
	bincSpPosInf
	bincSpNegInf
	bincSpZeroFloat
	bincSpZero
	bincSpNegOne
)

const (
	bincFlBin16 byte = iota
	bincFlBin32
	_ // bincFlBin32e
	bincFlBin64
	_ // bincFlBin64e
	// others not currently supported
)

type bincEncDriver struct {
	e *Encoder
	w encWriter
	m map[string]uint16 // symbols
	b [scratchByteArrayLen]byte
	s uint16 // symbols sequencer
	encNoSeparator
}

func (e *bincEncDriver) IsBuiltinType(rt uintptr) bool {
	return rt == timeTypId
}

func (e *bincEncDriver) EncodeBuiltin(rt uintptr, v interface{}) {
	if rt == timeTypId {
		var bs []byte
		switch x := v.(type) {
		case time.Time:
			bs = encodeTime(x)
		case *time.Time:
			bs = encodeTime(*x)
		default:
			e.e.errorf("binc error encoding builtin: expect time.Time, received %T", v)
		}
		e.w.writen1(bincVdTimestamp<<4 | uint8(len(bs)))
		e.w.writeb(bs)
	}
}

func (e *bincEncDriver) EncodeNil() {
	e.w.writen1(bincVdSpecial<<4 | bincSpNil)
}

func (e *bincEncDriver) EncodeBool(b bool) {
	if b {
		e.w.writen1(bincVdSpecial<<4 | bincSpTrue)
	} else {
		e.w.writen1(bincVdSpecial<<4 | bincSpFalse)
	}
}

func (e *bincEncDriver) EncodeFloat32(f float32) {
	if f == 0 {
		e.w.writen1(bincVdSpecial<<4 | bincSpZeroFloat)
		return
	}
	e.w.writen1(bincVdFloat<<4 | bincFlBin32)
	bigenHelper{e.b[:4], e.w}.writeUint32(math.Float32bits(f))
}

func (e *bincEncDriver) EncodeFloat64(f float64) {
	if f == 0 {
		e.w.writen1(bincVdSpecial<<4 | bincSpZeroFloat)
		return
	}
	bigen.PutUint64(e.b[:8], math.Float64bits(f))
	if bincDoPrune {
		i := 7
		for ; i >= 0 && (e.b[i] == 0); i-- {
		}
		i++
		if i <= 6 {
			e.w.writen1(bincVdFloat<<4 | 0x8 | bincFlBin64)
			e.w.writen1(byte(i))
			e.w.writeb(e.b[:i])
			return
		}
	}
	e.w.writen1(bincVdFloat<<4 | bincFlBin64)
	e.w.writeb(e.b[:8])
}

func (e *bincEncDriver) encIntegerPrune(bd byte, pos bool, v uint64, lim uint8) {
	if lim == 4 {
		bigen.PutUint32(e.b[:lim], uint32(v))
	} else {
		bigen.PutUint64(e.b[:lim], v)
	}
	if bincDoPrune {
		i := pruneSignExt(e.b[:lim], pos)
		e.w.writen1(bd | lim - 1 - byte(i))
		e.w.writeb(e.b[i:lim])
	} else {
		e.w.writen1(bd | lim - 1)
		e.w.writeb(e.b[:lim])
	}
}

func (e *bincEncDriver) EncodeInt(v int64) {
	const nbd byte = bincVdNegInt << 4
	if v >= 0 {
		e.encUint(bincVdPosInt<<4, true, uint64(v))
	} else if v == -1 {
		e.w.writen1(bincVdSpecial<<4 | bincSpNegOne)
	} else {
		e.encUint(bincVdNegInt<<4, false, uint64(-v))
	}
}

func (e *bincEncDriver) EncodeUint(v uint64) {
	e.encUint(bincVdPosInt<<4, true, v)
}

func (e *bincEncDriver) encUint(bd byte, pos bool, v uint64) {
	if v == 0 {
		e.w.writen1(bincVdSpecial<<4 | bincSpZero)
	} else if pos && v >= 1 && v <= 16 {
		e.w.writen1(bincVdSmallInt<<4 | byte(v-1))
	} else if v <= math.MaxUint8 {
		e.w.writen2(bd|0x0, byte(v))
	} else if v <= math.MaxUint16 {
		e.w.writen1(bd | 0x01)
		bigenHelper{e.b[:2], e.w}.writeUint16(uint16(v))
	} else if v <= math.MaxUint32 {
		e.encIntegerPrune(bd, pos, v, 4)
	} else {
		e.encIntegerPrune(bd, pos, v, 8)
	}
}

func (e *bincEncDriver) EncodeExt(rv interface{}, xtag uint64, ext Ext, _ *Encoder) {
	bs := ext.WriteExt(rv)
	if bs == nil {
		e.EncodeNil()
		return
	}
	e.encodeExtPreamble(uint8(xtag), len(bs))
	e.w.writeb(bs)
}

func (e *bincEncDriver) EncodeRawExt(re *RawExt, _ *Encoder) {
	e.encodeExtPreamble(uint8(re.Tag), len(re.Data))
	e.w.writeb(re.Data)
}

func (e *bincEncDriver) encodeExtPreamble(xtag byte, length int) {
	e.encLen(bincVdCustomExt<<4, uint64(length))
	e.w.writen1(xtag)
}

func (e *bincEncDriver) EncodeArrayStart(length int) {
	e.encLen(bincVdArray<<4, uint64(length))
}

func (e *bincEncDriver) EncodeMapStart(length int) {
	e.encLen(bincVdMap<<4, uint64(length))
}

func (e *bincEncDriver) EncodeString(c charEncoding, v string) {
	l := uint64(len(v))
	e.encBytesLen(c, l)
	if l > 0 {
		e.w.writestr(v)
	}
}

func (e *bincEncDriver) EncodeSymbol(v string) {
	// if WriteSymbolsNoRefs {
	// 	e.encodeString(c_UTF8, v)
	// 	return
	// }

	//symbols only offer benefit when string length > 1.
	//This is because strings with length 1 take only 2 bytes to store
	//(bd with embedded length, and single byte for string val).

	l := len(v)
	if l == 0 {
		e.encBytesLen(c_UTF8, 0)
		return
	} else if l == 1 {
		e.encBytesLen(c_UTF8, 1)
		e.w.writen1(v[0])
		return
	}
	if e.m == nil {
		e.m = make(map[string]uint16, 16)
	}
	ui, ok := e.m[v]
	if ok {
		if ui <= math.MaxUint8 {
			e.w.writen2(bincVdSymbol<<4, byte(ui))
		} else {
			e.w.writen1(bincVdSymbol<<4 | 0x8)
			bigenHelper{e.b[:2], e.w}.writeUint16(ui)
		}
	} else {
		e.s++
		ui = e.s
		//ui = uint16(atomic.AddUint32(&e.s, 1))
		e.m[v] = ui
		var lenprec uint8
		if l <= math.MaxUint8 {
			// lenprec = 0
		} else if l <= math.MaxUint16 {
			lenprec = 1
		} else if int64(l) <= math.MaxUint32 {
			lenprec = 2
		} else {
			lenprec = 3
		}
		if ui <= math.MaxUint8 {
			e.w.writen2(bincVdSymbol<<4|0x0|0x4|lenprec, byte(ui))
		} else {
			e.w.writen1(bincVdSymbol<<4 | 0x8 | 0x4 | lenprec)
			bigenHelper{e.b[:2], e.w}.writeUint16(ui)
		}
		if lenprec == 0 {
			e.w.writen1(byte(l))
		} else if lenprec == 1 {
			bigenHelper{e.b[:2], e.w}.writeUint16(uint16(l))
		} else if lenprec == 2 {
			bigenHelper{e.b[:4], e.w}.writeUint32(uint32(l))
		} else {
			bigenHelper{e.b[:8], e.w}.writeUint64(uint64(l))
		}
		e.w.writestr(v)
	}
}

func (e *bincEncDriver) EncodeStringBytes(c charEncoding, v []byte) {
	l := uint64(len(v))
	e.encBytesLen(c, l)
	if l > 0 {
		e.w.writeb(v)
	}
}

func (e *bincEncDriver) encBytesLen(c charEncoding, length uint64) {
	//TODO: support bincUnicodeOther (for now, just use string or bytearray)
	if c == c_RAW {
		e.encLen(bincVdByteArray<<4, length)
	} else {
		e.encLen(bincVdString<<4, length)
	}
}

func (e *bincEncDriver) encLen(bd byte, l uint64) {
	if l < 12 {
		e.w.writen1(bd | uint8(l+4))
	} else {
		e.encLenNumber(bd, l)
	}
}

func (e *bincEncDriver) encLenNumber(bd byte, v uint64) {
	if v <= math.MaxUint8 {
		e.w.writen2(bd, byte(v))
	} else if v <= math.MaxUint16 {
		e.w.writen1(bd | 0x01)
		bigenHelper{e.b[:2], e.w}.writeUint16(uint16(v))
	} else if v <= math.MaxUint32 {
		e.w.writen1(bd | 0x02)
		bigenHelper{e.b[:4], e.w}.writeUint32(uint32(v))
	} else {
		e.w.writen1(bd | 0x03)
		bigenHelper{e.b[:8], e.w}.writeUint64(uint64(v))
	}
}

//------------------------------------

type bincDecSymbol struct {
	s string
	b []byte
	i uint16
}

type bincDecDriver struct {
	d      *Decoder
	h      *BincHandle
	r      decReader
	br     bool // bytes reader
	bdRead bool
	bd     byte
	vd     byte
	vs     byte
	noStreamingCodec
	decNoSeparator
	b [scratchByteArrayLen]byte

	// linear searching on this slice is ok,
	// because we typically expect < 32 symbols in each stream.
	s []bincDecSymbol
}

func (d *bincDecDriver) readNextBd() {
	d.bd = d.r.readn1()
	d.vd = d.bd >> 4
	d.vs = d.bd & 0x0f
	d.bdRead = true
}

func (d *bincDecDriver) uncacheRead() {
	if d.bdRead {
		d.r.unreadn1()
		d.bdRead = false
	}
}

func (d *bincDecDriver) ContainerType() (vt valueType) {
	if d.vd == bincVdSpecial && d.vs == bincSpNil {
		return valueTypeNil
	} else if d.vd == bincVdByteArray {
		return valueTypeBytes
	} else if d.vd == bincVdString {
		return valueTypeString
	} else if d.vd == bincVdArray {
		return valueTypeArray
	} else if d.vd == bincVdMap {
		return valueTypeMap
	} else {
		// d.d.errorf("isContainerType: unsupported parameter: %v", vt)
	}
	return valueTypeUnset
}

func (d *bincDecDriver) TryDecodeAsNil() bool {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == bincVdSpecial<<4|bincSpNil {
		d.bdRead = false
		return true
	}
	return false
}

func (d *bincDecDriver) IsBuiltinType(rt uintptr) bool {
	return rt == timeTypId
}

func (d *bincDecDriver) DecodeBuiltin(rt uintptr, v interface{}) {
	if !d.bdRead {
		d.readNextBd()
	}
	if rt == timeTypId {
		if d.vd != bincVdTimestamp {
			d.d.errorf("Invalid d.vd. Expecting 0x%x. Received: 0x%x", bincVdTimestamp, d.vd)
			return
		}
		tt, err := decodeTime(d.r.readx(int(d.vs)))
		if err != nil {
			panic(err)
		}
		var vt *time.Time = v.(*time.Time)
		*vt = tt
		d.bdRead = false
	}
}

func (d *bincDecDriver) decFloatPre(vs, defaultLen byte) {
	if vs&0x8 == 0 {
		d.r.readb(d.b[0:defaultLen])
	} else {
		l := d.r.readn1()
		if l > 8 {
			d.d.errorf("At most 8 bytes used to represent float. Received: %v bytes", l)
			return
		}
		for i := l; i < 8; i++ {
			d.b[i] = 0
		}
		d.r.readb(d.b[0:l])
	}
}

func (d *bincDecDriver) decFloat() (f float64) {
	//if true { f = math.Float64frombits(bigen.Uint64(d.r.readx(8))); break; }
	if x := d.vs & 0x7; x == bincFlBin32 {
		d.decFloatPre(d.vs, 4)
		f = float64(math.Float32frombits(bigen.Uint32(d.b[0:4])))
	} else if x == bincFlBin64 {
		d.decFloatPre(d.vs, 8)
		f = math.Float64frombits(bigen.Uint64(d.b[0:8]))
	} else {
		d.d.errorf("only float32 and float64 are supported. d.vd: 0x%x, d.vs: 0x%x", d.vd, d.vs)
		return
	}
	return
}

func (d *bincDecDriver) decUint() (v uint64) {
	// need to inline the code (interface conversion and type assertion expensive)
	switch d.vs {
	case 0:
		v = uint64(d.r.readn1())
	case 1:
		d.r.readb(d.b[6:8])
		v = uint64(bigen.Uint16(d.b[6:8]))
	case 2:
		d.b[4] = 0
		d.r.readb(d.b[5:8])
		v = uint64(bigen.Uint32(d.b[4:8]))
	case 3:
		d.r.readb(d.b[4:8])
		v = uint64(bigen.Uint32(d.b[4:8]))
	case 4, 5, 6:
		lim := int(7 - d.vs)
		d.r.readb(d.b[lim:8])
		for i := 0; i < lim; i++ {
			d.b[i] = 0
		}
		v = uint64(bigen.Uint64(d.b[:8]))
	case 7:
		d.r.readb(d.b[:8])
		v = uint64(bigen.Uint64(d.b[:8]))
	default:
		d.d.errorf("unsigned integers with greater than 64 bits of precision not supported")
		return
	}
	return
}

func (d *bincDecDriver) decCheckInteger() (ui uint64, neg bool) {
	if !d.bdRead {
		d.readNextBd()
	}
	vd, vs := d.vd, d.vs
	if vd == bincVdPosInt {
		ui = d.decUint()
	} else if vd == bincVdNegInt {
		ui = d.decUint()
		neg = true
	} else if vd == bincVdSmallInt {
		ui = uint64(d.vs) + 1
	} else if vd == bincVdSpecial {
		if vs == bincSpZero {
			//i = 0
		} else if vs == bincSpNegOne {
			neg = true
			ui = 1
		} else {
			d.d.errorf("numeric decode fails for special value: d.vs: 0x%x", d.vs)
			return
		}
	} else {
		d.d.errorf("number can only be decoded from uint or int values. d.bd: 0x%x, d.vd: 0x%x", d.bd, d.vd)
		return
	}
	return
}

func (d *bincDecDriver) DecodeInt(bitsize uint8) (i int64) {
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
		d.d.errorf("binc: overflow integer: %v", i)
		return
	}
	d.bdRead = false
	return
}

func (d *bincDecDriver) DecodeUint(bitsize uint8) (ui uint64) {
	ui, neg := d.decCheckInteger()
	if neg {
		d.d.errorf("Assigning negative signed value to unsigned type")
		return
	}
	if chkOvf.Uint(ui, bitsize) {
		d.d.errorf("binc: overflow integer: %v", ui)
		return
	}
	d.bdRead = false
	return
}

func (d *bincDecDriver) DecodeFloat(chkOverflow32 bool) (f float64) {
	if !d.bdRead {
		d.readNextBd()
	}
	vd, vs := d.vd, d.vs
	if vd == bincVdSpecial {
		d.bdRead = false
		if vs == bincSpNan {
			return math.NaN()
		} else if vs == bincSpPosInf {
			return math.Inf(1)
		} else if vs == bincSpZeroFloat || vs == bincSpZero {
			return
		} else if vs == bincSpNegInf {
			return math.Inf(-1)
		} else {
			d.d.errorf("Invalid d.vs decoding float where d.vd=bincVdSpecial: %v", d.vs)
			return
		}
	} else if vd == bincVdFloat {
		f = d.decFloat()
	} else {
		f = float64(d.DecodeInt(64))
	}
	if chkOverflow32 && chkOvf.Float32(f) {
		d.d.errorf("binc: float32 overflow: %v", f)
		return
	}
	d.bdRead = false
	return
}

// bool can be decoded from bool only (single byte).
func (d *bincDecDriver) DecodeBool() (b bool) {
	if !d.bdRead {
		d.readNextBd()
	}
	if bd := d.bd; bd == (bincVdSpecial | bincSpFalse) {
		// b = false
	} else if bd == (bincVdSpecial | bincSpTrue) {
		b = true
	} else {
		d.d.errorf("Invalid single-byte value for bool: %s: %x", msgBadDesc, d.bd)
		return
	}
	d.bdRead = false
	return
}

func (d *bincDecDriver) ReadMapStart() (length int) {
	if d.vd != bincVdMap {
		d.d.errorf("Invalid d.vd for map. Expecting 0x%x. Got: 0x%x", bincVdMap, d.vd)
		return
	}
	length = d.decLen()
	d.bdRead = false
	return
}

func (d *bincDecDriver) ReadArrayStart() (length int) {
	if d.vd != bincVdArray {
		d.d.errorf("Invalid d.vd for array. Expecting 0x%x. Got: 0x%x", bincVdArray, d.vd)
		return
	}
	length = d.decLen()
	d.bdRead = false
	return
}

func (d *bincDecDriver) decLen() int {
	if d.vs > 3 {
		return int(d.vs - 4)
	}
	return int(d.decLenNumber())
}

func (d *bincDecDriver) decLenNumber() (v uint64) {
	if x := d.vs; x == 0 {
		v = uint64(d.r.readn1())
	} else if x == 1 {
		d.r.readb(d.b[6:8])
		v = uint64(bigen.Uint16(d.b[6:8]))
	} else if x == 2 {
		d.r.readb(d.b[4:8])
		v = uint64(bigen.Uint32(d.b[4:8]))
	} else {
		d.r.readb(d.b[:8])
		v = bigen.Uint64(d.b[:8])
	}
	return
}

func (d *bincDecDriver) decStringAndBytes(bs []byte, withString, zerocopy bool) (bs2 []byte, s string) {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == bincVdSpecial<<4|bincSpNil {
		d.bdRead = false
		return
	}
	var slen int = -1
	// var ok bool
	switch d.vd {
	case bincVdString, bincVdByteArray:
		slen = d.decLen()
		if zerocopy {
			if d.br {
				bs2 = d.r.readx(slen)
			} else if len(bs) == 0 {
				bs2 = decByteSlice(d.r, slen, d.b[:])
			} else {
				bs2 = decByteSlice(d.r, slen, bs)
			}
		} else {
			bs2 = decByteSlice(d.r, slen, bs)
		}
		if withString {
			s = string(bs2)
		}
	case bincVdSymbol:
		// zerocopy doesn't apply for symbols,
		// as the values must be stored in a table for later use.
		//
		//from vs: extract numSymbolBytes, containsStringVal, strLenPrecision,
		//extract symbol
		//if containsStringVal, read it and put in map
		//else look in map for string value
		var symbol uint16
		vs := d.vs
		if vs&0x8 == 0 {
			symbol = uint16(d.r.readn1())
		} else {
			symbol = uint16(bigen.Uint16(d.r.readx(2)))
		}
		if d.s == nil {
			d.s = make([]bincDecSymbol, 0, 16)
		}

		if vs&0x4 == 0 {
			for i := range d.s {
				j := &d.s[i]
				if j.i == symbol {
					bs2 = j.b
					if withString {
						if j.s == "" && bs2 != nil {
							j.s = string(bs2)
						}
						s = j.s
					}
					break
				}
			}
		} else {
			switch vs & 0x3 {
			case 0:
				slen = int(d.r.readn1())
			case 1:
				slen = int(bigen.Uint16(d.r.readx(2)))
			case 2:
				slen = int(bigen.Uint32(d.r.readx(4)))
			case 3:
				slen = int(bigen.Uint64(d.r.readx(8)))
			}
			// since using symbols, do not store any part of
			// the parameter bs in the map, as it might be a shared buffer.
			// bs2 = decByteSlice(d.r, slen, bs)
			bs2 = decByteSlice(d.r, slen, nil)
			if withString {
				s = string(bs2)
			}
			d.s = append(d.s, bincDecSymbol{i: symbol, s: s, b: bs2})
		}
	default:
		d.d.errorf("Invalid d.vd. Expecting string:0x%x, bytearray:0x%x or symbol: 0x%x. Got: 0x%x",
			bincVdString, bincVdByteArray, bincVdSymbol, d.vd)
		return
	}
	d.bdRead = false
	return
}

func (d *bincDecDriver) DecodeString() (s string) {
	// DecodeBytes does not accommodate symbols, whose impl stores string version in map.
	// Use decStringAndBytes directly.
	// return string(d.DecodeBytes(d.b[:], true, true))
	_, s = d.decStringAndBytes(d.b[:], true, true)
	return
}

func (d *bincDecDriver) DecodeBytes(bs []byte, isstring, zerocopy bool) (bsOut []byte) {
	if isstring {
		bsOut, _ = d.decStringAndBytes(bs, false, zerocopy)
		return
	}
	if !d.bdRead {
		d.readNextBd()
	}
	if d.bd == bincVdSpecial<<4|bincSpNil {
		d.bdRead = false
		return nil
	}
	var clen int
	if d.vd == bincVdString || d.vd == bincVdByteArray {
		clen = d.decLen()
	} else {
		d.d.errorf("Invalid d.vd for bytes. Expecting string:0x%x or bytearray:0x%x. Got: 0x%x",
			bincVdString, bincVdByteArray, d.vd)
		return
	}
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

func (d *bincDecDriver) DecodeExt(rv interface{}, xtag uint64, ext Ext) (realxtag uint64) {
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

func (d *bincDecDriver) decodeExtV(verifyTag bool, tag byte) (xtag byte, xbs []byte) {
	if !d.bdRead {
		d.readNextBd()
	}
	if d.vd == bincVdCustomExt {
		l := d.decLen()
		xtag = d.r.readn1()
		if verifyTag && xtag != tag {
			d.d.errorf("Wrong extension tag. Got %b. Expecting: %v", xtag, tag)
			return
		}
		xbs = d.r.readx(l)
	} else if d.vd == bincVdByteArray {
		xbs = d.DecodeBytes(nil, false, true)
	} else {
		d.d.errorf("Invalid d.vd for extensions (Expecting extensions or byte array). Got: 0x%x", d.vd)
		return
	}
	d.bdRead = false
	return
}

func (d *bincDecDriver) DecodeNaked() {
	if !d.bdRead {
		d.readNextBd()
	}

	n := &d.d.n
	var decodeFurther bool

	switch d.vd {
	case bincVdSpecial:
		switch d.vs {
		case bincSpNil:
			n.v = valueTypeNil
		case bincSpFalse:
			n.v = valueTypeBool
			n.b = false
		case bincSpTrue:
			n.v = valueTypeBool
			n.b = true
		case bincSpNan:
			n.v = valueTypeFloat
			n.f = math.NaN()
		case bincSpPosInf:
			n.v = valueTypeFloat
			n.f = math.Inf(1)
		case bincSpNegInf:
			n.v = valueTypeFloat
			n.f = math.Inf(-1)
		case bincSpZeroFloat:
			n.v = valueTypeFloat
			n.f = float64(0)
		case bincSpZero:
			n.v = valueTypeUint
			n.u = uint64(0) // int8(0)
		case bincSpNegOne:
			n.v = valueTypeInt
			n.i = int64(-1) // int8(-1)
		default:
			d.d.errorf("decodeNaked: Unrecognized special value 0x%x", d.vs)
		}
	case bincVdSmallInt:
		n.v = valueTypeUint
		n.u = uint64(int8(d.vs)) + 1 // int8(d.vs) + 1
	case bincVdPosInt:
		n.v = valueTypeUint
		n.u = d.decUint()
	case bincVdNegInt:
		n.v = valueTypeInt
		n.i = -(int64(d.decUint()))
	case bincVdFloat:
		n.v = valueTypeFloat
		n.f = d.decFloat()
	case bincVdSymbol:
		n.v = valueTypeSymbol
		n.s = d.DecodeString()
	case bincVdString:
		n.v = valueTypeString
		n.s = d.DecodeString()
	case bincVdByteArray:
		n.v = valueTypeBytes
		n.l = d.DecodeBytes(nil, false, false)
	case bincVdTimestamp:
		n.v = valueTypeTimestamp
		tt, err := decodeTime(d.r.readx(int(d.vs)))
		if err != nil {
			panic(err)
		}
		n.t = tt
	case bincVdCustomExt:
		n.v = valueTypeExt
		l := d.decLen()
		n.u = uint64(d.r.readn1())
		n.l = d.r.readx(l)
	case bincVdArray:
		n.v = valueTypeArray
		decodeFurther = true
	case bincVdMap:
		n.v = valueTypeMap
		decodeFurther = true
	default:
		d.d.errorf("decodeNaked: Unrecognized d.vd: 0x%x", d.vd)
	}

	if !decodeFurther {
		d.bdRead = false
	}
	if n.v == valueTypeUint && d.h.SignedInteger {
		n.v = valueTypeInt
		n.i = int64(n.u)
	}
	return
}

//------------------------------------

//BincHandle is a Handle for the Binc Schema-Free Encoding Format
//defined at https://github.com/ugorji/binc .
//
//BincHandle currently supports all Binc features with the following EXCEPTIONS:
//  - only integers up to 64 bits of precision are supported.
//    big integers are unsupported.
//  - Only IEEE 754 binary32 and binary64 floats are supported (ie Go float32 and float64 types).
//    extended precision and decimal IEEE 754 floats are unsupported.
//  - Only UTF-8 strings supported.
//    Unicode_Other Binc types (UTF16, UTF32) are currently unsupported.
//
//Note that these EXCEPTIONS are temporary and full support is possible and may happen soon.
type BincHandle struct {
	BasicHandle
	binaryEncodingType
}

func (h *BincHandle) SetBytesExt(rt reflect.Type, tag uint64, ext BytesExt) (err error) {
	return h.SetExt(rt, tag, &setExtWrapper{b: ext})
}

func (h *BincHandle) newEncDriver(e *Encoder) encDriver {
	return &bincEncDriver{e: e, w: e.w}
}

func (h *BincHandle) newDecDriver(d *Decoder) decDriver {
	return &bincDecDriver{d: d, r: d.r, h: h, br: d.bytes}
}

func (e *bincEncDriver) reset() {
	e.w = e.e.w
	e.s = 0
	e.m = nil
}

func (d *bincDecDriver) reset() {
	d.r = d.d.r
	d.s = nil
	d.bd, d.bdRead, d.vd, d.vs = 0, false, 0, 0
}

var _ decDriver = (*bincDecDriver)(nil)
var _ encDriver = (*bincEncDriver)(nil)
