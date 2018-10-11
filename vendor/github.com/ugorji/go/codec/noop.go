// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// +build ignore

package codec

import (
	"math/rand"
	"time"
)

// NoopHandle returns a no-op handle. It basically does nothing.
// It is only useful for benchmarking, as it gives an idea of the
// overhead from the codec framework.
//
// LIBRARY USERS: *** DO NOT USE ***
func NoopHandle(slen int) *noopHandle {
	h := noopHandle{}
	h.rand = rand.New(rand.NewSource(time.Now().UnixNano()))
	h.B = make([][]byte, slen)
	h.S = make([]string, slen)
	for i := 0; i < len(h.S); i++ {
		b := make([]byte, i+1)
		for j := 0; j < len(b); j++ {
			b[j] = 'a' + byte(i)
		}
		h.B[i] = b
		h.S[i] = string(b)
	}
	return &h
}

// noopHandle does nothing.
// It is used to simulate the overhead of the codec framework.
type noopHandle struct {
	BasicHandle
	binaryEncodingType
	noopDrv // noopDrv is unexported here, so we can get a copy of it when needed.
}

type noopDrv struct {
	d    *Decoder
	e    *Encoder
	i    int
	S    []string
	B    [][]byte
	mks  []bool    // stack. if map (true), else if array (false)
	mk   bool      // top of stack. what container are we on? map or array?
	ct   valueType // last response for IsContainerType.
	cb   int       // counter for ContainerType
	rand *rand.Rand
}

func (h *noopDrv) r(v int) int { return h.rand.Intn(v) }
func (h *noopDrv) m(v int) int { h.i++; return h.i % v }

func (h *noopDrv) newEncDriver(e *Encoder) encDriver { h.e = e; return h }
func (h *noopDrv) newDecDriver(d *Decoder) decDriver { h.d = d; return h }

func (h *noopDrv) reset()       {}
func (h *noopDrv) uncacheRead() {}

// --- encDriver

// stack functions (for map and array)
func (h *noopDrv) start(b bool) {
	// println("start", len(h.mks)+1)
	h.mks = append(h.mks, b)
	h.mk = b
}
func (h *noopDrv) end() {
	// println("end: ", len(h.mks)-1)
	h.mks = h.mks[:len(h.mks)-1]
	if len(h.mks) > 0 {
		h.mk = h.mks[len(h.mks)-1]
	} else {
		h.mk = false
	}
}

func (h *noopDrv) EncodeBuiltin(rt uintptr, v interface{}) {}
func (h *noopDrv) EncodeNil()                              {}
func (h *noopDrv) EncodeInt(i int64)                       {}
func (h *noopDrv) EncodeUint(i uint64)                     {}
func (h *noopDrv) EncodeBool(b bool)                       {}
func (h *noopDrv) EncodeFloat32(f float32)                 {}
func (h *noopDrv) EncodeFloat64(f float64)                 {}
func (h *noopDrv) EncodeRawExt(re *RawExt, e *Encoder)     {}
func (h *noopDrv) EncodeArrayStart(length int)             { h.start(true) }
func (h *noopDrv) EncodeMapStart(length int)               { h.start(false) }
func (h *noopDrv) EncodeEnd()                              { h.end() }

func (h *noopDrv) EncodeString(c charEncoding, v string)      {}
func (h *noopDrv) EncodeSymbol(v string)                      {}
func (h *noopDrv) EncodeStringBytes(c charEncoding, v []byte) {}

func (h *noopDrv) EncodeExt(rv interface{}, xtag uint64, ext Ext, e *Encoder) {}

// ---- decDriver
func (h *noopDrv) initReadNext()                              {}
func (h *noopDrv) CheckBreak() bool                           { return false }
func (h *noopDrv) IsBuiltinType(rt uintptr) bool              { return false }
func (h *noopDrv) DecodeBuiltin(rt uintptr, v interface{})    {}
func (h *noopDrv) DecodeInt(bitsize uint8) (i int64)          { return int64(h.m(15)) }
func (h *noopDrv) DecodeUint(bitsize uint8) (ui uint64)       { return uint64(h.m(35)) }
func (h *noopDrv) DecodeFloat(chkOverflow32 bool) (f float64) { return float64(h.m(95)) }
func (h *noopDrv) DecodeBool() (b bool)                       { return h.m(2) == 0 }
func (h *noopDrv) DecodeString() (s string)                   { return h.S[h.m(8)] }
func (h *noopDrv) DecodeStringAsBytes() []byte                { return h.DecodeBytes(nil, true) }

func (h *noopDrv) DecodeBytes(bs []byte, zerocopy bool) []byte { return h.B[h.m(len(h.B))] }

func (h *noopDrv) ReadEnd() { h.end() }

// toggle map/slice
func (h *noopDrv) ReadMapStart() int   { h.start(true); return h.m(10) }
func (h *noopDrv) ReadArrayStart() int { h.start(false); return h.m(10) }

func (h *noopDrv) ContainerType() (vt valueType) {
	// return h.m(2) == 0
	// handle kStruct, which will bomb is it calls this and doesn't get back a map or array.
	// consequently, if the return value is not map or array, reset it to one of them based on h.m(7) % 2
	// for kstruct: at least one out of every 2 times, return one of valueTypeMap or Array (else kstruct bombs)
	// however, every 10th time it is called, we just return something else.
	var vals = [...]valueType{valueTypeArray, valueTypeMap}
	//  ------------ TAKE ------------
	// if h.cb%2 == 0 {
	// 	if h.ct == valueTypeMap || h.ct == valueTypeArray {
	// 	} else {
	// 		h.ct = vals[h.m(2)]
	// 	}
	// } else if h.cb%5 == 0 {
	// 	h.ct = valueType(h.m(8))
	// } else {
	// 	h.ct = vals[h.m(2)]
	// }
	//  ------------ TAKE ------------
	// if h.cb%16 == 0 {
	// 	h.ct = valueType(h.cb % 8)
	// } else {
	// 	h.ct = vals[h.cb%2]
	// }
	h.ct = vals[h.cb%2]
	h.cb++
	return h.ct

	// if h.ct == valueTypeNil || h.ct == valueTypeString || h.ct == valueTypeBytes {
	// 	return h.ct
	// }
	// return valueTypeUnset
	// TODO: may need to tweak this so it works.
	// if h.ct == valueTypeMap && vt == valueTypeArray || h.ct == valueTypeArray && vt == valueTypeMap {
	// 	h.cb = !h.cb
	// 	h.ct = vt
	// 	return h.cb
	// }
	// // go in a loop and check it.
	// h.ct = vt
	// h.cb = h.m(7) == 0
	// return h.cb
}
func (h *noopDrv) TryDecodeAsNil() bool {
	if h.mk {
		return false
	} else {
		return h.m(8) == 0
	}
}
func (h *noopDrv) DecodeExt(rv interface{}, xtag uint64, ext Ext) uint64 {
	return 0
}

func (h *noopDrv) DecodeNaked() {
	// use h.r (random) not h.m() because h.m() could cause the same value to be given.
	var sk int
	if h.mk {
		// if mapkey, do not support values of nil OR bytes, array, map or rawext
		sk = h.r(7) + 1
	} else {
		sk = h.r(12)
	}
	n := &h.d.n
	switch sk {
	case 0:
		n.v = valueTypeNil
	case 1:
		n.v, n.b = valueTypeBool, false
	case 2:
		n.v, n.b = valueTypeBool, true
	case 3:
		n.v, n.i = valueTypeInt, h.DecodeInt(64)
	case 4:
		n.v, n.u = valueTypeUint, h.DecodeUint(64)
	case 5:
		n.v, n.f = valueTypeFloat, h.DecodeFloat(true)
	case 6:
		n.v, n.f = valueTypeFloat, h.DecodeFloat(false)
	case 7:
		n.v, n.s = valueTypeString, h.DecodeString()
	case 8:
		n.v, n.l = valueTypeBytes, h.B[h.m(len(h.B))]
	case 9:
		n.v = valueTypeArray
	case 10:
		n.v = valueTypeMap
	default:
		n.v = valueTypeExt
		n.u = h.DecodeUint(64)
		n.l = h.B[h.m(len(h.B))]
	}
	h.ct = n.v
	return
}
