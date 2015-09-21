// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

import (
	"math/rand"
	"time"
)

// NoopHandle returns a no-op handle. It basically does nothing.
// It is only useful for benchmarking, as it gives an idea of the
// overhead from the codec framework.
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
	i    int
	S    []string
	B    [][]byte
	mk   bool      // are we about to read a map key?
	ct   valueType // last request for IsContainerType.
	cb   bool      // last response for IsContainerType.
	rand *rand.Rand
}

func (h *noopDrv) r(v int) int { return h.rand.Intn(v) }
func (h *noopDrv) m(v int) int { h.i++; return h.i % v }

func (h *noopDrv) newEncDriver(_ *Encoder) encDriver { return h }
func (h *noopDrv) newDecDriver(_ *Decoder) decDriver { return h }

// --- encDriver

func (h *noopDrv) EncodeBuiltin(rt uintptr, v interface{})    {}
func (h *noopDrv) EncodeNil()                                 {}
func (h *noopDrv) EncodeInt(i int64)                          {}
func (h *noopDrv) EncodeUint(i uint64)                        {}
func (h *noopDrv) EncodeBool(b bool)                          {}
func (h *noopDrv) EncodeFloat32(f float32)                    {}
func (h *noopDrv) EncodeFloat64(f float64)                    {}
func (h *noopDrv) EncodeRawExt(re *RawExt, e *Encoder)        {}
func (h *noopDrv) EncodeArrayStart(length int)                {}
func (h *noopDrv) EncodeArrayEnd()                            {}
func (h *noopDrv) EncodeArrayEntrySeparator()                 {}
func (h *noopDrv) EncodeMapStart(length int)                  {}
func (h *noopDrv) EncodeMapEnd()                              {}
func (h *noopDrv) EncodeMapEntrySeparator()                   {}
func (h *noopDrv) EncodeMapKVSeparator()                      {}
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

// func (h *noopDrv) DecodeStringAsBytes(bs []byte) []byte       { return h.DecodeBytes(bs) }

func (h *noopDrv) DecodeBytes(bs []byte, isstring, zerocopy bool) []byte { return h.B[h.m(len(h.B))] }

func (h *noopDrv) ReadMapEnd()              { h.mk = false }
func (h *noopDrv) ReadArrayEnd()            {}
func (h *noopDrv) ReadArrayEntrySeparator() {}
func (h *noopDrv) ReadMapEntrySeparator()   { h.mk = true }
func (h *noopDrv) ReadMapKVSeparator()      { h.mk = false }

// toggle map/slice
func (h *noopDrv) ReadMapStart() int   { h.mk = true; return h.m(10) }
func (h *noopDrv) ReadArrayStart() int { return h.m(10) }

func (h *noopDrv) IsContainerType(vt valueType) bool {
	// return h.m(2) == 0
	// handle kStruct
	if h.ct == valueTypeMap && vt == valueTypeArray || h.ct == valueTypeArray && vt == valueTypeMap {
		h.cb = !h.cb
		h.ct = vt
		return h.cb
	}
	// go in a loop and check it.
	h.ct = vt
	h.cb = h.m(7) == 0
	return h.cb
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

func (h *noopDrv) DecodeNaked() (v interface{}, vt valueType, decodeFurther bool) {
	// use h.r (random) not h.m() because h.m() could cause the same value to be given.
	var sk int
	if h.mk {
		// if mapkey, do not support values of nil OR bytes, array, map or rawext
		sk = h.r(7) + 1
	} else {
		sk = h.r(12)
	}
	switch sk {
	case 0:
		vt = valueTypeNil
	case 1:
		vt, v = valueTypeBool, false
	case 2:
		vt, v = valueTypeBool, true
	case 3:
		vt, v = valueTypeInt, h.DecodeInt(64)
	case 4:
		vt, v = valueTypeUint, h.DecodeUint(64)
	case 5:
		vt, v = valueTypeFloat, h.DecodeFloat(true)
	case 6:
		vt, v = valueTypeFloat, h.DecodeFloat(false)
	case 7:
		vt, v = valueTypeString, h.DecodeString()
	case 8:
		vt, v = valueTypeBytes, h.B[h.m(len(h.B))]
	case 9:
		vt, decodeFurther = valueTypeArray, true
	case 10:
		vt, decodeFurther = valueTypeMap, true
	default:
		vt, v = valueTypeExt, &RawExt{Tag: h.DecodeUint(64), Data: h.B[h.m(len(h.B))]}
	}
	h.ct = vt
	return
}
