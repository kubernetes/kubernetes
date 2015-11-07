// //+build ignore

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

// ************************************************************
// DO NOT EDIT.
// THIS FILE IS AUTO-GENERATED from gen-helper.go.tmpl
// ************************************************************

package codec

import (
	"encoding"
	"reflect"
)

// This file is used to generate helper code for codecgen.
// The values here i.e. genHelper(En|De)coder are not to be used directly by
// library users. They WILL change continously and without notice.
//
// To help enforce this, we create an unexported type with exported members.
// The only way to get the type is via the one exported type that we control (somewhat).
//
// When static codecs are created for types, they will use this value
// to perform encoding or decoding of primitives or known slice or map types.

// GenHelperEncoder is exported so that it can be used externally by codecgen.
// Library users: DO NOT USE IT DIRECTLY. IT WILL CHANGE CONTINOUSLY WITHOUT NOTICE.
func GenHelperEncoder(e *Encoder) (genHelperEncoder, encDriver) {
	return genHelperEncoder{e: e}, e.e
}

// GenHelperDecoder is exported so that it can be used externally by codecgen.
// Library users: DO NOT USE IT DIRECTLY. IT WILL CHANGE CONTINOUSLY WITHOUT NOTICE.
func GenHelperDecoder(d *Decoder) (genHelperDecoder, decDriver) {
	return genHelperDecoder{d: d}, d.d
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
type genHelperEncoder struct {
	e *Encoder
	F fastpathT
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
type genHelperDecoder struct {
	d *Decoder
	F fastpathT
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncBasicHandle() *BasicHandle {
	return f.e.h
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncBinary() bool {
	return f.e.be // f.e.hh.isBinaryEncoding()
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncFallback(iv interface{}) {
	// println(">>>>>>>>> EncFallback")
	f.e.encodeI(iv, false, false)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncTextMarshal(iv encoding.TextMarshaler) {
	bs, fnerr := iv.MarshalText()
	f.e.marshal(bs, fnerr, false, c_UTF8)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncJSONMarshal(iv jsonMarshaler) {
	bs, fnerr := iv.MarshalJSON()
	f.e.marshal(bs, fnerr, true, c_UTF8)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncBinaryMarshal(iv encoding.BinaryMarshaler) {
	bs, fnerr := iv.MarshalBinary()
	f.e.marshal(bs, fnerr, false, c_RAW)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) TimeRtidIfBinc() uintptr {
	if _, ok := f.e.hh.(*BincHandle); ok {
		return timeTypId
	}
	return 0
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) IsJSONHandle() bool {
	return f.e.js
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) HasExtensions() bool {
	return len(f.e.h.extHandle) != 0
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperEncoder) EncExt(v interface{}) (r bool) {
	rt := reflect.TypeOf(v)
	if rt.Kind() == reflect.Ptr {
		rt = rt.Elem()
	}
	rtid := reflect.ValueOf(rt).Pointer()
	if xfFn := f.e.h.getExt(rtid); xfFn != nil {
		f.e.e.EncodeExt(v, xfFn.tag, xfFn.ext, f.e)
		return true
	}
	return false
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecBasicHandle() *BasicHandle {
	return f.d.h
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecBinary() bool {
	return f.d.be // f.d.hh.isBinaryEncoding()
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecSwallow() {
	f.d.swallow()
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecScratchBuffer() []byte {
	return f.d.b[:]
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecFallback(iv interface{}, chkPtr bool) {
	// println(">>>>>>>>> DecFallback")
	f.d.decodeI(iv, chkPtr, false, false, false)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecSliceHelperStart() (decSliceHelper, int) {
	return f.d.decSliceHelperStart()
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecStructFieldNotFound(index int, name string) {
	f.d.structFieldNotFound(index, name)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecArrayCannotExpand(sliceLen, streamLen int) {
	f.d.arrayCannotExpand(sliceLen, streamLen)
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecTextUnmarshal(tm encoding.TextUnmarshaler) {
	fnerr := tm.UnmarshalText(f.d.d.DecodeBytes(f.d.b[:], true, true))
	if fnerr != nil {
		panic(fnerr)
	}
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecJSONUnmarshal(tm jsonUnmarshaler) {
	// bs := f.dd.DecodeBytes(f.d.b[:], true, true)
	f.d.r.track()
	f.d.swallow()
	bs := f.d.r.stopTrack()
	// fmt.Printf(">>>>>> CODECGEN JSON: %s\n", bs)
	fnerr := tm.UnmarshalJSON(bs)
	if fnerr != nil {
		panic(fnerr)
	}
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecBinaryUnmarshal(bm encoding.BinaryUnmarshaler) {
	fnerr := bm.UnmarshalBinary(f.d.d.DecodeBytes(nil, false, true))
	if fnerr != nil {
		panic(fnerr)
	}
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) TimeRtidIfBinc() uintptr {
	if _, ok := f.d.hh.(*BincHandle); ok {
		return timeTypId
	}
	return 0
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) IsJSONHandle() bool {
	return f.d.js
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) HasExtensions() bool {
	return len(f.d.h.extHandle) != 0
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecExt(v interface{}) (r bool) {
	rt := reflect.TypeOf(v).Elem()
	rtid := reflect.ValueOf(rt).Pointer()
	if xfFn := f.d.h.getExt(rtid); xfFn != nil {
		f.d.d.DecodeExt(v, xfFn.tag, xfFn.ext)
		return true
	}
	return false
}

// FOR USE BY CODECGEN ONLY. IT *WILL* CHANGE WITHOUT NOTICE. *DO NOT USE*
func (f genHelperDecoder) DecInferLen(clen, maxlen, unit int) (rvlen int, truncated bool) {
	return decInferLen(clen, maxlen, unit)
}
