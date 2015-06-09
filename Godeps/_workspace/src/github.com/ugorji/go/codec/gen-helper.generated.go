// //+build ignore

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

// ************************************************************
// DO NOT EDIT.
// THIS FILE IS AUTO-GENERATED from gen-helper.go.tmpl
// ************************************************************

package codec

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
