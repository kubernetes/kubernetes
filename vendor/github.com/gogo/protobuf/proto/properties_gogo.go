// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto

import (
	"fmt"
	"os"
	"reflect"
)

func (p *Properties) setCustomEncAndDec(typ reflect.Type) {
	p.ctype = typ
	if p.Repeated {
		p.enc = (*Buffer).enc_custom_slice_bytes
		p.dec = (*Buffer).dec_custom_slice_bytes
		p.size = size_custom_slice_bytes
	} else if typ.Kind() == reflect.Ptr {
		p.enc = (*Buffer).enc_custom_bytes
		p.dec = (*Buffer).dec_custom_bytes
		p.size = size_custom_bytes
	} else {
		p.enc = (*Buffer).enc_custom_ref_bytes
		p.dec = (*Buffer).dec_custom_ref_bytes
		p.size = size_custom_ref_bytes
	}
}

func (p *Properties) setDurationEncAndDec(typ reflect.Type) {
	if p.Repeated {
		if typ.Elem().Kind() == reflect.Ptr {
			p.enc = (*Buffer).enc_slice_duration
			p.dec = (*Buffer).dec_slice_duration
			p.size = size_slice_duration
		} else {
			p.enc = (*Buffer).enc_slice_ref_duration
			p.dec = (*Buffer).dec_slice_ref_duration
			p.size = size_slice_ref_duration
		}
	} else if typ.Kind() == reflect.Ptr {
		p.enc = (*Buffer).enc_duration
		p.dec = (*Buffer).dec_duration
		p.size = size_duration
	} else {
		p.enc = (*Buffer).enc_ref_duration
		p.dec = (*Buffer).dec_ref_duration
		p.size = size_ref_duration
	}
}

func (p *Properties) setTimeEncAndDec(typ reflect.Type) {
	if p.Repeated {
		if typ.Elem().Kind() == reflect.Ptr {
			p.enc = (*Buffer).enc_slice_time
			p.dec = (*Buffer).dec_slice_time
			p.size = size_slice_time
		} else {
			p.enc = (*Buffer).enc_slice_ref_time
			p.dec = (*Buffer).dec_slice_ref_time
			p.size = size_slice_ref_time
		}
	} else if typ.Kind() == reflect.Ptr {
		p.enc = (*Buffer).enc_time
		p.dec = (*Buffer).dec_time
		p.size = size_time
	} else {
		p.enc = (*Buffer).enc_ref_time
		p.dec = (*Buffer).dec_ref_time
		p.size = size_ref_time
	}

}

func (p *Properties) setSliceOfNonPointerStructs(typ reflect.Type) {
	t2 := typ.Elem()
	p.sstype = typ
	p.stype = t2
	p.isMarshaler = isMarshaler(t2)
	p.isUnmarshaler = isUnmarshaler(t2)
	p.enc = (*Buffer).enc_slice_ref_struct_message
	p.dec = (*Buffer).dec_slice_ref_struct_message
	p.size = size_slice_ref_struct_message
	if p.Wire != "bytes" {
		fmt.Fprintf(os.Stderr, "proto: no ptr oenc for %T -> %T \n", typ, t2)
	}
}
