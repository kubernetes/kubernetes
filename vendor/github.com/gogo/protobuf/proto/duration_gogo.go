// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2016, The GoGo Authors. All rights reserved.
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
	"reflect"
	"time"
)

var durationType = reflect.TypeOf((*time.Duration)(nil)).Elem()

type duration struct {
	Seconds int64 `protobuf:"varint,1,opt,name=seconds,proto3" json:"seconds,omitempty"`
	Nanos   int32 `protobuf:"varint,2,opt,name=nanos,proto3" json:"nanos,omitempty"`
}

func (m *duration) Reset()       { *m = duration{} }
func (*duration) ProtoMessage()  {}
func (*duration) String() string { return "duration<string>" }

func init() {
	RegisterType((*duration)(nil), "gogo.protobuf.proto.duration")
}

func (o *Buffer) decDuration() (time.Duration, error) {
	b, err := o.DecodeRawBytes(true)
	if err != nil {
		return 0, err
	}
	dproto := &duration{}
	if err := Unmarshal(b, dproto); err != nil {
		return 0, err
	}
	return durationFromProto(dproto)
}

func (o *Buffer) dec_duration(p *Properties, base structPointer) error {
	d, err := o.decDuration()
	if err != nil {
		return err
	}
	word64_Set(structPointer_Word64(base, p.field), o, uint64(d))
	return nil
}

func (o *Buffer) dec_ref_duration(p *Properties, base structPointer) error {
	d, err := o.decDuration()
	if err != nil {
		return err
	}
	word64Val_Set(structPointer_Word64Val(base, p.field), o, uint64(d))
	return nil
}

func (o *Buffer) dec_slice_duration(p *Properties, base structPointer) error {
	d, err := o.decDuration()
	if err != nil {
		return err
	}
	newBas := appendStructPointer(base, p.field, reflect.SliceOf(reflect.PtrTo(durationType)))
	var zero field
	setPtrCustomType(newBas, zero, &d)
	return nil
}

func (o *Buffer) dec_slice_ref_duration(p *Properties, base structPointer) error {
	d, err := o.decDuration()
	if err != nil {
		return err
	}
	structPointer_Word64Slice(base, p.field).Append(uint64(d))
	return nil
}

func size_duration(p *Properties, base structPointer) (n int) {
	structp := structPointer_GetStructPointer(base, p.field)
	if structPointer_IsNil(structp) {
		return 0
	}
	dur := structPointer_Interface(structp, durationType).(*time.Duration)
	d := durationProto(*dur)
	size := Size(d)
	return size + sizeVarint(uint64(size)) + len(p.tagcode)
}

func (o *Buffer) enc_duration(p *Properties, base structPointer) error {
	structp := structPointer_GetStructPointer(base, p.field)
	if structPointer_IsNil(structp) {
		return ErrNil
	}
	dur := structPointer_Interface(structp, durationType).(*time.Duration)
	d := durationProto(*dur)
	data, err := Marshal(d)
	if err != nil {
		return err
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeRawBytes(data)
	return nil
}

func size_ref_duration(p *Properties, base structPointer) (n int) {
	dur := structPointer_InterfaceAt(base, p.field, durationType).(*time.Duration)
	d := durationProto(*dur)
	size := Size(d)
	return size + sizeVarint(uint64(size)) + len(p.tagcode)
}

func (o *Buffer) enc_ref_duration(p *Properties, base structPointer) error {
	dur := structPointer_InterfaceAt(base, p.field, durationType).(*time.Duration)
	d := durationProto(*dur)
	data, err := Marshal(d)
	if err != nil {
		return err
	}
	o.buf = append(o.buf, p.tagcode...)
	o.EncodeRawBytes(data)
	return nil
}

func size_slice_duration(p *Properties, base structPointer) (n int) {
	pdurs := structPointer_InterfaceAt(base, p.field, reflect.SliceOf(reflect.PtrTo(durationType))).(*[]*time.Duration)
	durs := *pdurs
	for i := 0; i < len(durs); i++ {
		if durs[i] == nil {
			return 0
		}
		dproto := durationProto(*durs[i])
		size := Size(dproto)
		n += len(p.tagcode) + size + sizeVarint(uint64(size))
	}
	return n
}

func (o *Buffer) enc_slice_duration(p *Properties, base structPointer) error {
	pdurs := structPointer_InterfaceAt(base, p.field, reflect.SliceOf(reflect.PtrTo(durationType))).(*[]*time.Duration)
	durs := *pdurs
	for i := 0; i < len(durs); i++ {
		if durs[i] == nil {
			return errRepeatedHasNil
		}
		dproto := durationProto(*durs[i])
		data, err := Marshal(dproto)
		if err != nil {
			return err
		}
		o.buf = append(o.buf, p.tagcode...)
		o.EncodeRawBytes(data)
	}
	return nil
}

func size_slice_ref_duration(p *Properties, base structPointer) (n int) {
	pdurs := structPointer_InterfaceAt(base, p.field, reflect.SliceOf(durationType)).(*[]time.Duration)
	durs := *pdurs
	for i := 0; i < len(durs); i++ {
		dproto := durationProto(durs[i])
		size := Size(dproto)
		n += len(p.tagcode) + size + sizeVarint(uint64(size))
	}
	return n
}

func (o *Buffer) enc_slice_ref_duration(p *Properties, base structPointer) error {
	pdurs := structPointer_InterfaceAt(base, p.field, reflect.SliceOf(durationType)).(*[]time.Duration)
	durs := *pdurs
	for i := 0; i < len(durs); i++ {
		dproto := durationProto(durs[i])
		data, err := Marshal(dproto)
		if err != nil {
			return err
		}
		o.buf = append(o.buf, p.tagcode...)
		o.EncodeRawBytes(data)
	}
	return nil
}
