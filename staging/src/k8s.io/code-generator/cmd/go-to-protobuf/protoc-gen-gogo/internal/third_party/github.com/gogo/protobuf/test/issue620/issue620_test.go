// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2019, The GoGo Authors. All rights reserved.
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

package issue620

import (
	"bytes"
	"testing"

	"github.com/gogo/protobuf/proto"
)

func TestEncodeShort(t *testing.T) {
	exp := []byte{8, 10, 6, 66, 97, 114, 49, 50, 51}
	s := "Bar123"
	b1 := &Bar{Field1: &s}
	bufProto := proto.NewBuffer(nil)

	b2 := &BarFast{Field1: &s}
	bufProtoFast := proto.NewBuffer(nil)

	encodeMessageCheck(t, b1, b2, bufProto, bufProtoFast, exp)
}

func TestEncodeLong(t *testing.T) {
	exp := []byte{8, 10, 6, 66, 97, 114, 49, 50, 51}
	s := "Bar123"
	b1 := &Bar{Field1: &s}
	bufProto := proto.NewBuffer(make([]byte, 0, 480))
	b2 := &BarFast{Field1: &s}
	bufProtoFast := proto.NewBuffer(make([]byte, 0, 480))

	encodeMessageCheck(t, b1, b2, bufProto, bufProtoFast, exp)
}

func TestEncodeDecode(t *testing.T) {
	s := "Bar123"
	bar := &BarFast{Field1: &s}
	bufProtoFast := proto.NewBuffer(make([]byte, 0, 480))
	err := bufProtoFast.EncodeMessage(bar)
	errCheck(t, err)
	dec := &BarFast{}
	err = bufProtoFast.DecodeMessage(dec)
	errCheck(t, err)
	if !dec.Equal(bar) {
		t.Errorf("Expect %+v got %+v after encode/decode", bar, dec)
	}
}

func encodeMessageCheck(t *testing.T, b1, b2 proto.Message, bufProto, bufProtoFast *proto.Buffer, exp []byte) {
	err := bufProto.EncodeMessage(b1)
	errCheck(t, err)
	err = bufProtoFast.EncodeMessage(b2)
	errCheck(t, err)
	checkEqual(t, exp, bufProto.Bytes())
	checkEqual(t, exp, bufProtoFast.Bytes())

	bufProto.Reset()
	bufProtoFast.Reset()
	expMore := make([]byte, 0, len(exp))
	copy(expMore, exp)
	for i := 0; i < 10; i++ {
		err = bufProto.EncodeMessage(b1)
		errCheck(t, err)
		err = bufProtoFast.EncodeMessage(b2)
		errCheck(t, err)
		expMore = append(expMore, exp...)
		checkEqual(t, expMore, bufProto.Bytes())
		checkEqual(t, expMore, bufProtoFast.Bytes())
	}

	bufProto.Reset()
	bufProtoFast.Reset()
	err = bufProto.EncodeMessage(b1)
	errCheck(t, err)
	err = bufProtoFast.EncodeMessage(b2)
	errCheck(t, err)
	checkEqual(t, exp, bufProto.Bytes())
	checkEqual(t, exp, bufProtoFast.Bytes())
}

func errCheck(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Error(err.Error())
	}
}

func checkEqual(t *testing.T, b1, b2 []byte) {
	t.Helper()
	if !bytes.Equal(b1, b2) {
		t.Errorf("%v != %v\n", b1, b2)
	}
}
