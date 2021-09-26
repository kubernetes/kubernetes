// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2018, The GoGo Authors. All rights reserved.
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

package issue449

import (
	"testing"

	"github.com/gogo/protobuf/proto"
)

func TestCodeGenMsgMarshalUnmarshal(t *testing.T) {
	src := &CodeGenMsg{
		Int64ReqPtr: proto.Int64(111),
		Int32OptPtr: proto.Int32(222),
		Int64Req:    333,
		Int32Opt:    444,
	}
	buf, err := proto.Marshal(src)
	if err != nil {
		t.Fatal(err)
	}
	dst := &CodeGenMsg{}
	if err := proto.Unmarshal(buf, dst); err != nil {
		t.Fatal(err)
	}
	if !src.Equal(dst) {
		t.Fatal("message is not equals")
	}
}

func TestNonCodeGenMsgMarshalUnmarshal(t *testing.T) {
	src := &NonCodeGenMsg{
		Int64ReqPtr: proto.Int64(111),
		Int32OptPtr: proto.Int32(222),
		Int64Req:    333,
		Int32Opt:    444,
	}
	buf, err := proto.Marshal(src)
	if err != nil {
		t.Fatal(err)
	}
	dst := &NonCodeGenMsg{}
	if err := proto.Unmarshal(buf, dst); err != nil {
		t.Fatal(err)
	}
	if !src.Equal(dst) {
		t.Fatal("message is not equals")
	}
}

func TestRequiredFieldCheck(t *testing.T) {
	tbl := []struct {
		In      proto.Message
		Success bool
	}{
		// Generated Code Message
		{
			// filled message
			In: &CodeGenMsg{
				Int64ReqPtr: proto.Int64(111),
				Int32OptPtr: proto.Int32(222),
				Int64Req:    333,
				Int32Opt:    444,
			},
			Success: true,
		},
		{
			// filled message (set zero value)
			In: &CodeGenMsg{
				Int64ReqPtr: proto.Int64(0),
				Int32OptPtr: proto.Int32(0),
				Int64Req:    0,
				Int32Opt:    0,
			},
			Success: true,
		},
		{
			// non filled message (Int64ReqPtr is not set)
			In: &CodeGenMsg{
				Int64Req: 333,
			},
			Success: false,
		},
		{
			// non filled message (Int64Req is not set, but can't inspect)
			In: &CodeGenMsg{
				Int64ReqPtr: proto.Int64(111),
			},
			Success: true,
		},

		// Non Generated Code Message
		{
			// filled message
			In: &NonCodeGenMsg{
				Int64ReqPtr: proto.Int64(111),
				Int32OptPtr: proto.Int32(222),
				Int64Req:    333,
				Int32Opt:    444,
			},
			Success: true,
		},
		{
			// filled message (set zero value)
			In: &NonCodeGenMsg{
				Int64ReqPtr: proto.Int64(0),
				Int32OptPtr: proto.Int32(0),
				Int64Req:    0,
				Int32Opt:    0,
			},
			Success: true,
		},
		{
			// non filled message (Int64ReqPtr is not set)
			In: &NonCodeGenMsg{
				Int64Req: 333,
			},
			Success: false,
		},
		{
			// non filled message (Int64Req is not set, but can't inspect)
			In: &NonCodeGenMsg{
				Int64ReqPtr: proto.Int64(111),
			},
			Success: true,
		},
	}
	for i, v := range tbl {
		_, err := proto.Marshal(v.In)
		switch v.Success {
		case true:
			if err != nil {
				t.Fatalf("[%d] failed to proto.Marshal(%v), %s", i, v.In, err)
			}
		case false:
			if err == nil {
				t.Fatalf("[%d] required field check is not working", i)
			}
			if _, ok := err.(*proto.RequiredNotSetError); !ok {
				t.Fatalf("[%d] failed to proto.Marshal(%v), %s", i, v.In, err)
			}
		}
	}
}
