// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
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
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
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

// +build go1.7

package proto_test

import (
	"strconv"
	"testing"

	"github.com/golang/protobuf/proto"
	tpb "github.com/golang/protobuf/proto/proto3_proto"
	"github.com/golang/protobuf/ptypes"
)

var (
	blackhole []byte
)

// BenchmarkAny creates increasingly large arbitrary Any messages.  The type is always the
// same.
func BenchmarkAny(b *testing.B) {
	data := make([]byte, 1<<20)
	quantum := 1 << 10
	for i := uint(0); i <= 10; i++ {
		b.Run(strconv.Itoa(quantum<<i), func(b *testing.B) {
			for k := 0; k < b.N; k++ {
				inner := &tpb.Message{
					Data: data[:quantum<<i],
				}
				outer, err := ptypes.MarshalAny(inner)
				if err != nil {
					b.Error("wrong encode", err)
				}
				raw, err := proto.Marshal(&tpb.Message{
					Anything: outer,
				})
				if err != nil {
					b.Error("wrong encode", err)
				}
				blackhole = raw
			}
		})
	}
}

// BenchmarkEmpy measures the overhead of doing the minimal possible encode.
func BenchmarkEmpy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		raw, err := proto.Marshal(&tpb.Message{})
		if err != nil {
			b.Error("wrong encode", err)
		}
		blackhole = raw
	}
}
