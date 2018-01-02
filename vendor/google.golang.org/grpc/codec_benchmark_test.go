// +build go1.7

/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package grpc

import (
	"fmt"
	"testing"

	"github.com/golang/protobuf/proto"
	"google.golang.org/grpc/test/codec_perf"
)

func setupBenchmarkProtoCodecInputs(b *testing.B, payloadBaseSize uint32) []proto.Message {
	payloadBase := make([]byte, payloadBaseSize)
	// arbitrary byte slices
	payloadSuffixes := [][]byte{
		[]byte("one"),
		[]byte("two"),
		[]byte("three"),
		[]byte("four"),
		[]byte("five"),
	}
	protoStructs := make([]proto.Message, 0)

	for _, p := range payloadSuffixes {
		ps := &codec_perf.Buffer{}
		ps.Body = append(payloadBase, p...)
		protoStructs = append(protoStructs, ps)
	}

	return protoStructs
}

// The possible use of certain protobuf APIs like the proto.Buffer API potentially involves caching
// on our side. This can add checks around memory allocations and possible contention.
// Example run: go test -v -run=^$ -bench=BenchmarkProtoCodec -benchmem
func BenchmarkProtoCodec(b *testing.B) {
	// range of message sizes
	payloadBaseSizes := make([]uint32, 0)
	for i := uint32(0); i <= 12; i += 4 {
		payloadBaseSizes = append(payloadBaseSizes, 1<<i)
	}
	// range of SetParallelism
	parallelisms := make([]uint32, 0)
	for i := uint32(0); i <= 16; i += 4 {
		parallelisms = append(parallelisms, 1<<i)
	}
	for _, s := range payloadBaseSizes {
		for _, p := range parallelisms {
			func(parallelism int, payloadBaseSize uint32) {
				protoStructs := setupBenchmarkProtoCodecInputs(b, payloadBaseSize)
				name := fmt.Sprintf("MinPayloadSize:%v/SetParallelism(%v)", payloadBaseSize, parallelism)
				b.Run(name, func(b *testing.B) {
					codec := &protoCodec{}
					b.SetParallelism(parallelism)
					b.RunParallel(func(pb *testing.PB) {
						benchmarkProtoCodec(codec, protoStructs, pb, b)
					})
				})
			}(int(p), s)
		}
	}
}

func benchmarkProtoCodec(codec *protoCodec, protoStructs []proto.Message, pb *testing.PB, b *testing.B) {
	counter := 0
	for pb.Next() {
		counter++
		ps := protoStructs[counter%len(protoStructs)]
		fastMarshalAndUnmarshal(codec, ps, b)
	}
}

func fastMarshalAndUnmarshal(protoCodec Codec, protoStruct proto.Message, b *testing.B) {
	marshaledBytes, err := protoCodec.Marshal(protoStruct)
	if err != nil {
		b.Errorf("protoCodec.Marshal(_) returned an error")
	}
	if err := protoCodec.Unmarshal(marshaledBytes, protoStruct); err != nil {
		b.Errorf("protoCodec.Unmarshal(_) returned an error")
	}
}
