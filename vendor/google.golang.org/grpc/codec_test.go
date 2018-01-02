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
	"bytes"
	"sync"
	"testing"

	"google.golang.org/grpc/test/codec_perf"
)

func marshalAndUnmarshal(t *testing.T, protoCodec Codec, expectedBody []byte) {
	p := &codec_perf.Buffer{}
	p.Body = expectedBody

	marshalledBytes, err := protoCodec.Marshal(p)
	if err != nil {
		t.Errorf("protoCodec.Marshal(_) returned an error")
	}

	if err := protoCodec.Unmarshal(marshalledBytes, p); err != nil {
		t.Errorf("protoCodec.Unmarshal(_) returned an error")
	}

	if bytes.Compare(p.GetBody(), expectedBody) != 0 {
		t.Errorf("Unexpected body; got %v; want %v", p.GetBody(), expectedBody)
	}
}

func TestBasicProtoCodecMarshalAndUnmarshal(t *testing.T) {
	marshalAndUnmarshal(t, protoCodec{}, []byte{1, 2, 3})
}

// Try to catch possible race conditions around use of pools
func TestConcurrentUsage(t *testing.T) {
	const (
		numGoRoutines   = 100
		numMarshUnmarsh = 1000
	)

	// small, arbitrary byte slices
	protoBodies := [][]byte{
		[]byte("one"),
		[]byte("two"),
		[]byte("three"),
		[]byte("four"),
		[]byte("five"),
	}

	var wg sync.WaitGroup
	codec := protoCodec{}

	for i := 0; i < numGoRoutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for k := 0; k < numMarshUnmarsh; k++ {
				marshalAndUnmarshal(t, codec, protoBodies[k%len(protoBodies)])
			}
		}()
	}

	wg.Wait()
}

// TestStaggeredMarshalAndUnmarshalUsingSamePool tries to catch potential errors in which slices get
// stomped on during reuse of a proto.Buffer.
func TestStaggeredMarshalAndUnmarshalUsingSamePool(t *testing.T) {
	codec1 := protoCodec{}
	codec2 := protoCodec{}

	expectedBody1 := []byte{1, 2, 3}
	expectedBody2 := []byte{4, 5, 6}

	proto1 := codec_perf.Buffer{Body: expectedBody1}
	proto2 := codec_perf.Buffer{Body: expectedBody2}

	var m1, m2 []byte
	var err error

	if m1, err = codec1.Marshal(&proto1); err != nil {
		t.Errorf("protoCodec.Marshal(%v) failed", proto1)
	}

	if m2, err = codec2.Marshal(&proto2); err != nil {
		t.Errorf("protoCodec.Marshal(%v) failed", proto2)
	}

	if err = codec1.Unmarshal(m1, &proto1); err != nil {
		t.Errorf("protoCodec.Unmarshal(%v) failed", m1)
	}

	if err = codec2.Unmarshal(m2, &proto2); err != nil {
		t.Errorf("protoCodec.Unmarshal(%v) failed", m2)
	}

	b1 := proto1.GetBody()
	b2 := proto2.GetBody()

	for i, v := range b1 {
		if expectedBody1[i] != v {
			t.Errorf("expected %v at index %v but got %v", i, expectedBody1[i], v)
		}
	}

	for i, v := range b2 {
		if expectedBody2[i] != v {
			t.Errorf("expected %v at index %v but got %v", i, expectedBody2[i], v)
		}
	}
}
