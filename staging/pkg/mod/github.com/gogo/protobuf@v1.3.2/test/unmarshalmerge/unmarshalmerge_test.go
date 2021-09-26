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

package unmarshalmerge

import (
	"github.com/gogo/protobuf/proto"
	math_rand "math/rand"
	"testing"
	"time"
)

func TestUnmarshalMerge(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedBig(popr, true)
	if p.GetSub() == nil {
		p.Sub = &Sub{SubNumber: proto.Int64(12345)}
	}
	data, err := proto.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	s := &Sub{}
	b := &Big{
		Sub: s,
	}
	err = proto.UnmarshalMerge(data, b)
	if err != nil {
		t.Fatal(err)
	}
	if s.GetSubNumber() != p.GetSub().GetSubNumber() {
		t.Fatalf("should have unmarshaled subnumber into sub")
	}
}

func TestUnsafeUnmarshalMerge(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedBigUnsafe(popr, true)
	if p.GetSub() == nil {
		p.Sub = &Sub{SubNumber: proto.Int64(12345)}
	}
	data, err := proto.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	s := &Sub{}
	b := &BigUnsafe{
		Sub: s,
	}
	err = proto.UnmarshalMerge(data, b)
	if err != nil {
		t.Fatal(err)
	}

	if s.GetSubNumber() != p.GetSub().GetSubNumber() {
		t.Fatalf("should have unmarshaled subnumber into sub")
	}
}

func TestInt64Merge(t *testing.T) {
	popr := math_rand.New(math_rand.NewSource(time.Now().UnixNano()))
	p := NewPopulatedIntMerge(popr, true)
	p2 := NewPopulatedIntMerge(popr, true)
	data, err := proto.Marshal(p2)
	if err != nil {
		t.Fatal(err)
	}
	if err := proto.UnmarshalMerge(data, p); err != nil {
		t.Fatal(err)
	}
	if !p.Equal(p2) {
		t.Fatalf("exptected %#v but got %#v", p2, p)
	}
}
