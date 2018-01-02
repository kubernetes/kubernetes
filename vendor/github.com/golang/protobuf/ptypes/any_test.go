// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2016 The Go Authors.  All rights reserved.
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

package ptypes

import (
	"testing"

	"github.com/golang/protobuf/proto"
	pb "github.com/golang/protobuf/protoc-gen-go/descriptor"
	"github.com/golang/protobuf/ptypes/any"
)

func TestMarshalUnmarshal(t *testing.T) {
	orig := &any.Any{Value: []byte("test")}

	packed, err := MarshalAny(orig)
	if err != nil {
		t.Errorf("MarshalAny(%+v): got: _, %v exp: _, nil", orig, err)
	}

	unpacked := &any.Any{}
	err = UnmarshalAny(packed, unpacked)
	if err != nil || !proto.Equal(unpacked, orig) {
		t.Errorf("got: %v, %+v; want nil, %+v", err, unpacked, orig)
	}
}

func TestIs(t *testing.T) {
	a, err := MarshalAny(&pb.FileDescriptorProto{})
	if err != nil {
		t.Fatal(err)
	}
	if Is(a, &pb.DescriptorProto{}) {
		t.Error("FileDescriptorProto is not a DescriptorProto, but Is says it is")
	}
	if !Is(a, &pb.FileDescriptorProto{}) {
		t.Error("FileDescriptorProto is indeed a FileDescriptorProto, but Is says it is not")
	}
}

func TestIsDifferentUrlPrefixes(t *testing.T) {
	m := &pb.FileDescriptorProto{}
	a := &any.Any{TypeUrl: "foo/bar/" + proto.MessageName(m)}
	if !Is(a, m) {
		t.Errorf("message with type url %q didn't satisfy Is for type %q", a.TypeUrl, proto.MessageName(m))
	}
}

func TestUnmarshalDynamic(t *testing.T) {
	want := &pb.FileDescriptorProto{Name: proto.String("foo")}
	a, err := MarshalAny(want)
	if err != nil {
		t.Fatal(err)
	}
	var got DynamicAny
	if err := UnmarshalAny(a, &got); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(got.Message, want) {
		t.Errorf("invalid result from UnmarshalAny, got %q want %q", got.Message, want)
	}
}

func TestEmpty(t *testing.T) {
	want := &pb.FileDescriptorProto{}
	a, err := MarshalAny(want)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Empty(a)
	if err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(got, want) {
		t.Errorf("unequal empty message, got %q, want %q", got, want)
	}

	// that's a valid type_url for a message which shouldn't be linked into this
	// test binary. We want an error.
	a.TypeUrl = "type.googleapis.com/google.protobuf.FieldMask"
	if _, err := Empty(a); err == nil {
		t.Errorf("got no error for an attempt to create a message of type %q, which shouldn't be linked in", a.TypeUrl)
	}
}
