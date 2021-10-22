// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ptypes

import (
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"

	descriptorpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
	anypb "github.com/golang/protobuf/ptypes/any"
)

func TestMarshalUnmarshal(t *testing.T) {
	orig := &anypb.Any{Value: []byte("test")}

	packed, err := MarshalAny(orig)
	if err != nil {
		t.Errorf("MarshalAny(%+v): got: _, %v exp: _, nil", orig, err)
	}

	unpacked := &anypb.Any{}
	err = UnmarshalAny(packed, unpacked)
	if err != nil || !proto.Equal(unpacked, orig) {
		t.Errorf("got: %v, %+v; want nil, %+v", err, unpacked, orig)
	}
}

func TestIs(t *testing.T) {
	a, err := MarshalAny(&descriptorpb.FileDescriptorProto{})
	if err != nil {
		t.Fatal(err)
	}
	if Is(a, &descriptorpb.DescriptorProto{}) {
		// No spurious match for message names of different length.
		t.Error("FileDescriptorProto is not a DescriptorProto, but Is says it is")
	}
	if Is(a, &descriptorpb.EnumDescriptorProto{}) {
		// No spurious match for message names of equal length.
		t.Error("FileDescriptorProto is not an EnumDescriptorProto, but Is says it is")
	}
	if !Is(a, &descriptorpb.FileDescriptorProto{}) {
		t.Error("FileDescriptorProto is indeed a FileDescriptorProto, but Is says it is not")
	}
}

func TestIsDifferentUrlPrefixes(t *testing.T) {
	m := &descriptorpb.FileDescriptorProto{}
	a := &anypb.Any{TypeUrl: "foo/bar/" + proto.MessageName(m)}
	if !Is(a, m) {
		t.Errorf("message with type url %q didn't satisfy Is for type %q", a.TypeUrl, proto.MessageName(m))
	}
}

func TestIsCornerCases(t *testing.T) {
	m := &descriptorpb.FileDescriptorProto{}
	if Is(nil, m) {
		t.Errorf("message with nil type url incorrectly claimed to be %q", proto.MessageName(m))
	}
	noPrefix := &anypb.Any{TypeUrl: proto.MessageName(m)}
	if !Is(noPrefix, m) {
		t.Errorf("message with type url %q didn't satisfy Is for type %q", noPrefix.TypeUrl, proto.MessageName(m))
	}
	shortPrefix := &anypb.Any{TypeUrl: "/" + proto.MessageName(m)}
	if !Is(shortPrefix, m) {
		t.Errorf("message with type url %q didn't satisfy Is for type %q", shortPrefix.TypeUrl, proto.MessageName(m))
	}
}

func TestUnmarshalDynamic(t *testing.T) {
	want := &descriptorpb.FileDescriptorProto{Name: proto.String("foo")}
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
	want := &descriptorpb.FileDescriptorProto{}
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

func TestEmptyCornerCases(t *testing.T) {
	_, err := Empty(nil)
	if err == nil {
		t.Error("expected Empty for nil to fail")
	}
	want := &descriptorpb.FileDescriptorProto{}
	noPrefix := &anypb.Any{TypeUrl: proto.MessageName(want)}
	got, err := Empty(noPrefix)
	if err != nil {
		t.Errorf("Empty for any type %q failed: %s", noPrefix.TypeUrl, err)
	}
	if !proto.Equal(got, want) {
		t.Errorf("Empty for any type %q differs, got %q, want %q", noPrefix.TypeUrl, got, want)
	}
	shortPrefix := &anypb.Any{TypeUrl: "/" + proto.MessageName(want)}
	got, err = Empty(shortPrefix)
	if err != nil {
		t.Errorf("Empty for any type %q failed: %s", shortPrefix.TypeUrl, err)
	}
	if !proto.Equal(got, want) {
		t.Errorf("Empty for any type %q differs, got %q, want %q", shortPrefix.TypeUrl, got, want)
	}
}

func TestAnyReflect(t *testing.T) {
	want := &descriptorpb.FileDescriptorProto{Name: proto.String("foo")}
	a, err := MarshalAny(want)
	if err != nil {
		t.Fatal(err)
	}
	var got DynamicAny
	if err := UnmarshalAny(a, &got); err != nil {
		t.Fatal(err)
	}
	wantName := want.ProtoReflect().Descriptor().FullName()
	gotName := got.ProtoReflect().Descriptor().FullName()
	if gotName != wantName {
		t.Errorf("name mismatch: got %v, want %v", gotName, wantName)
	}
	wantType := reflect.TypeOf(got)
	gotType := reflect.TypeOf(got.ProtoReflect().Interface())
	if gotType != wantType {
		t.Errorf("ProtoReflect().Interface() round-trip type mismatch: got %v, want %v", gotType, wantType)
	}
	gotType = reflect.TypeOf(got.ProtoReflect().New().Interface())
	if gotType != wantType {
		t.Errorf("ProtoReflect().New().Interface() type mismatch: got %v, want %v", gotType, wantType)
	}
	gotType = reflect.TypeOf(got.ProtoReflect().Type().New().Interface())
	if gotType != wantType {
		t.Errorf("ProtoReflect().Type().New().Interface() type mismatch: got %v, want %v", gotType, wantType)
	}
	gotType = reflect.TypeOf(got.ProtoReflect().Type().Zero().Interface())
	if gotType != wantType {
		t.Errorf("ProtoReflect().Type().Zero().Interface() type mismatch: got %v, want %v", gotType, wantType)
	}
}
