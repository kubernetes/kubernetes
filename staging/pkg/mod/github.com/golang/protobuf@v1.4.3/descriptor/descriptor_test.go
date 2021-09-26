// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package descriptor

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/reflect/protoreflect"

	descpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
)

func TestEnumDescriptor(t *testing.T) {
	tests := []struct {
		enum protoreflect.Enum
		idxs []int
		name string
	}{{
		enum: descpb.FieldDescriptorProto_Type(0),
		idxs: []int{
			new(descpb.FieldDescriptorProto).ProtoReflect().Descriptor().Index(),
			new(descpb.FieldDescriptorProto_Type).Descriptor().Index(),
		},
		name: "Type",
	}, {
		enum: descpb.FieldOptions_CType(0),
		idxs: []int{
			new(descpb.FieldOptions).ProtoReflect().Descriptor().Index(),
			new(descpb.FieldOptions_CType).Descriptor().Index(),
		},
		name: "CType",
	}}

	for _, tt := range tests {
		e := struct{ protoreflect.Enum }{tt.enum} // v2-only enum

		_, idxs := EnumRawDescriptor(e)
		if diff := cmp.Diff(tt.idxs, idxs); diff != "" {
			t.Errorf("path index mismatch (-want +got):\n%v", diff)
		}

		_, ed := EnumDescriptorProto(e)
		if ed.GetName() != tt.name {
			t.Errorf("mismatching enum name: got %v, want %v", ed.GetName(), tt.name)
		}
	}
}

func TestMessageDescriptor(t *testing.T) {
	tests := []struct {
		message protoreflect.ProtoMessage
		idxs    []int
		name    string
	}{{
		message: (*descpb.SourceCodeInfo_Location)(nil),
		idxs: []int{
			new(descpb.SourceCodeInfo).ProtoReflect().Descriptor().Index(),
			new(descpb.SourceCodeInfo_Location).ProtoReflect().Descriptor().Index(),
		},
		name: "Location",
	}, {
		message: (*descpb.FileDescriptorProto)(nil),
		idxs: []int{
			new(descpb.FileDescriptorProto).ProtoReflect().Descriptor().Index(),
		},
		name: "FileDescriptorProto",
	}}

	for _, tt := range tests {
		m := struct{ protoreflect.ProtoMessage }{tt.message} // v2-only message

		_, idxs := MessageRawDescriptor(m)
		if diff := cmp.Diff(tt.idxs, idxs); diff != "" {
			t.Errorf("path index mismatch (-want +got):\n%v", diff)
		}

		_, md := MessageDescriptorProto(m)
		if md.GetName() != tt.name {
			t.Errorf("mismatching message name: got %v, want %v", md.GetName(), tt.name)
		}
	}
}
