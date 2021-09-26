// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"reflect"
	"testing"

	"github.com/golang/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	descpb "github.com/golang/protobuf/protoc-gen-go/descriptor"
)

func TestRegistry(t *testing.T) {
	file := new(descpb.DescriptorProto).ProtoReflect().Descriptor().ParentFile()
	path := file.Path()
	pkg := file.Package()
	if got := proto.FileDescriptor(path); len(got) == 0 {
		t.Errorf("FileDescriptor(%q) = empty, want non-empty", path)
	}

	name := protoreflect.FullName(pkg + ".FieldDescriptorProto_Label")
	if got := proto.EnumValueMap(string(name)); len(got) == 0 {
		t.Errorf("EnumValueMap(%q) = empty, want non-empty", name)
	}

	msg := new(descpb.EnumDescriptorProto_EnumReservedRange)
	name = msg.ProtoReflect().Descriptor().FullName()
	wantType := reflect.TypeOf(msg)
	gotType := proto.MessageType(string(name))
	if gotType != wantType {
		t.Errorf("MessageType(%q) = %v, want %v", name, gotType, wantType)
	}
}
