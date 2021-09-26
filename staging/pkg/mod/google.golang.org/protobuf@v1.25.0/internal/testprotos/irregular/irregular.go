// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package irregular

import (
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/reflect/protodesc"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"

	"google.golang.org/protobuf/types/descriptorpb"
)

type IrregularMessage struct {
	set   bool
	value string
}

func (m *IrregularMessage) ProtoReflect() pref.Message { return (*message)(m) }

type message IrregularMessage

func (m *message) Descriptor() pref.MessageDescriptor { return fileDesc.Messages().Get(0) }
func (m *message) Type() pref.MessageType             { return m }
func (m *message) New() pref.Message                  { return &message{} }
func (m *message) Zero() pref.Message                 { return (*message)(nil) }
func (m *message) Interface() pref.ProtoMessage       { return (*IrregularMessage)(m) }
func (m *message) ProtoMethods() *protoiface.Methods  { return nil }

var fieldDescS = fileDesc.Messages().Get(0).Fields().Get(0)

func (m *message) Range(f func(pref.FieldDescriptor, pref.Value) bool) {
	if m.set {
		f(fieldDescS, pref.ValueOf(m.value))
	}
}

func (m *message) Has(fd pref.FieldDescriptor) bool {
	if fd == fieldDescS {
		return m.set
	}
	panic("invalid field descriptor")
}

func (m *message) Clear(fd pref.FieldDescriptor) {
	if fd == fieldDescS {
		m.value = ""
		m.set = false
		return
	}
	panic("invalid field descriptor")
}

func (m *message) Get(fd pref.FieldDescriptor) pref.Value {
	if fd == fieldDescS {
		return pref.ValueOf(m.value)
	}
	panic("invalid field descriptor")
}

func (m *message) Set(fd pref.FieldDescriptor, v pref.Value) {
	if fd == fieldDescS {
		m.value = v.String()
		m.set = true
		return
	}
	panic("invalid field descriptor")
}

func (m *message) Mutable(pref.FieldDescriptor) pref.Value {
	panic("invalid field descriptor")
}

func (m *message) NewField(pref.FieldDescriptor) pref.Value {
	panic("invalid field descriptor")
}

func (m *message) WhichOneof(pref.OneofDescriptor) pref.FieldDescriptor {
	panic("invalid oneof descriptor")
}

func (m *message) GetUnknown() pref.RawFields { return nil }
func (m *message) SetUnknown(pref.RawFields)  { return }

func (m *message) IsValid() bool {
	return m != nil
}

var fileDesc = func() pref.FileDescriptor {
	p := &descriptorpb.FileDescriptorProto{}
	if err := prototext.Unmarshal([]byte(descriptorText), p); err != nil {
		panic(err)
	}
	file, err := protodesc.NewFile(p, nil)
	if err != nil {
		panic(err)
	}
	return file
}()

func file_internal_testprotos_irregular_irregular_proto_init() { _ = fileDesc }

const descriptorText = `
  name: "internal/testprotos/irregular/irregular.proto"
  package: "goproto.proto.thirdparty"
  message_type {
    name: "IrregularMessage"
    field {
      name: "s"
      number: 1
      label: LABEL_OPTIONAL
      type: TYPE_STRING
      json_name: "s"
    }
  }
  options {
    go_package: "google.golang.org/protobuf/internal/testprotos/irregular"
  }
`

type AberrantMessage int

func (m AberrantMessage) ProtoMessage()            {}
func (m AberrantMessage) Reset()                   {}
func (m AberrantMessage) String() string           { return "" }
func (m AberrantMessage) Marshal() ([]byte, error) { return nil, nil }
func (m AberrantMessage) Unmarshal([]byte) error   { return nil }
