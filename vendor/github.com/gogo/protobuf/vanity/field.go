// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2015, The GoGo Authors.  rights reserved.
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

package vanity

import (
	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
)

func FieldHasBoolExtension(field *descriptor.FieldDescriptorProto, extension *proto.ExtensionDesc) bool {
	if field.Options == nil {
		return false
	}
	value, err := proto.GetExtension(field.Options, extension)
	if err != nil {
		return false
	}
	if value == nil {
		return false
	}
	if value.(*bool) == nil {
		return false
	}
	return true
}

func SetBoolFieldOption(extension *proto.ExtensionDesc, value bool) func(field *descriptor.FieldDescriptorProto) {
	return func(field *descriptor.FieldDescriptorProto) {
		if FieldHasBoolExtension(field, extension) {
			return
		}
		if field.Options == nil {
			field.Options = &descriptor.FieldOptions{}
		}
		if err := proto.SetExtension(field.Options, extension, &value); err != nil {
			panic(err)
		}
	}
}

func TurnOffNullable(field *descriptor.FieldDescriptorProto) {
	if field.IsRepeated() && !field.IsMessage() {
		return
	}
	SetBoolFieldOption(gogoproto.E_Nullable, false)(field)
}

func TurnOffNullableForNativeTypes(field *descriptor.FieldDescriptorProto) {
	if field.IsRepeated() || field.IsMessage() {
		return
	}
	SetBoolFieldOption(gogoproto.E_Nullable, false)(field)
}

func TurnOffNullableForNativeTypesWithoutDefaultsOnly(field *descriptor.FieldDescriptorProto) {
	if field.IsRepeated() || field.IsMessage() {
		return
	}
	if field.DefaultValue != nil {
		return
	}
	SetBoolFieldOption(gogoproto.E_Nullable, false)(field)
}
