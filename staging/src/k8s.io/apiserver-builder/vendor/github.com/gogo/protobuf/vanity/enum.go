// Extensions for Protocol Buffers to create more go like structures.
//
// Copyright (c) 2015, Vastech SA (PTY) LTD.  rights reserved.
// http://github.com/gogo/protobuf/gogoproto
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

func EnumHasBoolExtension(enum *descriptor.EnumDescriptorProto, extension *proto.ExtensionDesc) bool {
	if enum.Options == nil {
		return false
	}
	value, err := proto.GetExtension(enum.Options, extension)
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

func SetBoolEnumOption(extension *proto.ExtensionDesc, value bool) func(enum *descriptor.EnumDescriptorProto) {
	return func(enum *descriptor.EnumDescriptorProto) {
		if EnumHasBoolExtension(enum, extension) {
			return
		}
		if enum.Options == nil {
			enum.Options = &descriptor.EnumOptions{}
		}
		if err := proto.SetExtension(enum.Options, extension, &value); err != nil {
			panic(err)
		}
	}
}

func TurnOffGoEnumPrefix(enum *descriptor.EnumDescriptorProto) {
	SetBoolEnumOption(gogoproto.E_GoprotoEnumPrefix, false)(enum)
}

func TurnOffGoEnumStringer(enum *descriptor.EnumDescriptorProto) {
	SetBoolEnumOption(gogoproto.E_GoprotoEnumStringer, false)(enum)
}

func TurnOnEnumStringer(enum *descriptor.EnumDescriptorProto) {
	SetBoolEnumOption(gogoproto.E_EnumStringer, true)(enum)
}
