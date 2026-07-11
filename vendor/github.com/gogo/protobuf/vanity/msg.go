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

func MessageHasBoolExtension(msg *descriptor.DescriptorProto, extension *proto.ExtensionDesc) bool {
	if msg.Options == nil {
		return false
	}
	value, err := proto.GetExtension(msg.Options, extension)
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

func SetBoolMessageOption(extension *proto.ExtensionDesc, value bool) func(msg *descriptor.DescriptorProto) {
	return func(msg *descriptor.DescriptorProto) {
		if MessageHasBoolExtension(msg, extension) {
			return
		}
		if msg.Options == nil {
			msg.Options = &descriptor.MessageOptions{}
		}
		if err := proto.SetExtension(msg.Options, extension, &value); err != nil {
			panic(err)
		}
	}
}

func TurnOffGoGetters(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_GoprotoGetters, false)(msg)
}

func TurnOffGoStringer(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_GoprotoStringer, false)(msg)
}

func TurnOnVerboseEqual(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_VerboseEqual, true)(msg)
}

func TurnOnFace(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Face, true)(msg)
}

func TurnOnGoString(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Face, true)(msg)
}

func TurnOnPopulate(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Populate, true)(msg)
}

func TurnOnStringer(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Stringer, true)(msg)
}

func TurnOnEqual(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Equal, true)(msg)
}

func TurnOnDescription(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Description, true)(msg)
}

func TurnOnTestGen(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Testgen, true)(msg)
}

func TurnOnBenchGen(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Benchgen, true)(msg)
}

func TurnOnMarshaler(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Marshaler, true)(msg)
}

func TurnOnUnmarshaler(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Unmarshaler, true)(msg)
}

func TurnOnSizer(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Sizer, true)(msg)
}

func TurnOnUnsafeUnmarshaler(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_UnsafeUnmarshaler, true)(msg)
}

func TurnOnUnsafeMarshaler(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_UnsafeMarshaler, true)(msg)
}

func TurnOffGoExtensionsMap(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_GoprotoExtensionsMap, false)(msg)
}

func TurnOffGoUnrecognized(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_GoprotoUnrecognized, false)(msg)
}

func TurnOffGoUnkeyed(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_GoprotoUnkeyed, false)(msg)
}

func TurnOffGoSizecache(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_GoprotoSizecache, false)(msg)
}

func TurnOnCompare(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Compare, true)(msg)
}

func TurnOnMessageName(msg *descriptor.DescriptorProto) {
	SetBoolMessageOption(gogoproto.E_Messagename, true)(msg)
}
