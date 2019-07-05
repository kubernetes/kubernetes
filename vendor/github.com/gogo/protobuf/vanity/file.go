// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2015, The GoGo Authors. All rights reserved.
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
	"path/filepath"

	"github.com/gogo/protobuf/gogoproto"
	"github.com/gogo/protobuf/proto"
	descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
)

func NotGoogleProtobufDescriptorProto(file *descriptor.FileDescriptorProto) bool {
	// can not just check if file.GetName() == "google/protobuf/descriptor.proto" because we do not want to assume compile path
	_, fileName := filepath.Split(file.GetName())
	return !(file.GetPackage() == "google.protobuf" && fileName == "descriptor.proto")
}

func FilterFiles(files []*descriptor.FileDescriptorProto, f func(file *descriptor.FileDescriptorProto) bool) []*descriptor.FileDescriptorProto {
	filtered := make([]*descriptor.FileDescriptorProto, 0, len(files))
	for i := range files {
		if !f(files[i]) {
			continue
		}
		filtered = append(filtered, files[i])
	}
	return filtered
}

func FileHasBoolExtension(file *descriptor.FileDescriptorProto, extension *proto.ExtensionDesc) bool {
	if file.Options == nil {
		return false
	}
	value, err := proto.GetExtension(file.Options, extension)
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

func SetBoolFileOption(extension *proto.ExtensionDesc, value bool) func(file *descriptor.FileDescriptorProto) {
	return func(file *descriptor.FileDescriptorProto) {
		if FileHasBoolExtension(file, extension) {
			return
		}
		if file.Options == nil {
			file.Options = &descriptor.FileOptions{}
		}
		if err := proto.SetExtension(file.Options, extension, &value); err != nil {
			panic(err)
		}
	}
}

func TurnOffGoGettersAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GoprotoGettersAll, false)(file)
}

func TurnOffGoEnumPrefixAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GoprotoEnumPrefixAll, false)(file)
}

func TurnOffGoStringerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GoprotoStringerAll, false)(file)
}

func TurnOnVerboseEqualAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_VerboseEqualAll, true)(file)
}

func TurnOnFaceAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_FaceAll, true)(file)
}

func TurnOnGoStringAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GostringAll, true)(file)
}

func TurnOnPopulateAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_PopulateAll, true)(file)
}

func TurnOnStringerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_StringerAll, true)(file)
}

func TurnOnEqualAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_EqualAll, true)(file)
}

func TurnOnDescriptionAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_DescriptionAll, true)(file)
}

func TurnOnTestGenAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_TestgenAll, true)(file)
}

func TurnOnBenchGenAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_BenchgenAll, true)(file)
}

func TurnOnMarshalerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_MarshalerAll, true)(file)
}

func TurnOnUnmarshalerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_UnmarshalerAll, true)(file)
}

func TurnOnStable_MarshalerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_StableMarshalerAll, true)(file)
}

func TurnOnSizerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_SizerAll, true)(file)
}

func TurnOffGoEnumStringerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GoprotoEnumStringerAll, false)(file)
}

func TurnOnEnumStringerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_EnumStringerAll, true)(file)
}

func TurnOnUnsafeUnmarshalerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_UnsafeUnmarshalerAll, true)(file)
}

func TurnOnUnsafeMarshalerAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_UnsafeMarshalerAll, true)(file)
}

func TurnOffGoExtensionsMapAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GoprotoExtensionsMapAll, false)(file)
}

func TurnOffGoUnrecognizedAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GoprotoUnrecognizedAll, false)(file)
}

func TurnOffGogoImport(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_GogoprotoImport, false)(file)
}

func TurnOnCompareAll(file *descriptor.FileDescriptorProto) {
	SetBoolFileOption(gogoproto.E_CompareAll, true)(file)
}
