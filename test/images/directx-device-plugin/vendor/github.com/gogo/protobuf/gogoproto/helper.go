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

package gogoproto

import google_protobuf "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"
import proto "github.com/gogo/protobuf/proto"

func IsEmbed(field *google_protobuf.FieldDescriptorProto) bool {
	return proto.GetBoolExtension(field.Options, E_Embed, false)
}

func IsNullable(field *google_protobuf.FieldDescriptorProto) bool {
	return proto.GetBoolExtension(field.Options, E_Nullable, true)
}

func IsStdTime(field *google_protobuf.FieldDescriptorProto) bool {
	return proto.GetBoolExtension(field.Options, E_Stdtime, false)
}

func IsStdDuration(field *google_protobuf.FieldDescriptorProto) bool {
	return proto.GetBoolExtension(field.Options, E_Stdduration, false)
}

func NeedsNilCheck(proto3 bool, field *google_protobuf.FieldDescriptorProto) bool {
	nullable := IsNullable(field)
	if field.IsMessage() || IsCustomType(field) {
		return nullable
	}
	if proto3 {
		return false
	}
	return nullable || *field.Type == google_protobuf.FieldDescriptorProto_TYPE_BYTES
}

func IsCustomType(field *google_protobuf.FieldDescriptorProto) bool {
	typ := GetCustomType(field)
	if len(typ) > 0 {
		return true
	}
	return false
}

func IsCastType(field *google_protobuf.FieldDescriptorProto) bool {
	typ := GetCastType(field)
	if len(typ) > 0 {
		return true
	}
	return false
}

func IsCastKey(field *google_protobuf.FieldDescriptorProto) bool {
	typ := GetCastKey(field)
	if len(typ) > 0 {
		return true
	}
	return false
}

func IsCastValue(field *google_protobuf.FieldDescriptorProto) bool {
	typ := GetCastValue(field)
	if len(typ) > 0 {
		return true
	}
	return false
}

func HasEnumDecl(file *google_protobuf.FileDescriptorProto, enum *google_protobuf.EnumDescriptorProto) bool {
	return proto.GetBoolExtension(enum.Options, E_Enumdecl, proto.GetBoolExtension(file.Options, E_EnumdeclAll, true))
}

func HasTypeDecl(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Typedecl, proto.GetBoolExtension(file.Options, E_TypedeclAll, true))
}

func GetCustomType(field *google_protobuf.FieldDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Customtype)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func GetCastType(field *google_protobuf.FieldDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Casttype)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func GetCastKey(field *google_protobuf.FieldDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Castkey)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func GetCastValue(field *google_protobuf.FieldDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Castvalue)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func IsCustomName(field *google_protobuf.FieldDescriptorProto) bool {
	name := GetCustomName(field)
	if len(name) > 0 {
		return true
	}
	return false
}

func IsEnumCustomName(field *google_protobuf.EnumDescriptorProto) bool {
	name := GetEnumCustomName(field)
	if len(name) > 0 {
		return true
	}
	return false
}

func IsEnumValueCustomName(field *google_protobuf.EnumValueDescriptorProto) bool {
	name := GetEnumValueCustomName(field)
	if len(name) > 0 {
		return true
	}
	return false
}

func GetCustomName(field *google_protobuf.FieldDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Customname)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func GetEnumCustomName(field *google_protobuf.EnumDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_EnumCustomname)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func GetEnumValueCustomName(field *google_protobuf.EnumValueDescriptorProto) string {
	if field == nil {
		return ""
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_EnumvalueCustomname)
		if err == nil && v.(*string) != nil {
			return *(v.(*string))
		}
	}
	return ""
}

func GetJsonTag(field *google_protobuf.FieldDescriptorProto) *string {
	if field == nil {
		return nil
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Jsontag)
		if err == nil && v.(*string) != nil {
			return (v.(*string))
		}
	}
	return nil
}

func GetMoreTags(field *google_protobuf.FieldDescriptorProto) *string {
	if field == nil {
		return nil
	}
	if field.Options != nil {
		v, err := proto.GetExtension(field.Options, E_Moretags)
		if err == nil && v.(*string) != nil {
			return (v.(*string))
		}
	}
	return nil
}

type EnableFunc func(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool

func EnabledGoEnumPrefix(file *google_protobuf.FileDescriptorProto, enum *google_protobuf.EnumDescriptorProto) bool {
	return proto.GetBoolExtension(enum.Options, E_GoprotoEnumPrefix, proto.GetBoolExtension(file.Options, E_GoprotoEnumPrefixAll, true))
}

func EnabledGoStringer(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_GoprotoStringer, proto.GetBoolExtension(file.Options, E_GoprotoStringerAll, true))
}

func HasGoGetters(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_GoprotoGetters, proto.GetBoolExtension(file.Options, E_GoprotoGettersAll, true))
}

func IsUnion(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Onlyone, proto.GetBoolExtension(file.Options, E_OnlyoneAll, false))
}

func HasGoString(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Gostring, proto.GetBoolExtension(file.Options, E_GostringAll, false))
}

func HasEqual(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Equal, proto.GetBoolExtension(file.Options, E_EqualAll, false))
}

func HasVerboseEqual(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_VerboseEqual, proto.GetBoolExtension(file.Options, E_VerboseEqualAll, false))
}

func IsStringer(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Stringer, proto.GetBoolExtension(file.Options, E_StringerAll, false))
}

func IsFace(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Face, proto.GetBoolExtension(file.Options, E_FaceAll, false))
}

func HasDescription(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Description, proto.GetBoolExtension(file.Options, E_DescriptionAll, false))
}

func HasPopulate(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Populate, proto.GetBoolExtension(file.Options, E_PopulateAll, false))
}

func HasTestGen(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Testgen, proto.GetBoolExtension(file.Options, E_TestgenAll, false))
}

func HasBenchGen(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Benchgen, proto.GetBoolExtension(file.Options, E_BenchgenAll, false))
}

func IsMarshaler(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Marshaler, proto.GetBoolExtension(file.Options, E_MarshalerAll, false))
}

func IsUnmarshaler(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Unmarshaler, proto.GetBoolExtension(file.Options, E_UnmarshalerAll, false))
}

func IsStableMarshaler(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_StableMarshaler, proto.GetBoolExtension(file.Options, E_StableMarshalerAll, false))
}

func IsSizer(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Sizer, proto.GetBoolExtension(file.Options, E_SizerAll, false))
}

func IsProtoSizer(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Protosizer, proto.GetBoolExtension(file.Options, E_ProtosizerAll, false))
}

func IsGoEnumStringer(file *google_protobuf.FileDescriptorProto, enum *google_protobuf.EnumDescriptorProto) bool {
	return proto.GetBoolExtension(enum.Options, E_GoprotoEnumStringer, proto.GetBoolExtension(file.Options, E_GoprotoEnumStringerAll, true))
}

func IsEnumStringer(file *google_protobuf.FileDescriptorProto, enum *google_protobuf.EnumDescriptorProto) bool {
	return proto.GetBoolExtension(enum.Options, E_EnumStringer, proto.GetBoolExtension(file.Options, E_EnumStringerAll, false))
}

func IsUnsafeMarshaler(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_UnsafeMarshaler, proto.GetBoolExtension(file.Options, E_UnsafeMarshalerAll, false))
}

func IsUnsafeUnmarshaler(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_UnsafeUnmarshaler, proto.GetBoolExtension(file.Options, E_UnsafeUnmarshalerAll, false))
}

func HasExtensionsMap(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_GoprotoExtensionsMap, proto.GetBoolExtension(file.Options, E_GoprotoExtensionsMapAll, true))
}

func HasUnrecognized(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	if IsProto3(file) {
		return false
	}
	return proto.GetBoolExtension(message.Options, E_GoprotoUnrecognized, proto.GetBoolExtension(file.Options, E_GoprotoUnrecognizedAll, true))
}

func IsProto3(file *google_protobuf.FileDescriptorProto) bool {
	return file.GetSyntax() == "proto3"
}

func ImportsGoGoProto(file *google_protobuf.FileDescriptorProto) bool {
	return proto.GetBoolExtension(file.Options, E_GogoprotoImport, true)
}

func HasCompare(file *google_protobuf.FileDescriptorProto, message *google_protobuf.DescriptorProto) bool {
	return proto.GetBoolExtension(message.Options, E_Compare, proto.GetBoolExtension(file.Options, E_CompareAll, false))
}

func RegistersGolangProto(file *google_protobuf.FileDescriptorProto) bool {
	return proto.GetBoolExtension(file.Options, E_GoprotoRegistration, false)
}
