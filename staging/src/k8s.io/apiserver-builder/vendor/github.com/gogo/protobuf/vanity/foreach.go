// Extensions for Protocol Buffers to create more go like structures.
//
// Copyright (c) 2015, Vastech SA (PTY) LTD. All rights reserved.
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

import descriptor "github.com/gogo/protobuf/protoc-gen-gogo/descriptor"

func ForEachFile(files []*descriptor.FileDescriptorProto, f func(file *descriptor.FileDescriptorProto)) {
	for _, file := range files {
		f(file)
	}
}

func OnlyProto2(files []*descriptor.FileDescriptorProto) []*descriptor.FileDescriptorProto {
	outs := make([]*descriptor.FileDescriptorProto, 0, len(files))
	for i, file := range files {
		if file.GetSyntax() == "proto3" {
			continue
		}
		outs = append(outs, files[i])
	}
	return outs
}

func OnlyProto3(files []*descriptor.FileDescriptorProto) []*descriptor.FileDescriptorProto {
	outs := make([]*descriptor.FileDescriptorProto, 0, len(files))
	for i, file := range files {
		if file.GetSyntax() != "proto3" {
			continue
		}
		outs = append(outs, files[i])
	}
	return outs
}

func ForEachMessageInFiles(files []*descriptor.FileDescriptorProto, f func(msg *descriptor.DescriptorProto)) {
	for _, file := range files {
		ForEachMessage(file.MessageType, f)
	}
}

func ForEachMessage(msgs []*descriptor.DescriptorProto, f func(msg *descriptor.DescriptorProto)) {
	for _, msg := range msgs {
		f(msg)
		ForEachMessage(msg.NestedType, f)
	}
}

func ForEachFieldInFilesExcludingExtensions(files []*descriptor.FileDescriptorProto, f func(field *descriptor.FieldDescriptorProto)) {
	for _, file := range files {
		ForEachFieldExcludingExtensions(file.MessageType, f)
	}
}

func ForEachFieldInFiles(files []*descriptor.FileDescriptorProto, f func(field *descriptor.FieldDescriptorProto)) {
	for _, file := range files {
		for _, ext := range file.Extension {
			f(ext)
		}
		ForEachField(file.MessageType, f)
	}
}

func ForEachFieldExcludingExtensions(msgs []*descriptor.DescriptorProto, f func(field *descriptor.FieldDescriptorProto)) {
	for _, msg := range msgs {
		for _, field := range msg.Field {
			f(field)
		}
		ForEachField(msg.NestedType, f)
	}
}

func ForEachField(msgs []*descriptor.DescriptorProto, f func(field *descriptor.FieldDescriptorProto)) {
	for _, msg := range msgs {
		for _, field := range msg.Field {
			f(field)
		}
		for _, ext := range msg.Extension {
			f(ext)
		}
		ForEachField(msg.NestedType, f)
	}
}

func ForEachEnumInFiles(files []*descriptor.FileDescriptorProto, f func(enum *descriptor.EnumDescriptorProto)) {
	for _, file := range files {
		for _, enum := range file.EnumType {
			f(enum)
		}
	}
}

func ForEachEnum(msgs []*descriptor.DescriptorProto, f func(field *descriptor.EnumDescriptorProto)) {
	for _, msg := range msgs {
		for _, field := range msg.EnumType {
			f(field)
		}
		ForEachEnum(msg.NestedType, f)
	}
}
