// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonpb provides functionality to marshal and unmarshal between a
// protocol buffer message and JSON. It follows the specification at
// https://developers.google.com/protocol-buffers/docs/proto3#json.
//
// Do not rely on the default behavior of the standard encoding/json package
// when called on generated message types as it does not operate correctly.
//
// Deprecated: Use the "google.golang.org/protobuf/encoding/protojson"
// package instead.
package jsonpb

import (
	"github.com/golang/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoimpl"
)

// AnyResolver takes a type URL, present in an Any message,
// and resolves it into an instance of the associated message.
type AnyResolver interface {
	Resolve(typeURL string) (proto.Message, error)
}

type anyResolver struct{ AnyResolver }

func (r anyResolver) FindMessageByName(message protoreflect.FullName) (protoreflect.MessageType, error) {
	return r.FindMessageByURL(string(message))
}

func (r anyResolver) FindMessageByURL(url string) (protoreflect.MessageType, error) {
	m, err := r.Resolve(url)
	if err != nil {
		return nil, err
	}
	return protoimpl.X.MessageTypeOf(m), nil
}

func (r anyResolver) FindExtensionByName(field protoreflect.FullName) (protoreflect.ExtensionType, error) {
	return protoregistry.GlobalTypes.FindExtensionByName(field)
}

func (r anyResolver) FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error) {
	return protoregistry.GlobalTypes.FindExtensionByNumber(message, field)
}

func wellKnownType(s protoreflect.FullName) string {
	if s.Parent() == "google.protobuf" {
		switch s.Name() {
		case "Empty", "Any",
			"BoolValue", "BytesValue", "StringValue",
			"Int32Value", "UInt32Value", "FloatValue",
			"Int64Value", "UInt64Value", "DoubleValue",
			"Duration", "Timestamp",
			"NullValue", "Struct", "Value", "ListValue":
			return string(s.Name())
		}
	}
	return ""
}

func isMessageSet(md protoreflect.MessageDescriptor) bool {
	ms, ok := md.(interface{ IsMessageSet() bool })
	return ok && ms.IsMessageSet()
}
