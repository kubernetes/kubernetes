// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ptypes

import (
	"fmt"
	"strings"

	"github.com/golang/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	anypb "github.com/golang/protobuf/ptypes/any"
)

const urlPrefix = "type.googleapis.com/"

// AnyMessageName returns the message name contained in an anypb.Any message.
// Most type assertions should use the Is function instead.
func AnyMessageName(any *anypb.Any) (string, error) {
	name, err := anyMessageName(any)
	return string(name), err
}
func anyMessageName(any *anypb.Any) (protoreflect.FullName, error) {
	if any == nil {
		return "", fmt.Errorf("message is nil")
	}
	name := protoreflect.FullName(any.TypeUrl)
	if i := strings.LastIndex(any.TypeUrl, "/"); i >= 0 {
		name = name[i+len("/"):]
	}
	if !name.IsValid() {
		return "", fmt.Errorf("message type url %q is invalid", any.TypeUrl)
	}
	return name, nil
}

// MarshalAny marshals the given message m into an anypb.Any message.
func MarshalAny(m proto.Message) (*anypb.Any, error) {
	switch dm := m.(type) {
	case DynamicAny:
		m = dm.Message
	case *DynamicAny:
		if dm == nil {
			return nil, proto.ErrNil
		}
		m = dm.Message
	}
	b, err := proto.Marshal(m)
	if err != nil {
		return nil, err
	}
	return &anypb.Any{TypeUrl: urlPrefix + proto.MessageName(m), Value: b}, nil
}

// Empty returns a new message of the type specified in an anypb.Any message.
// It returns protoregistry.NotFound if the corresponding message type could not
// be resolved in the global registry.
func Empty(any *anypb.Any) (proto.Message, error) {
	name, err := anyMessageName(any)
	if err != nil {
		return nil, err
	}
	mt, err := protoregistry.GlobalTypes.FindMessageByName(name)
	if err != nil {
		return nil, err
	}
	return proto.MessageV1(mt.New().Interface()), nil
}

// UnmarshalAny unmarshals the encoded value contained in the anypb.Any message
// into the provided message m. It returns an error if the target message
// does not match the type in the Any message or if an unmarshal error occurs.
//
// The target message m may be a *DynamicAny message. If the underlying message
// type could not be resolved, then this returns protoregistry.NotFound.
func UnmarshalAny(any *anypb.Any, m proto.Message) error {
	if dm, ok := m.(*DynamicAny); ok {
		if dm.Message == nil {
			var err error
			dm.Message, err = Empty(any)
			if err != nil {
				return err
			}
		}
		m = dm.Message
	}

	anyName, err := AnyMessageName(any)
	if err != nil {
		return err
	}
	msgName := proto.MessageName(m)
	if anyName != msgName {
		return fmt.Errorf("mismatched message type: got %q want %q", anyName, msgName)
	}
	return proto.Unmarshal(any.Value, m)
}

// Is reports whether the Any message contains a message of the specified type.
func Is(any *anypb.Any, m proto.Message) bool {
	if any == nil || m == nil {
		return false
	}
	name := proto.MessageName(m)
	if !strings.HasSuffix(any.TypeUrl, name) {
		return false
	}
	return len(any.TypeUrl) == len(name) || any.TypeUrl[len(any.TypeUrl)-len(name)-1] == '/'
}

// DynamicAny is a value that can be passed to UnmarshalAny to automatically
// allocate a proto.Message for the type specified in an anypb.Any message.
// The allocated message is stored in the embedded proto.Message.
//
// Example:
//   var x ptypes.DynamicAny
//   if err := ptypes.UnmarshalAny(a, &x); err != nil { ... }
//   fmt.Printf("unmarshaled message: %v", x.Message)
type DynamicAny struct{ proto.Message }

func (m DynamicAny) String() string {
	if m.Message == nil {
		return "<nil>"
	}
	return m.Message.String()
}
func (m DynamicAny) Reset() {
	if m.Message == nil {
		return
	}
	m.Message.Reset()
}
func (m DynamicAny) ProtoMessage() {
	return
}
func (m DynamicAny) ProtoReflect() protoreflect.Message {
	if m.Message == nil {
		return nil
	}
	return dynamicAny{proto.MessageReflect(m.Message)}
}

type dynamicAny struct{ protoreflect.Message }

func (m dynamicAny) Type() protoreflect.MessageType {
	return dynamicAnyType{m.Message.Type()}
}
func (m dynamicAny) New() protoreflect.Message {
	return dynamicAnyType{m.Message.Type()}.New()
}
func (m dynamicAny) Interface() protoreflect.ProtoMessage {
	return DynamicAny{proto.MessageV1(m.Message.Interface())}
}

type dynamicAnyType struct{ protoreflect.MessageType }

func (t dynamicAnyType) New() protoreflect.Message {
	return dynamicAny{t.MessageType.New()}
}
func (t dynamicAnyType) Zero() protoreflect.Message {
	return dynamicAny{t.MessageType.Zero()}
}
