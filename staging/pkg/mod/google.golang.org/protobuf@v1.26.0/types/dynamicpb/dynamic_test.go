// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dynamicpb_test

import (
	"testing"

	"google.golang.org/protobuf/proto"
	pref "google.golang.org/protobuf/reflect/protoreflect"
	preg "google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/testing/prototest"
	"google.golang.org/protobuf/types/dynamicpb"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	test3pb "google.golang.org/protobuf/internal/testprotos/test3"
)

func TestConformance(t *testing.T) {
	for _, message := range []proto.Message{
		(*testpb.TestAllTypes)(nil),
		(*test3pb.TestAllTypes)(nil),
		(*testpb.TestAllExtensions)(nil),
	} {
		mt := dynamicpb.NewMessageType(message.ProtoReflect().Descriptor())
		prototest.Message{}.Test(t, mt)
	}
}

func TestDynamicExtensions(t *testing.T) {
	for _, message := range []proto.Message{
		(*testpb.TestAllExtensions)(nil),
	} {
		mt := dynamicpb.NewMessageType(message.ProtoReflect().Descriptor())
		prototest.Message{
			Resolver: extResolver{},
		}.Test(t, mt)
	}
}

func TestDynamicEnums(t *testing.T) {
	for _, enum := range []pref.Enum{
		testpb.TestAllTypes_FOO,
		test3pb.TestAllTypes_FOO,
	} {
		et := dynamicpb.NewEnumType(enum.Descriptor())
		prototest.Enum{}.Test(t, et)
	}
}

type extResolver struct{}

func (extResolver) FindExtensionByName(field pref.FullName) (pref.ExtensionType, error) {
	xt, err := preg.GlobalTypes.FindExtensionByName(field)
	if err != nil {
		return nil, err
	}
	return dynamicpb.NewExtensionType(xt.TypeDescriptor().Descriptor()), nil
}

func (extResolver) FindExtensionByNumber(message pref.FullName, field pref.FieldNumber) (pref.ExtensionType, error) {
	xt, err := preg.GlobalTypes.FindExtensionByNumber(message, field)
	if err != nil {
		return nil, err
	}
	return dynamicpb.NewExtensionType(xt.TypeDescriptor().Descriptor()), nil
}

func (extResolver) RangeExtensionsByMessage(message pref.FullName, f func(pref.ExtensionType) bool) {
	preg.GlobalTypes.RangeExtensionsByMessage(message, func(xt pref.ExtensionType) bool {
		return f(dynamicpb.NewExtensionType(xt.TypeDescriptor().Descriptor()))
	})
}
