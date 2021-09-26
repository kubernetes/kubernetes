// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoding_test

import (
	"fmt"
	"testing"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	pref "google.golang.org/protobuf/reflect/protoreflect"

	tpb "google.golang.org/protobuf/internal/testprotos/test"
)

// The results of these microbenchmarks are unlikely to correspond well
// to real world peformance. They are mainly useful as a quick check to
// detect unexpected regressions and for profiling specific cases.

const maxRecurseLevel = 3

func makeProto() *tpb.TestAllTypes {
	m := &tpb.TestAllTypes{}
	fillMessage(m.ProtoReflect(), 0)
	return m
}

func fillMessage(m pref.Message, level int) {
	if level > maxRecurseLevel {
		return
	}

	fieldDescs := m.Descriptor().Fields()
	for i := 0; i < fieldDescs.Len(); i++ {
		fd := fieldDescs.Get(i)
		switch {
		case fd.IsList():
			setList(m.Mutable(fd).List(), fd, level)
		case fd.IsMap():
			setMap(m.Mutable(fd).Map(), fd, level)
		default:
			setScalarField(m, fd, level)
		}
	}
}

func setScalarField(m pref.Message, fd pref.FieldDescriptor, level int) {
	switch fd.Kind() {
	case pref.MessageKind, pref.GroupKind:
		val := m.NewField(fd)
		fillMessage(val.Message(), level+1)
		m.Set(fd, val)
	default:
		m.Set(fd, scalarField(fd.Kind()))
	}
}

func scalarField(kind pref.Kind) pref.Value {
	switch kind {
	case pref.BoolKind:
		return pref.ValueOfBool(true)

	case pref.Int32Kind, pref.Sint32Kind, pref.Sfixed32Kind:
		return pref.ValueOfInt32(1 << 30)

	case pref.Int64Kind, pref.Sint64Kind, pref.Sfixed64Kind:
		return pref.ValueOfInt64(1 << 30)

	case pref.Uint32Kind, pref.Fixed32Kind:
		return pref.ValueOfUint32(1 << 30)

	case pref.Uint64Kind, pref.Fixed64Kind:
		return pref.ValueOfUint64(1 << 30)

	case pref.FloatKind:
		return pref.ValueOfFloat32(3.14159265)

	case pref.DoubleKind:
		return pref.ValueOfFloat64(3.14159265)

	case pref.BytesKind:
		return pref.ValueOfBytes([]byte("hello world"))

	case pref.StringKind:
		return pref.ValueOfString("hello world")

	case pref.EnumKind:
		return pref.ValueOfEnum(42)
	}

	panic(fmt.Sprintf("FieldDescriptor.Kind %v is not valid", kind))
}

func setList(list pref.List, fd pref.FieldDescriptor, level int) {
	switch fd.Kind() {
	case pref.MessageKind, pref.GroupKind:
		for i := 0; i < 10; i++ {
			val := list.NewElement()
			fillMessage(val.Message(), level+1)
			list.Append(val)
		}
	default:
		for i := 0; i < 100; i++ {
			list.Append(scalarField(fd.Kind()))
		}
	}
}

func setMap(mmap pref.Map, fd pref.FieldDescriptor, level int) {
	fields := fd.Message().Fields()
	keyDesc := fields.ByNumber(1)
	valDesc := fields.ByNumber(2)

	pkey := scalarField(keyDesc.Kind())
	switch kind := valDesc.Kind(); kind {
	case pref.MessageKind, pref.GroupKind:
		val := mmap.NewValue()
		fillMessage(val.Message(), level+1)
		mmap.Set(pkey.MapKey(), val)
	default:
		mmap.Set(pkey.MapKey(), scalarField(kind))
	}
}

func BenchmarkTextEncode(b *testing.B) {
	m := makeProto()
	for i := 0; i < b.N; i++ {
		_, err := prototext.MarshalOptions{Indent: "  "}.Marshal(m)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTextDecode(b *testing.B) {
	m := makeProto()
	in, err := prototext.MarshalOptions{Indent: "  "}.Marshal(m)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		m := &tpb.TestAllTypes{}
		if err := prototext.Unmarshal(in, m); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJSONEncode(b *testing.B) {
	m := makeProto()
	for i := 0; i < b.N; i++ {
		_, err := protojson.MarshalOptions{Indent: "  "}.Marshal(m)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJSONDecode(b *testing.B) {
	m := makeProto()
	out, err := protojson.MarshalOptions{Indent: "  "}.Marshal(m)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
		m := &tpb.TestAllTypes{}
		if err := protojson.Unmarshal(out, m); err != nil {
			b.Fatal(err)
		}
	}
}
