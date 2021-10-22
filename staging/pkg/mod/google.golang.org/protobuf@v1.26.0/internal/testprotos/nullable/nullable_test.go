// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nullable

import (
	"reflect"
	"testing"

	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoimpl"
)

func Test(t *testing.T) {
	for _, mt := range []protoreflect.MessageType{
		protoimpl.X.ProtoMessageV2Of((*Proto2)(nil)).ProtoReflect().Type(),
		protoimpl.X.ProtoMessageV2Of((*Proto3)(nil)).ProtoReflect().Type(),
	} {
		t.Run(string(mt.Descriptor().FullName()), func(t *testing.T) {
			testEmptyMessage(t, mt.Zero(), false)
			testEmptyMessage(t, mt.New(), true)
			testMethods(t, mt)
		})
	}
}

var testMethods = func(*testing.T, protoreflect.MessageType) {}

func testEmptyMessage(t *testing.T, m protoreflect.Message, wantValid bool) {
	numFields := func(m protoreflect.Message) (n int) {
		m.Range(func(protoreflect.FieldDescriptor, protoreflect.Value) bool {
			n++
			return true
		})
		return n
	}

	md := m.Descriptor()
	if gotValid := m.IsValid(); gotValid != wantValid {
		t.Errorf("%v.IsValid = %v, want %v", md.FullName(), gotValid, wantValid)
	}
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		t.Errorf("%v.Range iterated over field %v, want no iteration", md.FullName(), fd.Name())
		return true
	})
	fds := md.Fields()
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		if m.Has(fd) {
			t.Errorf("%v.Has(%v) = true, want false", md.FullName(), fd.Name())
		}
		v := m.Get(fd)
		switch {
		case fd.IsList():
			if n := v.List().Len(); n > 0 {
				t.Errorf("%v.Get(%v).List().Len() = %v, want 0", md.FullName(), fd.Name(), n)
			}
			ls := m.NewField(fd).List()
			if fd.Message() != nil {
				if n := numFields(ls.NewElement().Message()); n > 0 {
					t.Errorf("%v.NewField(%v).List().NewElement().Message().Len() = %v, want 0", md.FullName(), fd.Name(), n)
				}
			}
		case fd.IsMap():
			if n := v.Map().Len(); n > 0 {
				t.Errorf("%v.Get(%v).Map().Len() = %v, want 0", md.FullName(), fd.Name(), n)
			}
			ms := m.NewField(fd).Map()
			if fd.MapValue().Message() != nil {
				if n := numFields(ms.NewValue().Message()); n > 0 {
					t.Errorf("%v.NewField(%v).Map().NewValue().Message().Len() = %v, want 0", md.FullName(), fd.Name(), n)
				}
			}
		case fd.Message() != nil:
			if n := numFields(v.Message()); n > 0 {
				t.Errorf("%v.Get(%v).Message().Len() = %v, want 0", md.FullName(), fd.Name(), n)
			}
			if n := numFields(m.NewField(fd).Message()); n > 0 {
				t.Errorf("%v.NewField(%v).Message().Len() = %v, want 0", md.FullName(), fd.Name(), n)
			}
		default:
			if !reflect.DeepEqual(v.Interface(), fd.Default().Interface()) {
				t.Errorf("%v.Get(%v) = %v, want %v", md.FullName(), fd.Name(), v, fd.Default())
			}
			m.NewField(fd) // should not panic
		}
	}
	ods := md.Oneofs()
	for i := 0; i < ods.Len(); i++ {
		od := ods.Get(i)
		if fd := m.WhichOneof(od); fd != nil {
			t.Errorf("%v.WhichOneof(%v) = %v, want nil", md.FullName(), od.Name(), fd.Name())
		}
	}
	if b := m.GetUnknown(); b != nil {
		t.Errorf("%v.GetUnknown() = %v, want nil", md.FullName(), b)
	}
}

func testPopulateMessage(t *testing.T, m protoreflect.Message, depth int) bool {
	if depth == 0 {
		return false
	}
	md := m.Descriptor()
	fds := md.Fields()
	var populatedMessage bool
	for i := 0; i < fds.Len(); i++ {
		populatedField := true
		fd := fds.Get(i)
		m.Clear(fd) // should not panic
		switch {
		case fd.IsList():
			ls := m.Mutable(fd).List()
			if fd.Message() == nil {
				ls.Append(scalarValue(fd.Kind()))
			} else {
				populatedField = testPopulateMessage(t, ls.AppendMutable().Message(), depth-1)
			}
		case fd.IsMap():
			ms := m.Mutable(fd).Map()
			if fd.MapValue().Message() == nil {
				ms.Set(
					scalarValue(fd.MapKey().Kind()).MapKey(),
					scalarValue(fd.MapValue().Kind()),
				)
			} else {
				// NOTE: Map.Mutable does not work with non-nullable fields.
				m2 := ms.NewValue().Message()
				populatedField = testPopulateMessage(t, m2, depth-1)
				ms.Set(
					scalarValue(fd.MapKey().Kind()).MapKey(),
					protoreflect.ValueOfMessage(m2),
				)
			}
		case fd.Message() != nil:
			populatedField = testPopulateMessage(t, m.Mutable(fd).Message(), depth-1)
		default:
			m.Set(fd, scalarValue(fd.Kind()))
		}
		if populatedField && !m.Has(fd) {
			t.Errorf("%v.Has(%v) = false, want true", md.FullName(), fd.Name())
		}
		populatedMessage = populatedMessage || populatedField
	}
	m.SetUnknown(m.GetUnknown()) // should not panic
	return populatedMessage
}

func scalarValue(k protoreflect.Kind) protoreflect.Value {
	switch k {
	case protoreflect.BoolKind:
		return protoreflect.ValueOfBool(true)
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		return protoreflect.ValueOfInt32(-32)
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		return protoreflect.ValueOfInt64(-64)
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		return protoreflect.ValueOfUint32(32)
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		return protoreflect.ValueOfUint64(64)
	case protoreflect.FloatKind:
		return protoreflect.ValueOfFloat32(32.32)
	case protoreflect.DoubleKind:
		return protoreflect.ValueOfFloat64(64.64)
	case protoreflect.StringKind:
		return protoreflect.ValueOfString(string("string"))
	case protoreflect.BytesKind:
		return protoreflect.ValueOfBytes([]byte("bytes"))
	case protoreflect.EnumKind:
		return protoreflect.ValueOfEnum(1)
	default:
		panic("unknown kind: " + k.String())
	}
}
