// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pb

import (
	"bytes"
	"reflect"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	anypb "google.golang.org/protobuf/types/known/anypb"
)

// Equal returns whether two proto.Message instances are equal using the following criteria:
//
//   - Messages must share the same instance of the type descriptor
//   - Known set fields are compared using semantics equality
//     - Bytes are compared using bytes.Equal
//     - Scalar values are compared with operator ==
//     - List and map types are equal if they have the same length and all elements are equal
//     - Messages are equal if they share the same descriptor and all set fields are equal
//   - Unknown fields are compared using byte equality
//   - NaN values are not equal to each other
//   - google.protobuf.Any values are unpacked before comparison
//   - If the type descriptor for a protobuf.Any cannot be found, byte equality is used rather than
//     semantic equality.
//
// This method of proto equality mirrors the behavior of the C++ protobuf MessageDifferencer
// whereas the golang proto.Equal implementation mirrors the Java protobuf equals() methods
// behaviors which needed to treat NaN values as equal due to Java semantics.
func Equal(x, y proto.Message) bool {
	if x == nil || y == nil {
		return x == nil && y == nil
	}
	xRef := x.ProtoReflect()
	yRef := y.ProtoReflect()
	return equalMessage(xRef, yRef)
}

func equalMessage(mx, my protoreflect.Message) bool {
	// Note, the original proto.Equal upon which this implementation is based does not specifically handle the
	// case when both messages are invalid. It is assumed that the descriptors will be equal and that byte-wise
	// comparison will be used, though the semantics of validity are neither clear, nor promised within the
	//  proto.Equal implementation.
	if mx.IsValid() != my.IsValid() || mx.Descriptor() != my.Descriptor() {
		return false
	}

	// This is an innovation on the default proto.Equal where protobuf.Any values are unpacked before comparison
	// as otherwise the Any values are compared by bytes rather than structurally.
	if isAny(mx) && isAny(my) {
		ax := mx.Interface().(*anypb.Any)
		ay := my.Interface().(*anypb.Any)
		// If the values are not the same type url, return false.
		if ax.GetTypeUrl() != ay.GetTypeUrl() {
			return false
		}
		// If the values are byte equal, then return true.
		if bytes.Equal(ax.GetValue(), ay.GetValue()) {
			return true
		}
		// Otherwise fall through to the semantic comparison of the any values.
		x, err := ax.UnmarshalNew()
		if err != nil {
			return false
		}
		y, err := ay.UnmarshalNew()
		if err != nil {
			return false
		}
		// Recursively compare the unwrapped messages to ensure nested Any values are unwrapped accordingly.
		return equalMessage(x.ProtoReflect(), y.ProtoReflect())
	}

	// Walk the set fields to determine field-wise equality
	nx := 0
	equal := true
	mx.Range(func(fd protoreflect.FieldDescriptor, vx protoreflect.Value) bool {
		nx++
		equal = my.Has(fd) && equalField(fd, vx, my.Get(fd))
		return equal
	})
	if !equal {
		return false
	}
	// Establish the count of set fields on message y
	ny := 0
	my.Range(func(protoreflect.FieldDescriptor, protoreflect.Value) bool {
		ny++
		return true
	})
	// If the number of set fields is not equal return false.
	if nx != ny {
		return false
	}

	return equalUnknown(mx.GetUnknown(), my.GetUnknown())
}

func equalField(fd protoreflect.FieldDescriptor, x, y protoreflect.Value) bool {
	switch {
	case fd.IsMap():
		return equalMap(fd, x.Map(), y.Map())
	case fd.IsList():
		return equalList(fd, x.List(), y.List())
	default:
		return equalValue(fd, x, y)
	}
}

func equalMap(fd protoreflect.FieldDescriptor, x, y protoreflect.Map) bool {
	if x.Len() != y.Len() {
		return false
	}
	equal := true
	x.Range(func(k protoreflect.MapKey, vx protoreflect.Value) bool {
		vy := y.Get(k)
		equal = y.Has(k) && equalValue(fd.MapValue(), vx, vy)
		return equal
	})
	return equal
}

func equalList(fd protoreflect.FieldDescriptor, x, y protoreflect.List) bool {
	if x.Len() != y.Len() {
		return false
	}
	for i := x.Len() - 1; i >= 0; i-- {
		if !equalValue(fd, x.Get(i), y.Get(i)) {
			return false
		}
	}
	return true
}

func equalValue(fd protoreflect.FieldDescriptor, x, y protoreflect.Value) bool {
	switch fd.Kind() {
	case protoreflect.BoolKind:
		return x.Bool() == y.Bool()
	case protoreflect.EnumKind:
		return x.Enum() == y.Enum()
	case protoreflect.Int32Kind, protoreflect.Sint32Kind,
		protoreflect.Int64Kind, protoreflect.Sint64Kind,
		protoreflect.Sfixed32Kind, protoreflect.Sfixed64Kind:
		return x.Int() == y.Int()
	case protoreflect.Uint32Kind, protoreflect.Uint64Kind,
		protoreflect.Fixed32Kind, protoreflect.Fixed64Kind:
		return x.Uint() == y.Uint()
	case protoreflect.FloatKind, protoreflect.DoubleKind:
		return x.Float() == y.Float()
	case protoreflect.StringKind:
		return x.String() == y.String()
	case protoreflect.BytesKind:
		return bytes.Equal(x.Bytes(), y.Bytes())
	case protoreflect.MessageKind, protoreflect.GroupKind:
		return equalMessage(x.Message(), y.Message())
	default:
		return x.Interface() == y.Interface()
	}
}

func equalUnknown(x, y protoreflect.RawFields) bool {
	lenX := len(x)
	lenY := len(y)
	if lenX != lenY {
		return false
	}
	if lenX == 0 {
		return true
	}
	if bytes.Equal([]byte(x), []byte(y)) {
		return true
	}

	mx := make(map[protoreflect.FieldNumber]protoreflect.RawFields)
	my := make(map[protoreflect.FieldNumber]protoreflect.RawFields)
	for len(x) > 0 {
		fnum, _, n := protowire.ConsumeField(x)
		mx[fnum] = append(mx[fnum], x[:n]...)
		x = x[n:]
	}
	for len(y) > 0 {
		fnum, _, n := protowire.ConsumeField(y)
		my[fnum] = append(my[fnum], y[:n]...)
		y = y[n:]
	}
	return reflect.DeepEqual(mx, my)
}

func isAny(m protoreflect.Message) bool {
	return string(m.Descriptor().FullName()) == "google.protobuf.Any"
}
