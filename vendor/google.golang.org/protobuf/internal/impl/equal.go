// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"bytes"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

func equal(in protoiface.EqualInput) protoiface.EqualOutput {
	return protoiface.EqualOutput{Equal: equalMessage(in.MessageA, in.MessageB)}
}

// equalMessage is a fast-path variant of protoreflect.equalMessage.
// It takes advantage of the internal messageState type to avoid
// unnecessary allocations, type assertions.
func equalMessage(mx, my protoreflect.Message) bool {
	if mx == nil || my == nil {
		return mx == my
	}
	if mx.Descriptor() != my.Descriptor() {
		return false
	}

	msx, ok := mx.(*messageState)
	if !ok {
		return protoreflect.ValueOfMessage(mx).Equal(protoreflect.ValueOfMessage(my))
	}
	msy, ok := my.(*messageState)
	if !ok {
		return protoreflect.ValueOfMessage(mx).Equal(protoreflect.ValueOfMessage(my))
	}

	mi := msx.messageInfo()
	miy := msy.messageInfo()
	if mi != miy {
		return protoreflect.ValueOfMessage(mx).Equal(protoreflect.ValueOfMessage(my))
	}
	mi.init()
	// Compares regular fields
	// Modified Message.Range code that compares two messages of the same type
	// while going over the fields.
	for _, ri := range mi.rangeInfos {
		var fd protoreflect.FieldDescriptor
		var vx, vy protoreflect.Value

		switch ri := ri.(type) {
		case *fieldInfo:
			hx := ri.has(msx.pointer())
			hy := ri.has(msy.pointer())
			if hx != hy {
				return false
			}
			if !hx {
				continue
			}
			fd = ri.fieldDesc
			vx = ri.get(msx.pointer())
			vy = ri.get(msy.pointer())
		case *oneofInfo:
			fnx := ri.which(msx.pointer())
			fny := ri.which(msy.pointer())
			if fnx != fny {
				return false
			}
			if fnx <= 0 {
				continue
			}
			fi := mi.fields[fnx]
			fd = fi.fieldDesc
			vx = fi.get(msx.pointer())
			vy = fi.get(msy.pointer())
		}

		if !equalValue(fd, vx, vy) {
			return false
		}
	}

	// Compare extensions.
	// This is more complicated because mx or my could have empty/nil extension maps,
	// however some populated extension map values are equal to nil extension maps.
	emx := mi.extensionMap(msx.pointer())
	emy := mi.extensionMap(msy.pointer())
	if emx != nil {
		for k, x := range *emx {
			xd := x.Type().TypeDescriptor()
			xv := x.Value()
			var y ExtensionField
			ok := false
			if emy != nil {
				y, ok = (*emy)[k]
			}
			// We need to treat empty lists as equal to nil values
			if emy == nil || !ok {
				if xd.IsList() && xv.List().Len() == 0 {
					continue
				}
				return false
			}

			if !equalValue(xd, xv, y.Value()) {
				return false
			}
		}
	}
	if emy != nil {
		// emy may have extensions emx does not have, need to check them as well
		for k, y := range *emy {
			if emx != nil {
				// emx has the field, so we already checked it
				if _, ok := (*emx)[k]; ok {
					continue
				}
			}
			// Empty lists are equal to nil
			if y.Type().TypeDescriptor().IsList() && y.Value().List().Len() == 0 {
				continue
			}

			// Cant be equal if the extension is populated
			return false
		}
	}

	return equalUnknown(mx.GetUnknown(), my.GetUnknown())
}

func equalValue(fd protoreflect.FieldDescriptor, vx, vy protoreflect.Value) bool {
	// slow path
	if fd.Kind() != protoreflect.MessageKind {
		return vx.Equal(vy)
	}

	// fast path special cases
	if fd.IsMap() {
		if fd.MapValue().Kind() == protoreflect.MessageKind {
			return equalMessageMap(vx.Map(), vy.Map())
		}
		return vx.Equal(vy)
	}

	if fd.IsList() {
		return equalMessageList(vx.List(), vy.List())
	}

	return equalMessage(vx.Message(), vy.Message())
}

// Mostly copied from protoreflect.equalMap.
// This variant only works for messages as map types.
// All other map types should be handled via Value.Equal.
func equalMessageMap(mx, my protoreflect.Map) bool {
	if mx.Len() != my.Len() {
		return false
	}
	equal := true
	mx.Range(func(k protoreflect.MapKey, vx protoreflect.Value) bool {
		if !my.Has(k) {
			equal = false
			return false
		}
		vy := my.Get(k)
		equal = equalMessage(vx.Message(), vy.Message())
		return equal
	})
	return equal
}

// Mostly copied from protoreflect.equalList.
// The only change is the usage of equalImpl instead of protoreflect.equalValue.
func equalMessageList(lx, ly protoreflect.List) bool {
	if lx.Len() != ly.Len() {
		return false
	}
	for i := 0; i < lx.Len(); i++ {
		// We only operate on messages here since equalImpl will not call us in any other case.
		if !equalMessage(lx.Get(i).Message(), ly.Get(i).Message()) {
			return false
		}
	}
	return true
}

// equalUnknown compares unknown fields by direct comparison on the raw bytes
// of each individual field number.
// Copied from protoreflect.equalUnknown.
func equalUnknown(x, y protoreflect.RawFields) bool {
	if len(x) != len(y) {
		return false
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
	if len(mx) != len(my) {
		return false
	}

	for k, v1 := range mx {
		if v2, ok := my[k]; !ok || !bytes.Equal([]byte(v1), []byte(v2)) {
			return false
		}
	}

	return true
}
