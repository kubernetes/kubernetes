// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"google.golang.org/protobuf/reflect/protoreflect"
)

// DiscardUnknown recursively discards all unknown fields from this message
// and all embedded messages.
//
// When unmarshaling a message with unrecognized fields, the tags and values
// of such fields are preserved in the Message. This allows a later call to
// marshal to be able to produce a message that continues to have those
// unrecognized fields. To avoid this, DiscardUnknown is used to
// explicitly clear the unknown fields after unmarshaling.
func DiscardUnknown(m Message) {
	if m != nil {
		discardUnknown(MessageReflect(m))
	}
}

func discardUnknown(m protoreflect.Message) {
	m.Range(func(fd protoreflect.FieldDescriptor, val protoreflect.Value) bool {
		switch {
		// Handle singular message.
		case fd.Cardinality() != protoreflect.Repeated:
			if fd.Message() != nil {
				discardUnknown(m.Get(fd).Message())
			}
		// Handle list of messages.
		case fd.IsList():
			if fd.Message() != nil {
				ls := m.Get(fd).List()
				for i := 0; i < ls.Len(); i++ {
					discardUnknown(ls.Get(i).Message())
				}
			}
		// Handle map of messages.
		case fd.IsMap():
			if fd.MapValue().Message() != nil {
				ms := m.Get(fd).Map()
				ms.Range(func(_ protoreflect.MapKey, v protoreflect.Value) bool {
					discardUnknown(v.Message())
					return true
				})
			}
		}
		return true
	})

	// Discard unknown fields.
	if len(m.GetUnknown()) > 0 {
		m.SetUnknown(nil)
	}
}
