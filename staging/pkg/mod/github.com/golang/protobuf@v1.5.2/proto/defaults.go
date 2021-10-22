// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"google.golang.org/protobuf/reflect/protoreflect"
)

// SetDefaults sets unpopulated scalar fields to their default values.
// Fields within a oneof are not set even if they have a default value.
// SetDefaults is recursively called upon any populated message fields.
func SetDefaults(m Message) {
	if m != nil {
		setDefaults(MessageReflect(m))
	}
}

func setDefaults(m protoreflect.Message) {
	fds := m.Descriptor().Fields()
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		if !m.Has(fd) {
			if fd.HasDefault() && fd.ContainingOneof() == nil {
				v := fd.Default()
				if fd.Kind() == protoreflect.BytesKind {
					v = protoreflect.ValueOf(append([]byte(nil), v.Bytes()...)) // copy the default bytes
				}
				m.Set(fd, v)
			}
			continue
		}
	}

	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		switch {
		// Handle singular message.
		case fd.Cardinality() != protoreflect.Repeated:
			if fd.Message() != nil {
				setDefaults(m.Get(fd).Message())
			}
		// Handle list of messages.
		case fd.IsList():
			if fd.Message() != nil {
				ls := m.Get(fd).List()
				for i := 0; i < ls.Len(); i++ {
					setDefaults(ls.Get(i).Message())
				}
			}
		// Handle map of messages.
		case fd.IsMap():
			if fd.MapValue().Message() != nil {
				ms := m.Get(fd).Map()
				ms.Range(func(_ protoreflect.MapKey, v protoreflect.Value) bool {
					setDefaults(v.Message())
					return true
				})
			}
		}
		return true
	})
}
