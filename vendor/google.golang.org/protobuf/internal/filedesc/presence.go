// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import "google.golang.org/protobuf/reflect/protoreflect"

// UsePresenceForField reports whether the presence bitmap should be used for
// the specified field.
func UsePresenceForField(fd protoreflect.FieldDescriptor) (usePresence, canBeLazy bool) {
	switch {
	case fd.ContainingOneof() != nil && !fd.ContainingOneof().IsSynthetic():
		// Oneof fields never use the presence bitmap.
		//
		// Synthetic oneofs are an exception: Those are used to implement proto3
		// optional fields and hence should follow non-oneof field semantics.
		return false, false

	case fd.IsMap():
		// Map-typed fields never use the presence bitmap.
		return false, false

	case fd.Kind() == protoreflect.MessageKind || fd.Kind() == protoreflect.GroupKind:
		// Lazy fields always use the presence bitmap (only messages can be lazy).
		isLazy := fd.(interface{ IsLazy() bool }).IsLazy()
		return isLazy, isLazy

	default:
		// If the field has presence, use the presence bitmap.
		return fd.HasPresence(), false
	}
}
