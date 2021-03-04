// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fieldsort defines an ordering of fields.
//
// The ordering defined by this package matches the historic behavior of the proto
// package, placing extensions first and oneofs last.
//
// There is no guarantee about stability of the wire encoding, and users should not
// depend on the order defined in this package as it is subject to change without
// notice.
package fieldsort

import (
	"google.golang.org/protobuf/reflect/protoreflect"
)

// Less returns true if field a comes before field j in ordered wire marshal output.
func Less(a, b protoreflect.FieldDescriptor) bool {
	ea := a.IsExtension()
	eb := b.IsExtension()
	oa := a.ContainingOneof()
	ob := b.ContainingOneof()
	switch {
	case ea != eb:
		return ea
	case oa != nil && ob != nil:
		if oa == ob {
			return a.Number() < b.Number()
		}
		return oa.Index() < ob.Index()
	case oa != nil && !oa.IsSynthetic():
		return false
	case ob != nil && !ob.IsSynthetic():
		return true
	default:
		return a.Number() < b.Number()
	}
}
