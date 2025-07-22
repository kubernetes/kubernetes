// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package genid

// Go names of implementation-specific struct fields in generated messages.
const (
	State_goname = "state"

	SizeCache_goname  = "sizeCache"
	SizeCacheA_goname = "XXX_sizecache"

	UnknownFields_goname  = "unknownFields"
	UnknownFieldsA_goname = "XXX_unrecognized"

	ExtensionFields_goname  = "extensionFields"
	ExtensionFieldsA_goname = "XXX_InternalExtensions"
	ExtensionFieldsB_goname = "XXX_extensions"
)
