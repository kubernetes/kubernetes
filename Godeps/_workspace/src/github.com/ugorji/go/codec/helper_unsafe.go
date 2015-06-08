//+build unsafe

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

import (
	"unsafe"
)

// This file has unsafe variants of some helper methods.

type unsafeString struct {
	Data uintptr
	Len  int
}

type unsafeBytes struct {
	Data uintptr
	Len  int
	Cap  int
}

// stringView returns a view of the []byte as a string.
// In unsafe mode, it doesn't incur allocation and copying caused by conversion.
// In regular safe mode, it is an allocation and copy.
func stringView(v []byte) string {
	x := unsafeString{uintptr(unsafe.Pointer(&v[0])), len(v)}
	return *(*string)(unsafe.Pointer(&x))
}

// bytesView returns a view of the string as a []byte.
// In unsafe mode, it doesn't incur allocation and copying caused by conversion.
// In regular safe mode, it is an allocation and copy.
func bytesView(v string) []byte {
	x := unsafeBytes{uintptr(unsafe.Pointer(&v)), len(v), len(v)}
	return *(*[]byte)(unsafe.Pointer(&x))
}
