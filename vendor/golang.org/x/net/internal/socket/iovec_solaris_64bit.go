// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64
// +build solaris

package socket

import "unsafe"

func (v *iovec) set(b []byte) {
	v.Base = (*int8)(unsafe.Pointer(&b[0]))
	v.Len = uint64(len(b))
}
