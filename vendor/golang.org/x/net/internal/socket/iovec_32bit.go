// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm mips mipsle 386
// +build darwin dragonfly freebsd linux netbsd openbsd

package socket

import "unsafe"

func (v *iovec) set(b []byte) {
	v.Base = (*byte)(unsafe.Pointer(&b[0]))
	v.Len = uint32(len(b))
}
