// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64 amd64 ppc64 ppc64le mips64 mips64le s390x
// +build linux

package socket

import "unsafe"

func (h *msghdr) setIov(vs []iovec) {
	h.Iov = &vs[0]
	h.Iovlen = uint64(len(vs))
}

func (h *msghdr) setControl(b []byte) {
	h.Control = (*byte)(unsafe.Pointer(&b[0]))
	h.Controllen = uint64(len(b))
}
