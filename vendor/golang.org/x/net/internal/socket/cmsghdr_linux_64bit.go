// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (arm64 || amd64 || ppc64 || ppc64le || mips64 || mips64le || riscv64 || s390x) && linux
// +build arm64 amd64 ppc64 ppc64le mips64 mips64le riscv64 s390x
// +build linux

package socket

func (h *cmsghdr) set(l, lvl, typ int) {
	h.Len = uint64(l)
	h.Level = int32(lvl)
	h.Type = int32(typ)
}
