// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !aix,!darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris,!zos

package socket

type msghdr struct{}

func (h *msghdr) pack(vs []iovec, bs [][]byte, oob []byte, sa []byte) {}
func (h *msghdr) name() []byte                                        { return nil }
func (h *msghdr) controllen() int                                     { return 0 }
func (h *msghdr) flags() int                                          { return 0 }
