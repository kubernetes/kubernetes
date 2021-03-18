// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !aix,!darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris,!zos

package socket

func controlHeaderLen() int {
	return 0
}

func controlMessageLen(dataLen int) int {
	return 0
}

func controlMessageSpace(dataLen int) int {
	return 0
}

type cmsghdr struct{}

func (h *cmsghdr) len() int { return 0 }
func (h *cmsghdr) lvl() int { return 0 }
func (h *cmsghdr) typ() int { return 0 }

func (h *cmsghdr) set(l, lvl, typ int) {}
