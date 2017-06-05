// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package ipv4

const sysSizeofICMPFilter = 0x0

type sysICMPFilter struct {
}

func (f *sysICMPFilter) accept(typ ICMPType) {
}

func (f *sysICMPFilter) block(typ ICMPType) {
}

func (f *sysICMPFilter) setAll(block bool) {
}

func (f *sysICMPFilter) willBlock(typ ICMPType) bool {
	return false
}
