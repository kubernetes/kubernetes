// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux
// +build !linux

package ipv4

const sizeofICMPFilter = 0x0

type icmpFilter struct {
}

func (f *icmpFilter) accept(typ ICMPType) {
}

func (f *icmpFilter) block(typ ICMPType) {
}

func (f *icmpFilter) setAll(block bool) {
}

func (f *icmpFilter) willBlock(typ ICMPType) bool {
	return false
}
