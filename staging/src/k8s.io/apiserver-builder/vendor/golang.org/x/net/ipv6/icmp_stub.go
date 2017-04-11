// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9

package ipv6

type sysICMPv6Filter struct {
}

func (f *sysICMPv6Filter) accept(typ ICMPType) {
}

func (f *sysICMPv6Filter) block(typ ICMPType) {
}

func (f *sysICMPv6Filter) setAll(block bool) {
}

func (f *sysICMPv6Filter) willBlock(typ ICMPType) bool {
	return false
}
