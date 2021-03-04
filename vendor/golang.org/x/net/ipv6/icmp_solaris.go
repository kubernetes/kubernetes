// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

func (f *icmpv6Filter) accept(typ ICMPType) {
	f.X__icmp6_filt[typ>>5] |= 1 << (uint32(typ) & 31)
}

func (f *icmpv6Filter) block(typ ICMPType) {
	f.X__icmp6_filt[typ>>5] &^= 1 << (uint32(typ) & 31)
}

func (f *icmpv6Filter) setAll(block bool) {
	for i := range f.X__icmp6_filt {
		if block {
			f.X__icmp6_filt[i] = 0
		} else {
			f.X__icmp6_filt[i] = 1<<32 - 1
		}
	}
}

func (f *icmpv6Filter) willBlock(typ ICMPType) bool {
	return f.X__icmp6_filt[typ>>5]&(1<<(uint32(typ)&31)) == 0
}
