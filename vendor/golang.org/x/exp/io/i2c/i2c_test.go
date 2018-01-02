// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package i2c

import (
	"testing"
)

func TestTenBit(t *testing.T) {
	tc := []struct {
		masked     int
		addrWant   int
		tenbitWant bool
	}{
		{TenBit(0x5), 0x5, true},
		{0x5, 0x5, false},
		{TenBit(0x200), 0x200, true},
	}

	for _, tt := range tc {
		unmasked, tenbit := ResolveAddr(tt.masked)
		if want, got := tt.tenbitWant, tenbit; got != want {
			t.Errorf("want address %b as 10-bit; got non 10-bit", want)
		}
		if want, got := tt.addrWant, unmasked; got != want {
			t.Errorf("want address %b; got %b", want, got)
		}
	}
}
