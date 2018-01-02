// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package i2c_test

import (
	"golang.org/x/exp/io/i2c"
)

func ExampleOpen() {
	d, err := i2c.Open(&i2c.Devfs{Dev: "/dev/i2c-1", Addr: 0x39})
	if err != nil {
		panic(err)
	}

	// opens a 10-bit address
	d, err = i2c.Open(&i2c.Devfs{Dev: "/dev/i2c-1", Addr: i2c.TenBit(0x78)})
	if err != nil {
		panic(err)
	}

	_ = d
}
