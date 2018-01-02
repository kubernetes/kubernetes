// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package spi_test

import "golang.org/x/exp/io/spi"

// Example illustrates a program that drives an APA-102 LED strip.
func Example() {
	dev, err := spi.Open(&spi.Devfs{
		Dev:      "/dev/spidev0.1",
		Mode:     spi.Mode3,
		MaxSpeed: 500000,
	})
	if err != nil {
		panic(err)
	}
	defer dev.Close()

	if err := dev.Tx([]byte{
		0, 0, 0, 0,
		0xff, 200, 0, 200,
		0xff, 200, 0, 200,
		0xe0, 200, 0, 200,
		0xff, 200, 0, 200,
		0xff, 8, 50, 0,
		0xff, 200, 0, 0,
		0xff, 0, 0, 0,
		0xff, 200, 0, 200,
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
	}, nil); err != nil {
		panic(err)
	}
}
