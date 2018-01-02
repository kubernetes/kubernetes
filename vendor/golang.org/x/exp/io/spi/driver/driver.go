// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package driver contains interfaces to be implemented by various SPI implementations.
package driver // import "golang.org/x/exp/io/spi/driver"

const (
	Mode = iota
	Bits
	MaxSpeed
	Order
	Delay
)

// Opener is an interface to be implemented by the SPI driver to open
// a connection to an SPI device.
type Opener interface {
	Open() (Conn, error)
}

// Conn is a connection to an SPI device.
// TODO(jbd): Extend the interface to query configuration values.
type Conn interface {
	// Configure configures the SPI device.
	//
	// Available configuration keys are:
	//  - Mode, the SPI mode (valid values are 0, 1, 2 and 3).
	//  - Bits, bits per word (default is 8-bit per word).
	//  - Speed, the max clock speed (in Hz).
	//  - Order, bit order to be used in transfers. Zero value represents
	//    the MSB-first, non-zero values represent LSB-first encoding.
	//  - Delay, the pause time between frames (in usecs).
	//    Some SPI devices require a minimum amount of wait time after
	//    each frame write. If set, Delay amount of usecs are inserted after
	//    each write.
	//
	// SPI devices can override these values.
	Configure(k, v int) error

	// Tx performs a SPI transaction: w is written if not nil, the result is
	// put into r if not nil. len(w) must be equal to len(r), otherwise the
	// driver should return an error.
	Tx(w, r []byte) error

	// Close frees the underlying resources and closes the connection.
	Close() error
}
