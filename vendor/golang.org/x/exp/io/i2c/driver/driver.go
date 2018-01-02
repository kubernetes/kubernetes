// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package driver contains interfaces to be implemented by various I2C implementations.
package driver // import "golang.org/x/exp/io/i2c/driver"

// Opener opens a connection to an I2C device.
type Opener interface {
	Open() (Conn, error)
}

// Conn represents an active connection to an I2C device.
type Conn interface {
	// Tx first writes w (if not nil), then reads len(r)
	// bytes from device into r (if not nil) in a single
	// I2C transaction.
	Tx(w, r []byte) error

	// Close closes the connection.
	Close() error
}
