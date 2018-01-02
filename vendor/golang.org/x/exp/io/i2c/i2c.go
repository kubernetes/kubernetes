// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package i2c allows users to read from and write to a slave I2C device.
package i2c // import "golang.org/x/exp/io/i2c"

import (
	"golang.org/x/exp/io/i2c/driver"
)

const tenbitMask = 1 << 12

// Device represents an I2C device. Devices must be closed once
// they are no longer in use.
type Device struct {
	conn driver.Conn
}

// TenBit marks an I2C address as a 10-bit address.
func TenBit(addr int) int {
	return addr | tenbitMask
}

// Read reads len(buf) bytes from the device.
func (d *Device) Read(buf []byte) error {
	return d.conn.Tx(nil, buf)
}

// ReadReg is similar to Read but it reads from a register.
func (d *Device) ReadReg(reg byte, buf []byte) error {
	return d.conn.Tx([]byte{reg}, buf)
}

// Write writes the buffer to the device. If it is required to write to a
// specific register, the register should be passed as the first byte in the
// given buffer.
func (d *Device) Write(buf []byte) (err error) {
	return d.conn.Tx(buf, nil)
}

// WriteReg is similar to Write but writes to a register.
func (d *Device) WriteReg(reg byte, buf []byte) (err error) {
	// TODO(jbd): Do not allocate, not optimal.
	return d.conn.Tx(append([]byte{reg}, buf...), nil)
}

// Close closes the device and releases the underlying sources.
func (d *Device) Close() error {
	return d.conn.Close()
}

// Open opens a connection to an I2C device.
// All devices must be closed once they are no longer in use.
func Open(o driver.Opener) (*Device, error) {
	conn, err := o.Open()
	if err != nil {
		return nil, err
	}
	return &Device{conn: conn}, nil
}

// ResolveAddr returns whether the addr is 10-bit masked or not.
// It also returns the unmasked address.
func ResolveAddr(addr int) (unmasked int, tenbit bool) {
	return addr & (tenbitMask - 1), addr&tenbitMask == tenbitMask
}
