// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package spi allows users to read from and write to an SPI device.
package spi // import "golang.org/x/exp/io/spi"

import (
	"time"

	"golang.org/x/exp/io/spi/driver"
)

// Mode represents the SPI mode number where clock parity (CPOL)
// is the high order and clock edge (CPHA) is the low order bit.
type Mode int

const (
	Mode0 = Mode(0)
	Mode1 = Mode(1)
	Mode2 = Mode(2)
	Mode3 = Mode(3)
)

// Order is the bit justification to be used while transfering
// words to the SPI device. MSB-first encoding is more popular
// than LSB-first.
type Order int

const (
	MSBFirst = Order(0)
	LSBFirst = Order(1)
)

type Device struct {
	conn driver.Conn
}

// SetMode sets the SPI mode. SPI mode is a combination of polarity and phases.
// CPOL is the high order bit, CPHA is the low order. Pre-computed mode
// values are Mode0, Mode1, Mode2 and Mode3.
// The value can be changed by SPI device's driver.
func (d *Device) SetMode(mode Mode) error {
	return d.conn.Configure(driver.Mode, int(mode))
}

// SetMaxSpeed sets the maximum clock speed in Hz.
// The value can be overriden by SPI device's driver.
func (d *Device) SetMaxSpeed(speed int) error {
	return d.conn.Configure(driver.MaxSpeed, speed)
}

// SetBitsPerWord sets how many bits it takes to represent a word, e.g. 8 represents 8-bit words.
// The default is 8 bits per word.
func (d *Device) SetBitsPerWord(bits int) error {
	return d.conn.Configure(driver.Bits, bits)
}

// SetBitOrder sets the bit justification used to transfer SPI words.
// Valid values are MSBFirst and LSBFirst.
func (d *Device) SetBitOrder(o Order) error {
	return d.conn.Configure(driver.Order, int(o))
}

// SetDelay sets the amount of pause will be added after each frame write.
func (d *Device) SetDelay(t time.Duration) error {
	return d.conn.Configure(driver.Delay, int(t.Nanoseconds()/1000))
}

// Tx performs a duplex transmission to write w to the SPI device
// and read len(r) bytes to r.
// User should not mutate the w and r until this call returns.
func (d *Device) Tx(w, r []byte) error {
	// TODO(jbd): Allow nil w.
	return d.conn.Tx(w, r)
}

// Open opens a device with the specified bus and chip select
// by using the given driver. If a nil driver is provided,
// the default driver (devfs) is used.

func Open(o driver.Opener) (*Device, error) {
	conn, err := o.Open()
	if err != nil {
		return nil, err
	}
	return &Device{conn: conn}, nil
}

// Close closes the SPI device and releases the related resources.
func (d *Device) Close() error {
	return d.conn.Close()
}
