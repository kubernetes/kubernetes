// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package spi

import (
	"errors"

	"golang.org/x/exp/io/spi/driver"
)

// Devfs is a no-implementation of an SPI driver that works against the devfs.
// You need to have loaded the Linux "spidev" module to use this driver.
type Devfs struct {
	// Dev is the device to be opened.
	// Device name is usually in the /dev/spidev<bus>.<chip> format.
	// Required.
	Dev string

	// Mode is the SPI mode. SPI mode is a combination of polarity and phases.
	// CPOL is the high order bit, CPHA is the low order. Pre-computed mode
	// values are Mode0, Mode1, Mode2 and Mode3. The value of the mode argument
	// can be overriden by the device's driver.
	// Required.
	Mode Mode

	// MaxSpeed is the max clock speed (Hz) and can be overriden by the device's driver.
	// Required.
	MaxSpeed int64
}

// Open opens the provided device with the speicifed options
// and returns a connection.
func (d *Devfs) Open() (driver.Conn, error) {
	return nil, errors.New("not implemented on this platform")
}
