// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package i2c

import (
	"fmt"
	"io"
	"os"
	"syscall"

	"golang.org/x/exp/io/i2c/driver"
)

// Devfs is an I2C driver that works against the devfs.
// You need to load the "i2c-dev" kernel module to use this driver.
type Devfs struct {
	// Dev is the I2C bus device, e.g. /dev/i2c-1.
	// Required.
	Dev string

	// Addr is the device's I2C address on the specified bus.
	// Use TenBit to mark your address if your device works with 10-bit addresses.
	// Required.
	Addr int
}

const (
	i2c_SLAVE  = 0x0703 // TODO(jbd): Allow users to use I2C_SLAVE_FORCE?
	i2c_TENBIT = 0x0704
)

// TODO(jbd): Support I2C_RETRIES and I2C_TIMEOUT at the driver and implementation level.

func (d *Devfs) Open() (driver.Conn, error) {
	addr, tenbit := ResolveAddr(d.Addr)
	f, err := os.OpenFile(d.Dev, os.O_RDWR, os.ModeDevice)
	if err != nil {
		return nil, err
	}
	conn := &devfsConn{f: f}
	if tenbit {
		if err := conn.ioctl(i2c_TENBIT, uintptr(1)); err != nil {
			conn.Close()
			return nil, fmt.Errorf("cannot enable the 10-bit address mode on bus %v: %v", d.Dev, err)
		}
	}
	if err := conn.ioctl(i2c_SLAVE, uintptr(addr)); err != nil {
		conn.Close()
		return nil, fmt.Errorf("error opening the address (%v) on the bus (%v): %v", addr, d.Dev, err)
	}
	return conn, nil
}

type devfsConn struct {
	f *os.File
}

func (c *devfsConn) Tx(w, r []byte) error {
	if w != nil {
		if _, err := c.f.Write(w); err != nil {
			return err
		}
	}
	if r != nil {
		if _, err := io.ReadFull(c.f, r); err != nil {
			return err
		}
	}
	return nil
}

func (c *devfsConn) Close() error {
	return c.f.Close()
}

func (c *devfsConn) ioctl(arg1, arg2 uintptr) error {
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, c.f.Fd(), arg1, arg2); errno != 0 {
		return syscall.Errno(errno)
	}
	return nil
}
