// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package i2c

import (
	"errors"

	"golang.org/x/exp/io/i2c/driver"
)

// Devfs is no-implementation so developers using cross compilation
// can rely on local tools even though the real implementation isn't
// available on their platform.
type Devfs struct {
	Dev  string
	Addr int
}

func (d *Devfs) Open() (driver.Conn, error) {
	return nil, errors.New("not implemented on this platform")
}
