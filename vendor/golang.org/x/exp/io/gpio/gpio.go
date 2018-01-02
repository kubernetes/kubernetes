// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gpio allows users to communicate with GPIO pins.
package gpio // import "golang.org/x/exp/io/gpio"

import "golang.org/x/exp/io/gpio/driver"

type Device struct {
	conn driver.Conn
}

// Direction determines the direction of the pin. A pin could be
// configured to be an input or an output.
type Direction string

const (
	In  = Direction("in")
	Out = Direction("out")
	// TODO(jbd): Out but initially high or initially low?
)

// TODO(jbd): Allow users to set active value.
// TODO(jbd): How to support analog pins?

// TODO(jbd): Support interrupts.
// TODO(jbd): Allow users to configure edge trigger type.

// Open opens a connection to the GPIO device with the given driver.
// Opened devices should be closed by calling Close.
func Open(d driver.Opener) (*Device, error) {
	// TODO(jbd): Open pin rather than GPIO device? It would help
	// some driver implementations such as sysfs.
	// TODO(jbd): Default to devfs if nothing is set?
	conn, err := d.Open()
	if err != nil {
		return nil, err
	}
	return &Device{conn: conn}, nil
}

// Value returns the value for the pin. 0 for low, 1 for high values.
func (d *Device) Value(pin int) (int, error) {
	return d.conn.Value(pin)
}

// SetValue sets the value of the pin. 0 for low, 1 for high values.
func (d *Device) SetValue(pin int, v int) error {
	return d.conn.SetValue(pin, v)
}

// SetDirection configures the direction of the pin.
func (d *Device) SetDirection(pin int, dir Direction) error {
	return d.conn.SetDirection(pin, driver.Direction(dir))
}

// Close closes the device and frees the resources.
func (d *Device) Close() error {
	return d.conn.Close()
}
