// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package driver contains interfaces that needs to be implemented by
// various GPIO implementations.
package driver // import "golang.org/x/exp/io/gpio/driver"

// Direction determines the direction of the pin. A pin could be
// configured to be an input or an output.
type Direction string

const (
	In  = Direction("in")
	Out = Direction("out")
	// TODO(jbd): Bidirectional pins?
)

// Opener is an interface to be implemented by GPIO drivers.
type Opener interface {
	Open() (Conn, error)
}

// Conn represents an open GPIO connection. Each driver should implement
// this interface to provide a full implementation of the GPIO protocol.
type Conn interface {
	// Value returns the value of the pin. 0 for low values, 1 for high.
	Value(pin int) (int, error)

	// SetValue sets the value of the pin. 0 for low values, 1 for high.
	SetValue(pin int, v int) error

	// SetDirection sets the direction of the pin.
	SetDirection(pin int, dir Direction) error

	// Close closes the connection and frees the underlying resources.
	Close() error
}
