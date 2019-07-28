// Copyright 2016 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"errors"
	"fmt"

	guuid "github.com/google/uuid"
)

// MarshalText implements encoding.TextMarshaler.
func (u UUID) MarshalText() ([]byte, error) {
	if len(u) != 16 {
		return nil, nil
	}
	var js [36]byte
	encodeHex(js[:], u)
	return js[:], nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (u *UUID) UnmarshalText(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	id := Parse(string(data))
	if id == nil {
		return errors.New("invalid UUID")
	}
	*u = id
	return nil
}

// MarshalBinary implements encoding.BinaryMarshaler.
func (u UUID) MarshalBinary() ([]byte, error) {
	return u[:], nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (u *UUID) UnmarshalBinary(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	if len(data) != 16 {
		return fmt.Errorf("invalid UUID (got %d bytes)", len(data))
	}
	var id [16]byte
	copy(id[:], data)
	*u = id[:]
	return nil
}

// MarshalText implements encoding.TextMarshaler.
func (u Array) MarshalText() ([]byte, error) {
	var js [36]byte
	encodeHex(js[:], u[:])
	return js[:], nil
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (u *Array) UnmarshalText(data []byte) error {
	id, err := guuid.ParseBytes(data)
	if err != nil {
		return err
	}
	*u = Array(id)
	return nil
}

// MarshalBinary implements encoding.BinaryMarshaler.
func (u Array) MarshalBinary() ([]byte, error) {
	return u[:], nil
}

// UnmarshalBinary implements encoding.BinaryUnmarshaler.
func (u *Array) UnmarshalBinary(data []byte) error {
	if len(data) != 16 {
		return fmt.Errorf("invalid UUID (got %d bytes)", len(data))
	}
	copy(u[:], data)
	return nil
}
