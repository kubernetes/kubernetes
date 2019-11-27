// Copyright (c) 2013, Suryandaru Triandana <syndtr@gmail.com>
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Package capability provides utilities for manipulating POSIX capabilities.
package capability

type Capabilities interface {
	// Get check whether a capability present in the given
	// capabilities set. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE, BOUNDING or AMBIENT.
	Get(which CapType, what Cap) bool

	// Empty check whether all capability bits of the given capabilities
	// set are zero. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE, BOUNDING or AMBIENT.
	Empty(which CapType) bool

	// Full check whether all capability bits of the given capabilities
	// set are one. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE, BOUNDING or AMBIENT.
	Full(which CapType) bool

	// Set sets capabilities of the given capabilities sets. The
	// 'which' value should be one or combination (OR'ed) of EFFECTIVE,
	// PERMITTED, INHERITABLE, BOUNDING or AMBIENT.
	Set(which CapType, caps ...Cap)

	// Unset unsets capabilities of the given capabilities sets. The
	// 'which' value should be one or combination (OR'ed) of EFFECTIVE,
	// PERMITTED, INHERITABLE, BOUNDING or AMBIENT.
	Unset(which CapType, caps ...Cap)

	// Fill sets all bits of the given capabilities kind to one. The
	// 'kind' value should be one or combination (OR'ed) of CAPS,
	// BOUNDS or AMBS.
	Fill(kind CapType)

	// Clear sets all bits of the given capabilities kind to zero. The
	// 'kind' value should be one or combination (OR'ed) of CAPS,
	// BOUNDS or AMBS.
	Clear(kind CapType)

	// String return current capabilities state of the given capabilities
	// set as string. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE BOUNDING or AMBIENT
	StringCap(which CapType) string

	// String return current capabilities state as string.
	String() string

	// Load load actual capabilities value. This will overwrite all
	// outstanding changes.
	Load() error

	// Apply apply the capabilities settings, so all changes will take
	// effect.
	Apply(kind CapType) error
}

// NewPid initializes a new Capabilities object for given pid when
// it is nonzero, or for the current process if pid is 0.
//
// Deprecated: Replace with NewPid2.  For example, replace:
//
//    c, err := NewPid(0)
//    if err != nil {
//      return err
//    }
//
// with:
//
//    c, err := NewPid2(0)
//    if err != nil {
//      return err
//    }
//    err = c.Load()
//    if err != nil {
//      return err
//    }
func NewPid(pid int) (Capabilities, error) {
	c, err := newPid(pid)
	if err != nil {
		return c, err
	}
	err = c.Load()
	return c, err
}

// NewPid2 initializes a new Capabilities object for given pid when
// it is nonzero, or for the current process if pid is 0.  This
// does not load the process's current capabilities; to do that you
// must call Load explicitly.
func NewPid2(pid int) (Capabilities, error) {
	return newPid(pid)
}

// NewFile initializes a new Capabilities object for given file path.
//
// Deprecated: Replace with NewFile2.  For example, replace:
//
//    c, err := NewFile(path)
//    if err != nil {
//      return err
//    }
//
// with:
//
//    c, err := NewFile2(path)
//    if err != nil {
//      return err
//    }
//    err = c.Load()
//    if err != nil {
//      return err
//    }
func NewFile(path string) (Capabilities, error) {
	c, err := newFile(path)
	if err != nil {
		return c, err
	}
	err = c.Load()
	return c, err
}

// NewFile2 creates a new initialized Capabilities object for given
// file path.  This does not load the process's current capabilities;
// to do that you must call Load explicitly.
func NewFile2(path string) (Capabilities, error) {
	return newFile(path)
}
