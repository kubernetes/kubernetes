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
	// PERMITTED, INHERITABLE or BOUNDING.
	Get(which CapType, what Cap) bool

	// Empty check whether all capability bits of the given capabilities
	// set are zero. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE or BOUNDING.
	Empty(which CapType) bool

	// Full check whether all capability bits of the given capabilities
	// set are one. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE or BOUNDING.
	Full(which CapType) bool

	// Set sets capabilities of the given capabilities sets. The
	// 'which' value should be one or combination (OR'ed) of EFFECTIVE,
	// PERMITTED, INHERITABLE or BOUNDING.
	Set(which CapType, caps ...Cap)

	// Unset unsets capabilities of the given capabilities sets. The
	// 'which' value should be one or combination (OR'ed) of EFFECTIVE,
	// PERMITTED, INHERITABLE or BOUNDING.
	Unset(which CapType, caps ...Cap)

	// Fill sets all bits of the given capabilities kind to one. The
	// 'kind' value should be one or combination (OR'ed) of CAPS or
	// BOUNDS.
	Fill(kind CapType)

	// Clear sets all bits of the given capabilities kind to zero. The
	// 'kind' value should be one or combination (OR'ed) of CAPS or
	// BOUNDS.
	Clear(kind CapType)

	// String return current capabilities state of the given capabilities
	// set as string. The 'which' value should be one of EFFECTIVE,
	// PERMITTED, INHERITABLE or BOUNDING.
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

// NewPid create new initialized Capabilities object for given pid.
func NewPid(pid int) (Capabilities, error) {
	return newPid(pid)
}

// NewFile create new initialized Capabilities object for given named file.
func NewFile(name string) (Capabilities, error) {
	return newFile(name)
}
