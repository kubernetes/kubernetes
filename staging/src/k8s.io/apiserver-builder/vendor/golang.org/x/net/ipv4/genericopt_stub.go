// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl plan9 solaris

package ipv4

// TOS returns the type-of-service field value for outgoing packets.
func (c *genericOpt) TOS() (int, error) {
	return 0, errOpNoSupport
}

// SetTOS sets the type-of-service field value for future outgoing
// packets.
func (c *genericOpt) SetTOS(tos int) error {
	return errOpNoSupport
}

// TTL returns the time-to-live field value for outgoing packets.
func (c *genericOpt) TTL() (int, error) {
	return 0, errOpNoSupport
}

// SetTTL sets the time-to-live field value for future outgoing
// packets.
func (c *genericOpt) SetTTL(ttl int) error {
	return errOpNoSupport
}
