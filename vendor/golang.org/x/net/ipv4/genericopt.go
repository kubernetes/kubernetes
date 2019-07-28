// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv4

// TOS returns the type-of-service field value for outgoing packets.
func (c *genericOpt) TOS() (int, error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	so, ok := sockOpts[ssoTOS]
	if !ok {
		return 0, errOpNoSupport
	}
	return so.GetInt(c.Conn)
}

// SetTOS sets the type-of-service field value for future outgoing
// packets.
func (c *genericOpt) SetTOS(tos int) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoTOS]
	if !ok {
		return errOpNoSupport
	}
	return so.SetInt(c.Conn, tos)
}

// TTL returns the time-to-live field value for outgoing packets.
func (c *genericOpt) TTL() (int, error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	so, ok := sockOpts[ssoTTL]
	if !ok {
		return 0, errOpNoSupport
	}
	return so.GetInt(c.Conn)
}

// SetTTL sets the time-to-live field value for future outgoing
// packets.
func (c *genericOpt) SetTTL(ttl int) error {
	if !c.ok() {
		return errInvalidConn
	}
	so, ok := sockOpts[ssoTTL]
	if !ok {
		return errOpNoSupport
	}
	return so.SetInt(c.Conn, ttl)
}
