// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ipv6

import "syscall"

// TrafficClass returns the traffic class field value for outgoing
// packets.
func (c *genericOpt) TrafficClass() (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	so, ok := sockOpts[ssoTrafficClass]
	if !ok {
		return 0, errOpNoSupport
	}
	return so.GetInt(c.Conn)
}

// SetTrafficClass sets the traffic class field value for future
// outgoing packets.
func (c *genericOpt) SetTrafficClass(tclass int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	so, ok := sockOpts[ssoTrafficClass]
	if !ok {
		return errOpNoSupport
	}
	return so.SetInt(c.Conn, tclass)
}

// HopLimit returns the hop limit field value for outgoing packets.
func (c *genericOpt) HopLimit() (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	so, ok := sockOpts[ssoHopLimit]
	if !ok {
		return 0, errOpNoSupport
	}
	return so.GetInt(c.Conn)
}

// SetHopLimit sets the hop limit field value for future outgoing
// packets.
func (c *genericOpt) SetHopLimit(hoplim int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	so, ok := sockOpts[ssoHopLimit]
	if !ok {
		return errOpNoSupport
	}
	return so.SetInt(c.Conn, hoplim)
}
