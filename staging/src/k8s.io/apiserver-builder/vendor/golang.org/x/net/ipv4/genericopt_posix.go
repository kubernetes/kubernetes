// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows

package ipv4

import "syscall"

// TOS returns the type-of-service field value for outgoing packets.
func (c *genericOpt) TOS() (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	fd, err := c.sysfd()
	if err != nil {
		return 0, err
	}
	return getInt(fd, &sockOpts[ssoTOS])
}

// SetTOS sets the type-of-service field value for future outgoing
// packets.
func (c *genericOpt) SetTOS(tos int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	fd, err := c.sysfd()
	if err != nil {
		return err
	}
	return setInt(fd, &sockOpts[ssoTOS], tos)
}

// TTL returns the time-to-live field value for outgoing packets.
func (c *genericOpt) TTL() (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	fd, err := c.sysfd()
	if err != nil {
		return 0, err
	}
	return getInt(fd, &sockOpts[ssoTTL])
}

// SetTTL sets the time-to-live field value for future outgoing
// packets.
func (c *genericOpt) SetTTL(ttl int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	fd, err := c.sysfd()
	if err != nil {
		return err
	}
	return setInt(fd, &sockOpts[ssoTTL], ttl)
}
