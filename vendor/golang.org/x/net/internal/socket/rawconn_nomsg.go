// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9
// +build !darwin,!dragonfly,!freebsd,!linux,!netbsd,!openbsd,!solaris,!windows

package socket

import "errors"

func (c *Conn) recvMsg(m *Message, flags int) error {
	return errors.New("not implemented")
}

func (c *Conn) sendMsg(m *Message, flags int) error {
	return errors.New("not implemented")
}
