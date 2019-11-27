// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package socket

func (c *Conn) recvMsgs(ms []Message, flags int) (int, error) {
	return 0, errNotImplemented
}

func (c *Conn) sendMsgs(ms []Message, flags int) (int, error) {
	return 0, errNotImplemented
}
