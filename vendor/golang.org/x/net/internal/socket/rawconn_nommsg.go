// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9
// +build !linux

package socket

import "errors"

func (c *Conn) recvMsgs(ms []Message, flags int) (int, error) {
	return 0, errors.New("not implemented")
}

func (c *Conn) sendMsgs(ms []Message, flags int) (int, error) {
	return 0, errors.New("not implemented")
}
