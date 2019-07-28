// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package ipv6

import (
	"net"
	"runtime"

	"golang.org/x/net/internal/socket"
)

// BUG(mikio): On Windows, the ReadBatch and WriteBatch methods of
// PacketConn are not implemented.

// A Message represents an IO message.
//
//	type Message struct {
//		Buffers [][]byte
//		OOB     []byte
//		Addr    net.Addr
//		N       int
//		NN      int
//		Flags   int
//	}
//
// The Buffers fields represents a list of contiguous buffers, which
// can be used for vectored IO, for example, putting a header and a
// payload in each slice.
// When writing, the Buffers field must contain at least one byte to
// write.
// When reading, the Buffers field will always contain a byte to read.
//
// The OOB field contains protocol-specific control or miscellaneous
// ancillary data known as out-of-band data.
// It can be nil when not required.
//
// The Addr field specifies a destination address when writing.
// It can be nil when the underlying protocol of the endpoint uses
// connection-oriented communication.
// After a successful read, it may contain the source address on the
// received packet.
//
// The N field indicates the number of bytes read or written from/to
// Buffers.
//
// The NN field indicates the number of bytes read or written from/to
// OOB.
//
// The Flags field contains protocol-specific information on the
// received message.
type Message = socket.Message

// ReadBatch reads a batch of messages.
//
// The provided flags is a set of platform-dependent flags, such as
// syscall.MSG_PEEK.
//
// On a successful read it returns the number of messages received, up
// to len(ms).
//
// On Linux, a batch read will be optimized.
// On other platforms, this method will read only a single message.
func (c *payloadHandler) ReadBatch(ms []Message, flags int) (int, error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	switch runtime.GOOS {
	case "linux":
		n, err := c.RecvMsgs([]socket.Message(ms), flags)
		if err != nil {
			err = &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
		return n, err
	default:
		n := 1
		err := c.RecvMsg(&ms[0], flags)
		if err != nil {
			n = 0
			err = &net.OpError{Op: "read", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
		return n, err
	}
}

// WriteBatch writes a batch of messages.
//
// The provided flags is a set of platform-dependent flags, such as
// syscall.MSG_DONTROUTE.
//
// It returns the number of messages written on a successful write.
//
// On Linux, a batch write will be optimized.
// On other platforms, this method will write only a single message.
func (c *payloadHandler) WriteBatch(ms []Message, flags int) (int, error) {
	if !c.ok() {
		return 0, errInvalidConn
	}
	switch runtime.GOOS {
	case "linux":
		n, err := c.SendMsgs([]socket.Message(ms), flags)
		if err != nil {
			err = &net.OpError{Op: "write", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
		return n, err
	default:
		n := 1
		err := c.SendMsg(&ms[0], flags)
		if err != nil {
			n = 0
			err = &net.OpError{Op: "write", Net: c.PacketConn.LocalAddr().Network(), Source: c.PacketConn.LocalAddr(), Err: err}
		}
		return n, err
	}
}
