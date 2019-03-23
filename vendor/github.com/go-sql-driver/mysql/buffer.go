// Go MySQL Driver - A MySQL-Driver for Go's database/sql package
//
// Copyright 2013 The Go-MySQL-Driver Authors. All rights reserved.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at http://mozilla.org/MPL/2.0/.

package mysql

import (
	"io"
	"net"
	"time"
)

const defaultBufSize = 4096

// A buffer which is used for both reading and writing.
// This is possible since communication on each connection is synchronous.
// In other words, we can't write and read simultaneously on the same connection.
// The buffer is similar to bufio.Reader / Writer but zero-copy-ish
// Also highly optimized for this particular use case.
type buffer struct {
	buf     []byte
	nc      net.Conn
	idx     int
	length  int
	timeout time.Duration
}

func newBuffer(nc net.Conn) buffer {
	var b [defaultBufSize]byte
	return buffer{
		buf: b[:],
		nc:  nc,
	}
}

// fill reads into the buffer until at least _need_ bytes are in it
func (b *buffer) fill(need int) error {
	n := b.length

	// move existing data to the beginning
	if n > 0 && b.idx > 0 {
		copy(b.buf[0:n], b.buf[b.idx:])
	}

	// grow buffer if necessary
	// TODO: let the buffer shrink again at some point
	//       Maybe keep the org buf slice and swap back?
	if need > len(b.buf) {
		// Round up to the next multiple of the default size
		newBuf := make([]byte, ((need/defaultBufSize)+1)*defaultBufSize)
		copy(newBuf, b.buf)
		b.buf = newBuf
	}

	b.idx = 0

	for {
		if b.timeout > 0 {
			if err := b.nc.SetReadDeadline(time.Now().Add(b.timeout)); err != nil {
				return err
			}
		}

		nn, err := b.nc.Read(b.buf[n:])
		n += nn

		switch err {
		case nil:
			if n < need {
				continue
			}
			b.length = n
			return nil

		case io.EOF:
			if n >= need {
				b.length = n
				return nil
			}
			return io.ErrUnexpectedEOF

		default:
			return err
		}
	}
}

// returns next N bytes from buffer.
// The returned slice is only guaranteed to be valid until the next read
func (b *buffer) readNext(need int) ([]byte, error) {
	if b.length < need {
		// refill
		if err := b.fill(need); err != nil {
			return nil, err
		}
	}

	offset := b.idx
	b.idx += need
	b.length -= need
	return b.buf[offset:b.idx], nil
}

// returns a buffer with the requested size.
// If possible, a slice from the existing buffer is returned.
// Otherwise a bigger buffer is made.
// Only one buffer (total) can be used at a time.
func (b *buffer) takeBuffer(length int) []byte {
	if b.length > 0 {
		return nil
	}

	// test (cheap) general case first
	if length <= defaultBufSize || length <= cap(b.buf) {
		return b.buf[:length]
	}

	if length < maxPacketSize {
		b.buf = make([]byte, length)
		return b.buf
	}
	return make([]byte, length)
}

// shortcut which can be used if the requested buffer is guaranteed to be
// smaller than defaultBufSize
// Only one buffer (total) can be used at a time.
func (b *buffer) takeSmallBuffer(length int) []byte {
	if b.length == 0 {
		return b.buf[:length]
	}
	return nil
}

// takeCompleteBuffer returns the complete existing buffer.
// This can be used if the necessary buffer size is unknown.
// Only one buffer (total) can be used at a time.
func (b *buffer) takeCompleteBuffer() []byte {
	if b.length == 0 {
		return b.buf
	}
	return nil
}
