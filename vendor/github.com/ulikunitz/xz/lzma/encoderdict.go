// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"fmt"
	"io"
)

// matcher is an interface that supports the identification of the next
// operation.
type matcher interface {
	io.Writer
	SetDict(d *encoderDict)
	NextOp(rep [4]uint32) operation
}

// encoderDict provides the dictionary of the encoder. It includes an
// addtional buffer atop of the actual dictionary.
type encoderDict struct {
	buf      buffer
	m        matcher
	head     int64
	capacity int
	// preallocated array
	data [maxMatchLen]byte
}

// newEncoderDict creates the encoder dictionary. The argument bufSize
// defines the size of the additional buffer.
func newEncoderDict(dictCap, bufSize int, m matcher) (d *encoderDict, err error) {
	if !(1 <= dictCap && int64(dictCap) <= MaxDictCap) {
		return nil, errors.New(
			"lzma: dictionary capacity out of range")
	}
	if bufSize < 1 {
		return nil, errors.New(
			"lzma: buffer size must be larger than zero")
	}
	d = &encoderDict{
		buf:      *newBuffer(dictCap + bufSize),
		capacity: dictCap,
		m:        m,
	}
	m.SetDict(d)
	return d, nil
}

// Discard discards n bytes. Note that n must not be larger than
// MaxMatchLen.
func (d *encoderDict) Discard(n int) {
	p := d.data[:n]
	k, _ := d.buf.Read(p)
	if k < n {
		panic(fmt.Errorf("lzma: can't discard %d bytes", n))
	}
	d.head += int64(n)
	d.m.Write(p)
}

// Len returns the data available in the encoder dictionary.
func (d *encoderDict) Len() int {
	n := d.buf.Available()
	if int64(n) > d.head {
		return int(d.head)
	}
	return n
}

// DictLen returns the actual length of data in the dictionary.
func (d *encoderDict) DictLen() int {
	if d.head < int64(d.capacity) {
		return int(d.head)
	}
	return d.capacity
}

// Available returns the number of bytes that can be written by a
// following Write call.
func (d *encoderDict) Available() int {
	return d.buf.Available() - d.DictLen()
}

// Write writes data into the dictionary buffer. Note that the position
// of the dictionary head will not be moved. If there is not enough
// space in the buffer ErrNoSpace will be returned.
func (d *encoderDict) Write(p []byte) (n int, err error) {
	m := d.Available()
	if len(p) > m {
		p = p[:m]
		err = ErrNoSpace
	}
	var e error
	if n, e = d.buf.Write(p); e != nil {
		err = e
	}
	return n, err
}

// Pos returns the position of the head.
func (d *encoderDict) Pos() int64 { return d.head }

// ByteAt returns the byte at the given distance.
func (d *encoderDict) ByteAt(distance int) byte {
	if !(0 < distance && distance <= d.Len()) {
		return 0
	}
	i := d.buf.rear - distance
	if i < 0 {
		i += len(d.buf.data)
	}
	return d.buf.data[i]
}

// CopyN copies the last n bytes from the dictionary into the provided
// writer. This is used for copying uncompressed data into an
// uncompressed segment.
func (d *encoderDict) CopyN(w io.Writer, n int) (written int, err error) {
	if n <= 0 {
		return 0, nil
	}
	m := d.Len()
	if n > m {
		n = m
		err = ErrNoSpace
	}
	i := d.buf.rear - n
	var e error
	if i < 0 {
		i += len(d.buf.data)
		if written, e = w.Write(d.buf.data[i:]); e != nil {
			return written, e
		}
		i = 0
	}
	var k int
	k, e = w.Write(d.buf.data[i:d.buf.rear])
	written += k
	if e != nil {
		err = e
	}
	return written, err
}

// Buffered returns the number of bytes in the buffer.
func (d *encoderDict) Buffered() int { return d.buf.Buffered() }
