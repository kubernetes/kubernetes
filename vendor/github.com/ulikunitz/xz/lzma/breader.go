// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"io"
)

// breader provides the ReadByte function for a Reader. It doesn't read
// more data from the reader than absolutely necessary.
type breader struct {
	io.Reader
	// helper slice to save allocations
	p []byte
}

// ByteReader converts an io.Reader into an io.ByteReader.
func ByteReader(r io.Reader) io.ByteReader {
	br, ok := r.(io.ByteReader)
	if !ok {
		return &breader{r, make([]byte, 1)}
	}
	return br
}

// ReadByte read byte function.
func (r *breader) ReadByte() (c byte, err error) {
	n, err := r.Reader.Read(r.p)
	if n < 1 {
		if err == nil {
			err = errors.New("breader.ReadByte: no data")
		}
		return 0, err
	}
	return r.p[0], nil
}
