// Copyright 2014-2017 Ulrich Kunitz. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzma

import (
	"errors"
	"io"
)

// ErrLimit indicates that the limit of the LimitedByteWriter has been
// reached.
var ErrLimit = errors.New("limit reached")

// LimitedByteWriter provides a byte writer that can be written until a
// limit is reached. The field N provides the number of remaining
// bytes.
type LimitedByteWriter struct {
	BW io.ByteWriter
	N  int64
}

// WriteByte writes a single byte to the limited byte writer. It returns
// ErrLimit if the limit has been reached. If the byte is successfully
// written the field N of the LimitedByteWriter will be decremented by
// one.
func (l *LimitedByteWriter) WriteByte(c byte) error {
	if l.N <= 0 {
		return ErrLimit
	}
	if err := l.BW.WriteByte(c); err != nil {
		return err
	}
	l.N--
	return nil
}
