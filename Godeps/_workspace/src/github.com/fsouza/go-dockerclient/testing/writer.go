// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"encoding/binary"
	"errors"
	"io"
)

type stdType [8]byte

var (
	stdin  = stdType{0: 0}
	stdout = stdType{0: 1}
	stderr = stdType{0: 2}
)

type stdWriter struct {
	io.Writer
	prefix  stdType
	sizeBuf []byte
}

func (w *stdWriter) Write(buf []byte) (n int, err error) {
	if w == nil || w.Writer == nil {
		return 0, errors.New("Writer not instanciated")
	}
	binary.BigEndian.PutUint32(w.prefix[4:], uint32(len(buf)))
	buf = append(w.prefix[:], buf...)

	n, err = w.Writer.Write(buf)
	return n - 8, err
}

func newStdWriter(w io.Writer, t stdType) *stdWriter {
	if len(t) != 8 {
		return nil
	}
	return &stdWriter{Writer: w, prefix: t, sizeBuf: make([]byte, 4)}
}
