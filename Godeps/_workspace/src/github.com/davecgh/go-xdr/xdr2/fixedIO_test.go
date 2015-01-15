/*
 * Copyright (c) 2014 Dave Collins <dave@davec.name>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

package xdr_test

import (
	"io"
)

// fixedWriter implements the io.Writer interface and intentially allows
// testing of error paths by forcing short writes.
type fixedWriter struct {
	b   []byte
	pos int
}

// Write writes the contents of p to w. When the contents of p would cause
// the writer to exceed the maximum allowed size of the fixed writer, the writer
// writes as many bytes as possible to reach the maximum allowed size and
// io.ErrShortWrite is returned.
//
// This satisfies the io.Writer interface.
func (w *fixedWriter) Write(p []byte) (int, error) {
	if w.pos+len(p) > cap(w.b) {
		n := copy(w.b[w.pos:], p[:cap(w.b)-w.pos])
		w.pos += n
		return n, io.ErrShortWrite
	}

	n := copy(w.b[w.pos:], p)
	w.pos += n
	return n, nil
}

// Bytes returns the bytes already written to the fixed writer.
func (w *fixedWriter) Bytes() []byte {
	return w.b
}

// newFixedWriter returns a new io.Writer that will error once more bytes than
// the specified max have been written.
func newFixedWriter(max int) *fixedWriter {
	b := make([]byte, max, max)
	fw := fixedWriter{b, 0}
	return &fw
}
