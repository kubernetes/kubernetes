/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package yaml

import "io"

// StreamReader is a reader designed for consuming streams of variable-length
// messages. It buffers data until it is explicitly consumed, and can be
// rewound to re-read previous data.
type StreamReader struct {
	r           io.Reader
	buf         []byte
	head        int // current read offset into buf
	ttlConsumed int // number of bytes which have been consumed
}

// NewStreamReader creates a new StreamReader wrapping the provided
// io.Reader.
func NewStreamReader(r io.Reader, size int) *StreamReader {
	if size == 0 {
		size = 4096
	}
	return &StreamReader{
		r:   r,
		buf: make([]byte, 0, size), // Start with a reasonable capacity
	}
}

// Read implements io.Reader. It first returns any buffered data after the
// current offset, and if that's exhausted, reads from the underlying reader
// and buffers the data. The returned data is not considered consumed until the
// Consume method is called.
func (r *StreamReader) Read(p []byte) (n int, err error) {
	// If we have buffered data, return it
	if r.head < len(r.buf) {
		n = copy(p, r.buf[r.head:])
		r.head += n
		return n, nil
	}

	// If we've already hit EOF, return it
	if r.r == nil {
		return 0, io.EOF
	}

	// Read from the underlying reader
	n, err = r.r.Read(p)
	if n > 0 {
		r.buf = append(r.buf, p[:n]...)
		r.head += n
	}
	if err == nil {
		return n, nil
	}
	if err == io.EOF {
		// Store that we've hit EOF by setting r to nil
		r.r = nil
	}
	return n, err
}

// ReadN reads exactly n bytes from the reader, blocking until all bytes are
// read or an error occurs. If an error occurs, the number of bytes read is
// returned along with the error. If EOF is hit before n bytes are read, this
// will return the bytes read so far, along with io.EOF. The returned data is
// not considered consumed until the Consume method is called.
func (r *StreamReader) ReadN(want int) ([]byte, error) {
	ret := make([]byte, want)
	off := 0
	for off < want {
		n, err := r.Read(ret[off:])
		if err != nil {
			return ret[:off+n], err
		}
		off += n
	}
	return ret, nil
}

// Peek returns the next n bytes without advancing the reader. The returned
// bytes are valid until the next call to Consume.
func (r *StreamReader) Peek(n int) ([]byte, error) {
	buf, err := r.ReadN(n)
	r.RewindN(len(buf))
	if err != nil {
		return buf, err
	}
	return buf, nil
}

// Rewind resets the reader to the beginning of the buffered data.
func (r *StreamReader) Rewind() {
	r.head = 0
}

// RewindN rewinds the reader by n bytes. If n is greater than the current
// buffer, the reader is rewound to the beginning of the buffer.
func (r *StreamReader) RewindN(n int) {
	r.head -= min(n, r.head)
}

// Consume discards up to n bytes of previously read data from the beginning of
// the buffer. Once consumed, that data is no longer available for rewinding.
// If n is greater than the current buffer, the buffer is cleared. Consume
// never consume data from the underlying reader.
func (r *StreamReader) Consume(n int) {
	n = min(n, len(r.buf))
	r.buf = r.buf[n:]
	r.head -= n
	r.ttlConsumed += n
}

// Consumed returns the number of bytes consumed from the input reader.
func (r *StreamReader) Consumed() int {
	return r.ttlConsumed
}
