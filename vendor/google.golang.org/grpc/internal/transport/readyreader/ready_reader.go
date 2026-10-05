/*
 *
 * Copyright 2026 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package readyreader provides utilities to perform non-memory-pinning reads.
package readyreader

import (
	"io"
	"net"
	"syscall"

	"google.golang.org/grpc/mem"
)

// Reader is an optional interface that can be implemented by [net.Conn]
// implementations to enable gRPC to perform non-memory-pinning reads.
type Reader interface {
	// ReadOnReady waits for data to arrive, fetches a buffer, and performs a
	// read. When the underlying IO is readable, it allocates a buffer of size
	// bufSize from the pool and reads up to bufSize bytes into the buffer.
	//
	// It returns a pointer to the buffer so it can be returned to the pool
	// later, the number of bytes read, and an error.
	//
	// Callers should always process the n > 0 bytes returned before considering
	// the error. Doing so correctly handles I/O errors that happen after
	// reading some bytes, as well as both of the allowed EOF behaviors.
	ReadOnReady(bufSize int, pool mem.BufferPool) (b *[]byte, n int, err error)
}

// nonBlockingReader is optimized for non-memory-pinning reads using the RawConn
// interface.
type nonBlockingReader struct {
	raw syscall.RawConn
	// The following fields are stored as field to avoid heap allocations.
	state  readState
	doRead func(fd uintptr) bool
}

type readState struct {
	// Request params.
	bufSize int
	pool    mem.BufferPool

	// Response params.
	readError error
	bytesRead int
	buf       *[]byte
}

// NewNonBlocking returns a ReadyReader if the passed reader supports
// non-memory-pinning reads, else nil.
func NewNonBlocking(r io.Reader) Reader {
	if rr, ok := r.(Reader); ok {
		return rr
	}
	if !isRawConnSupported() {
		return nil
	}
	// We restrict the types before asserting syscall.Conn. The credentials
	// package may return a wrapper that implements syscall.Conn by embedding
	// both the raw connection and the encrypted connection. If the code
	// attempts to read directly from the raw syscall.RawConn, it would read
	// encrypted data.
	switch r.(type) {
	case *net.TCPConn, *net.UDPConn, *net.UnixConn, *net.IPConn:
	default:
		return nil
	}
	sysConn, ok := r.(syscall.Conn)
	if !ok {
		return nil
	}
	raw, err := sysConn.SyscallConn()
	if err != nil {
		return nil
	}
	rr := &nonBlockingReader{raw: raw}
	rr.doRead = func(fd uintptr) bool {
		s := &rr.state

		s.buf = s.pool.Get(s.bufSize)
		s.bytesRead, s.readError = sysRead(fd, *s.buf)

		if s.readError != nil {
			s.pool.Put(s.buf)
			s.buf = nil
		}
		return !wouldBlock(s.readError)
	}
	return rr
}

func (c *nonBlockingReader) ReadOnReady(bufSize int, pool mem.BufferPool) (*[]byte, int, error) {
	c.state = readState{
		pool:    pool,
		bufSize: bufSize,
	}
	err := c.raw.Read(c.doRead)

	buf := c.state.buf
	n := c.state.bytesRead
	readErr := c.state.readError
	c.state = readState{}

	if err != nil {
		if buf != nil {
			pool.Put(buf)
		}
		return nil, 0, err
	}
	if readErr != nil {
		// buffer is already released in the callback.
		return nil, 0, readErr
	}
	if n == 0 {
		// syscall.Read doesn't consider a graceful socket closure to be an
		// error condition, but Go's io.Reader expects an EOF error.
		pool.Put(buf)
		return nil, 0, io.EOF
	}
	return buf, n, nil
}

type blockingReader struct {
	reader io.Reader
}

func (c *blockingReader) ReadOnReady(bufSize int, pool mem.BufferPool) (*[]byte, int, error) {
	buf := pool.Get(bufSize)
	n, err := c.reader.Read(*buf)
	if err != nil {
		pool.Put(buf)
		return nil, 0, err
	}
	return buf, n, nil
}

// New detects if [syscall.RawConn] is available for non-memory-pinning reads.
// If [syscall.RawConn] is unavailable, it falls back to using the simpler
// [io.Reader] interface for reads.
func New(r io.Reader) Reader {
	if r := NewNonBlocking(r); r != nil {
		return r
	}
	return &blockingReader{reader: r}
}

// bufReadyReader implements buffering for a ReadyReader object.
// A new bufReadyReader is created by calling [NewBuffered].
type bufReadyReader struct {
	buf       *[]byte
	pool      mem.BufferPool
	bufSize   int
	rd        Reader // reader provided by the caller
	r, w      int    // buf read and write positions
	err       error
	constPool constBufferPool // stored as a field to avoid heap allocations.
}

// NewBuffered returns a new [io.Reader] with a buffer of the specified size
// which is allocated from the provided pool.
func NewBuffered(rd Reader, size int, pool mem.BufferPool) io.Reader {
	return &bufReadyReader{
		rd:      rd,
		pool:    pool,
		bufSize: size,
	}
}

func (b *bufReadyReader) readErr() error {
	err := b.err
	b.err = nil
	return err
}

func (b *bufReadyReader) buffered() int { return b.w - b.r }

// Read reads data into p. It returns the number of bytes read into p. The
// bytes are taken from at most one Read on the underlying [ReadyReader],
// hence n may be less than len(p). If the underlying [ReadyReader] can return
// a non-zero count with io.EOF, then this Read method can do so as well; see
// the [io.Reader] docs.
func (b *bufReadyReader) Read(p []byte) (n int, err error) {
	n = len(p)
	if n == 0 {
		if b.buffered() > 0 {
			return 0, nil
		}
		return 0, b.readErr()
	}
	if b.r == b.w {
		if b.err != nil {
			return 0, b.readErr()
		}
		if len(p) >= b.bufSize {
			// Large read, empty buffer.
			// Read directly into p to avoid copy.
			b.constPool.buffer = p
			_, n, b.err = b.rd.ReadOnReady(len(p), &b.constPool)
			return n, b.readErr()
		}
		// One read.
		b.r = 0
		b.w = 0
		b.buf, n, b.err = b.rd.ReadOnReady(b.bufSize, b.pool)
		if n == 0 {
			if b.buf != nil {
				b.pool.Put(b.buf)
				b.buf = nil
			}
			return 0, b.readErr()
		}
		b.w += n
	}

	// copy as much as we can
	// b.buf must be non-nil since b.r != b.w.
	buf := *b.buf
	n = copy(p, buf[b.r:b.w])
	b.r += n
	if b.r == b.w {
		// Consumed entire buffer, release it.
		b.pool.Put(b.buf)
		b.buf = nil
	}
	return n, nil
}

type constBufferPool struct {
	buffer []byte
}

func (p *constBufferPool) Get(int) *[]byte {
	return &p.buffer
}

func (p *constBufferPool) Put(*[]byte) {}
