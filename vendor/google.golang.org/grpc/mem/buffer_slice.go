/*
 *
 * Copyright 2024 gRPC authors.
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

package mem

import (
	"fmt"
	"io"
)

const (
	// 32 KiB is what io.Copy uses.
	readAllBufSize = 32 * 1024
)

// BufferSlice offers a means to represent data that spans one or more Buffer
// instances. A BufferSlice is meant to be immutable after creation, and methods
// like Ref create and return copies of the slice. This is why all methods have
// value receivers rather than pointer receivers.
//
// Note that any of the methods that read the underlying buffers such as Ref,
// Len or CopyTo etc., will panic if any underlying buffers have already been
// freed. It is recommended to not directly interact with any of the underlying
// buffers directly, rather such interactions should be mediated through the
// various methods on this type.
//
// By convention, any APIs that return (mem.BufferSlice, error) should reduce
// the burden on the caller by never returning a mem.BufferSlice that needs to
// be freed if the error is non-nil, unless explicitly stated.
type BufferSlice []Buffer

// Len returns the sum of the length of all the Buffers in this slice.
//
// # Warning
//
// Invoking the built-in len on a BufferSlice will return the number of buffers
// in the slice, and *not* the value returned by this function.
func (s BufferSlice) Len() int {
	var length int
	for _, b := range s {
		length += b.Len()
	}
	return length
}

// Ref invokes Ref on each buffer in the slice.
func (s BufferSlice) Ref() {
	for _, b := range s {
		b.Ref()
	}
}

// Free invokes Buffer.Free() on each Buffer in the slice.
func (s BufferSlice) Free() {
	for _, b := range s {
		b.Free()
	}
}

// CopyTo copies each of the underlying Buffer's data into the given buffer,
// returning the number of bytes copied. Has the same semantics as the copy
// builtin in that it will copy as many bytes as it can, stopping when either dst
// is full or s runs out of data, returning the minimum of s.Len() and len(dst).
func (s BufferSlice) CopyTo(dst []byte) int {
	off := 0
	for _, b := range s {
		off += copy(dst[off:], b.ReadOnlyData())
	}
	return off
}

// Materialize concatenates all the underlying Buffer's data into a single
// contiguous buffer using CopyTo.
func (s BufferSlice) Materialize() []byte {
	l := s.Len()
	if l == 0 {
		return nil
	}
	out := make([]byte, l)
	s.CopyTo(out)
	return out
}

// MaterializeToBuffer functions like Materialize except that it writes the data
// to a single Buffer pulled from the given BufferPool.
//
// As a special case, if the input BufferSlice only actually has one Buffer, this
// function simply increases the refcount before returning said Buffer. Freeing this
// buffer won't release it until the BufferSlice is itself released.
func (s BufferSlice) MaterializeToBuffer(pool BufferPool) Buffer {
	if len(s) == 1 {
		s[0].Ref()
		return s[0]
	}
	sLen := s.Len()
	if sLen == 0 {
		return emptyBuffer{}
	}
	buf := pool.Get(sLen)
	s.CopyTo(*buf)
	return NewBuffer(buf, pool)
}

// Reader returns a new Reader for the input slice after taking references to
// each underlying buffer.
func (s BufferSlice) Reader() *Reader {
	s.Ref()
	return &Reader{
		data: s,
		len:  s.Len(),
	}
}

// Reader exposes a BufferSlice's data as an io.Reader, allowing it to interface
// with other systems.
//
// Buffers will be freed as they are read.
//
// A Reader can be constructed from a BufferSlice; alternatively the zero value
// of a Reader may be used after calling Reset on it.
type Reader struct {
	data BufferSlice
	len  int
	// The index into data[0].ReadOnlyData().
	bufferIdx int
}

// Remaining returns the number of unread bytes remaining in the slice.
func (r *Reader) Remaining() int {
	return r.len
}

// Reset frees the currently held buffer slice and starts reading from the
// provided slice. This allows reusing the reader object.
func (r *Reader) Reset(s BufferSlice) {
	r.data.Free()
	s.Ref()
	r.data = s
	r.len = s.Len()
	r.bufferIdx = 0
}

// Close frees the underlying BufferSlice and never returns an error. Subsequent
// calls to Read will return (0, io.EOF).
func (r *Reader) Close() error {
	r.data.Free()
	r.data = nil
	r.len = 0
	return nil
}

func (r *Reader) freeFirstBufferIfEmpty() bool {
	if len(r.data) == 0 || r.bufferIdx != len(r.data[0].ReadOnlyData()) {
		return false
	}

	r.data[0].Free()
	r.data = r.data[1:]
	r.bufferIdx = 0
	return true
}

func (r *Reader) Read(buf []byte) (n int, _ error) {
	if r.len == 0 {
		return 0, io.EOF
	}

	for len(buf) != 0 && r.len != 0 {
		// Copy as much as possible from the first Buffer in the slice into the
		// given byte slice.
		data := r.data[0].ReadOnlyData()
		copied := copy(buf, data[r.bufferIdx:])
		r.len -= copied       // Reduce len by the number of bytes copied.
		r.bufferIdx += copied // Increment the buffer index.
		n += copied           // Increment the total number of bytes read.
		buf = buf[copied:]    // Shrink the given byte slice.

		// If we have copied all the data from the first Buffer, free it and advance to
		// the next in the slice.
		r.freeFirstBufferIfEmpty()
	}

	return n, nil
}

// ReadByte reads a single byte.
func (r *Reader) ReadByte() (byte, error) {
	if r.len == 0 {
		return 0, io.EOF
	}

	// There may be any number of empty buffers in the slice, clear them all until a
	// non-empty buffer is reached. This is guaranteed to exit since r.len is not 0.
	for r.freeFirstBufferIfEmpty() {
	}

	b := r.data[0].ReadOnlyData()[r.bufferIdx]
	r.len--
	r.bufferIdx++
	// Free the first buffer in the slice if the last byte was read
	r.freeFirstBufferIfEmpty()
	return b, nil
}

var _ io.Writer = (*writer)(nil)

type writer struct {
	buffers *BufferSlice
	pool    BufferPool
}

func (w *writer) Write(p []byte) (n int, err error) {
	b := Copy(p, w.pool)
	*w.buffers = append(*w.buffers, b)
	return b.Len(), nil
}

// NewWriter wraps the given BufferSlice and BufferPool to implement the
// io.Writer interface. Every call to Write copies the contents of the given
// buffer into a new Buffer pulled from the given pool and the Buffer is
// added to the given BufferSlice.
func NewWriter(buffers *BufferSlice, pool BufferPool) io.Writer {
	return &writer{buffers: buffers, pool: pool}
}

// ReadAll reads from r until an error or EOF and returns the data it read.
// A successful call returns err == nil, not err == EOF. Because ReadAll is
// defined to read from src until EOF, it does not treat an EOF from Read
// as an error to be reported.
//
// Important: A failed call returns a non-nil error and may also return
// partially read buffers. It is the responsibility of the caller to free the
// BufferSlice returned, or its memory will not be reused.
func ReadAll(r io.Reader, pool BufferPool) (BufferSlice, error) {
	var result BufferSlice
	if wt, ok := r.(io.WriterTo); ok {
		// This is more optimal since wt knows the size of chunks it wants to
		// write and, hence, we can allocate buffers of an optimal size to fit
		// them. E.g. might be a single big chunk, and we wouldn't chop it
		// into pieces.
		w := NewWriter(&result, pool)
		_, err := wt.WriteTo(w)
		return result, err
	}
nextBuffer:
	for {
		buf := pool.Get(readAllBufSize)
		// We asked for 32KiB but may have been given a bigger buffer.
		// Use all of it if that's the case.
		*buf = (*buf)[:cap(*buf)]
		usedCap := 0
		for {
			n, err := r.Read((*buf)[usedCap:])
			usedCap += n
			if err != nil {
				if usedCap == 0 {
					// Nothing in this buf, put it back
					pool.Put(buf)
				} else {
					*buf = (*buf)[:usedCap]
					result = append(result, NewBuffer(buf, pool))
				}
				if err == io.EOF {
					err = nil
				}
				return result, err
			}
			if len(*buf) == usedCap {
				result = append(result, NewBuffer(buf, pool))
				continue nextBuffer
			}
		}
	}
}

// Discard skips the next n bytes, returning the number of bytes discarded.
//
// It frees buffers as they are fully consumed.
//
// If Discard skips fewer than n bytes, it also returns an error.
func (r *Reader) Discard(n int) (discarded int, err error) {
	total := n
	for n > 0 && r.len > 0 {
		curData := r.data[0].ReadOnlyData()
		curSize := min(n, len(curData)-r.bufferIdx)
		n -= curSize
		r.len -= curSize
		r.bufferIdx += curSize
		if r.bufferIdx >= len(curData) {
			r.data[0].Free()
			r.data = r.data[1:]
			r.bufferIdx = 0
		}
	}
	discarded = total - n
	if n > 0 {
		return discarded, fmt.Errorf("insufficient bytes in reader")
	}
	return discarded, nil
}

// Peek returns the next n bytes without advancing the reader.
//
// Peek appends results to the provided res slice and returns the updated slice.
// This pattern allows re-using the storage of res if it has sufficient
// capacity.
//
// The returned subslices are views into the underlying buffers and are only
// valid until the reader is advanced past the corresponding buffer.
//
// If Peek returns fewer than n bytes, it also returns an error.
func (r *Reader) Peek(n int, res [][]byte) ([][]byte, error) {
	for i := 0; n > 0 && i < len(r.data); i++ {
		curData := r.data[i].ReadOnlyData()
		start := 0
		if i == 0 {
			start = r.bufferIdx
		}
		curSize := min(n, len(curData)-start)
		if curSize == 0 {
			continue
		}
		res = append(res, curData[start:start+curSize])
		n -= curSize
	}
	if n > 0 {
		return nil, fmt.Errorf("insufficient bytes in reader")
	}
	return res, nil
}
