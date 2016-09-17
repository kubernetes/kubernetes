package circbuf

import (
	"fmt"
)

// Buffer implements a circular buffer. It is a fixed size,
// and new writes overwrite older data, such that for a buffer
// of size N, for any amount of writes, only the last N bytes
// are retained.
type Buffer struct {
	data        []byte
	size        int64
	writeCursor int64
	written     int64
}

// NewBuffer creates a new buffer of a given size. The size
// must be greater than 0.
func NewBuffer(size int64) (*Buffer, error) {
	if size <= 0 {
		return nil, fmt.Errorf("Size must be positive")
	}

	b := &Buffer{
		size: size,
		data: make([]byte, size),
	}
	return b, nil
}

// Write writes up to len(buf) bytes to the internal ring,
// overriding older data if necessary.
func (b *Buffer) Write(buf []byte) (int, error) {
	// Account for total bytes written
	n := len(buf)
	b.written += int64(n)

	// If the buffer is larger than ours, then we only care
	// about the last size bytes anyways
	if int64(n) > b.size {
		buf = buf[int64(n)-b.size:]
	}

	// Copy in place
	remain := b.size - b.writeCursor
	copy(b.data[b.writeCursor:], buf)
	if int64(len(buf)) > remain {
		copy(b.data, buf[remain:])
	}

	// Update location of the cursor
	b.writeCursor = ((b.writeCursor + int64(len(buf))) % b.size)
	return n, nil
}

// Size returns the size of the buffer
func (b *Buffer) Size() int64 {
	return b.size
}

// TotalWritten provides the total number of bytes written
func (b *Buffer) TotalWritten() int64 {
	return b.written
}

// Bytes provides a slice of the bytes written. This
// slice should not be written to.
func (b *Buffer) Bytes() []byte {
	switch {
	case b.written >= b.size && b.writeCursor == 0:
		return b.data
	case b.written > b.size:
		out := make([]byte, b.size)
		copy(out, b.data[b.writeCursor:])
		copy(out[b.size-b.writeCursor:], b.data[:b.writeCursor])
		return out
	default:
		return b.data[:b.writeCursor]
	}
}

// Reset resets the buffer so it has no content.
func (b *Buffer) Reset() {
	b.writeCursor = 0
	b.written = 0
}

// String returns the contents of the buffer as a string
func (b *Buffer) String() string {
	return string(b.Bytes())
}
