package ioutils

import (
	"errors"
	"io"
)

var errBufferFull = errors.New("buffer is full")

type fixedBuffer struct {
	buf      []byte
	pos      int
	lastRead int
}

func (b *fixedBuffer) Write(p []byte) (int, error) {
	n := copy(b.buf[b.pos:cap(b.buf)], p)
	b.pos += n

	if n < len(p) {
		if b.pos == cap(b.buf) {
			return n, errBufferFull
		}
		return n, io.ErrShortWrite
	}
	return n, nil
}

func (b *fixedBuffer) Read(p []byte) (int, error) {
	n := copy(p, b.buf[b.lastRead:b.pos])
	b.lastRead += n
	return n, nil
}

func (b *fixedBuffer) Len() int {
	return b.pos - b.lastRead
}

func (b *fixedBuffer) Cap() int {
	return cap(b.buf)
}

func (b *fixedBuffer) Reset() {
	b.pos = 0
	b.lastRead = 0
	b.buf = b.buf[:0]
}

func (b *fixedBuffer) String() string {
	return string(b.buf[b.lastRead:b.pos])
}
