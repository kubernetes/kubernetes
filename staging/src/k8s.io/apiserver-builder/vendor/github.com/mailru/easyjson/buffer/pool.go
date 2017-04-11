// Package buffer implements a buffer for serialization, consisting of a chain of []byte-s to
// reduce copying and to allow reuse of individual chunks.
package buffer

import (
	"io"
	"sync"
)

// PoolConfig contains configuration for the allocation and reuse strategy.
type PoolConfig struct {
	StartSize  int // Minimum chunk size that is allocated.
	PooledSize int // Minimum chunk size that is reused, reusing chunks too small will result in overhead.
	MaxSize    int // Maximum chunk size that will be allocated.
}

var config = PoolConfig{
	StartSize:  128,
	PooledSize: 512,
	MaxSize:    32768,
}

// Reuse pool: chunk size -> pool.
var buffers = map[int]*sync.Pool{}

func initBuffers() {
	for l := config.PooledSize; l <= config.MaxSize; l *= 2 {
		buffers[l] = new(sync.Pool)
	}
}

func init() {
	initBuffers()
}

// Init sets up a non-default pooling and allocation strategy. Should be run before serialization is done.
func Init(cfg PoolConfig) {
	config = cfg
	initBuffers()
}

// putBuf puts a chunk to reuse pool if it can be reused.
func putBuf(buf []byte) {
	size := cap(buf)
	if size < config.PooledSize {
		return
	}
	if c := buffers[size]; c != nil {
		c.Put(buf[:0])
	}
}

// getBuf gets a chunk from reuse pool or creates a new one if reuse failed.
func getBuf(size int) []byte {
	if size < config.PooledSize {
		return make([]byte, 0, size)
	}

	if c := buffers[size]; c != nil {
		v := c.Get()
		if v != nil {
			return v.([]byte)
		}
	}
	return make([]byte, 0, size)
}

// Buffer is a buffer optimized for serialization without extra copying.
type Buffer struct {

	// Buf is the current chunk that can be used for serialization.
	Buf []byte

	toPool []byte
	bufs   [][]byte
}

// EnsureSpace makes sure that the current chunk contains at least s free bytes,
// possibly creating a new chunk.
func (b *Buffer) EnsureSpace(s int) {
	if cap(b.Buf)-len(b.Buf) >= s {
		return
	}
	l := len(b.Buf)
	if l > 0 {
		if cap(b.toPool) != cap(b.Buf) {
			// Chunk was reallocated, toPool can be pooled.
			putBuf(b.toPool)
		}
		if cap(b.bufs) == 0 {
			b.bufs = make([][]byte, 0, 8)
		}
		b.bufs = append(b.bufs, b.Buf)
		l = cap(b.toPool) * 2
	} else {
		l = config.StartSize
	}

	if l > config.MaxSize {
		l = config.MaxSize
	}
	b.Buf = getBuf(l)
	b.toPool = b.Buf
}

// AppendByte appends a single byte to buffer.
func (b *Buffer) AppendByte(data byte) {
	if cap(b.Buf) == len(b.Buf) { // EnsureSpace won't be inlined.
		b.EnsureSpace(1)
	}
	b.Buf = append(b.Buf, data)
}

// AppendBytes appends a byte slice to buffer.
func (b *Buffer) AppendBytes(data []byte) {
	for len(data) > 0 {
		if cap(b.Buf) == len(b.Buf) { // EnsureSpace won't be inlined.
			b.EnsureSpace(1)
		}

		sz := cap(b.Buf) - len(b.Buf)
		if sz > len(data) {
			sz = len(data)
		}

		b.Buf = append(b.Buf, data[:sz]...)
		data = data[sz:]
	}
}

// AppendBytes appends a string to buffer.
func (b *Buffer) AppendString(data string) {
	for len(data) > 0 {
		if cap(b.Buf) == len(b.Buf) { // EnsureSpace won't be inlined.
			b.EnsureSpace(1)
		}

		sz := cap(b.Buf) - len(b.Buf)
		if sz > len(data) {
			sz = len(data)
		}

		b.Buf = append(b.Buf, data[:sz]...)
		data = data[sz:]
	}
}

// Size computes the size of a buffer by adding sizes of every chunk.
func (b *Buffer) Size() int {
	size := len(b.Buf)
	for _, buf := range b.bufs {
		size += len(buf)
	}
	return size
}

// DumpTo outputs the contents of a buffer to a writer and resets the buffer.
func (b *Buffer) DumpTo(w io.Writer) (written int, err error) {
	var n int
	for _, buf := range b.bufs {
		if err == nil {
			n, err = w.Write(buf)
			written += n
		}
		putBuf(buf)
	}

	if err == nil {
		n, err = w.Write(b.Buf)
		written += n
	}
	putBuf(b.toPool)

	b.bufs = nil
	b.Buf = nil
	b.toPool = nil

	return
}

// BuildBytes creates a single byte slice with all the contents of the buffer. Data is
// copied if it does not fit in a single chunk.
func (b *Buffer) BuildBytes() []byte {
	if len(b.bufs) == 0 {

		ret := b.Buf
		b.toPool = nil
		b.Buf = nil

		return ret
	}

	ret := make([]byte, 0, b.Size())
	for _, buf := range b.bufs {
		ret = append(ret, buf...)
		putBuf(buf)
	}

	ret = append(ret, b.Buf...)
	putBuf(b.toPool)

	b.bufs = nil
	b.toPool = nil
	b.Buf = nil

	return ret
}
