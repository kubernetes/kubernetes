// Package pools provides a collection of pools which provide various
// data types with buffers. These can be used to lower the number of
// memory allocations and reuse buffers.
//
// New pools should be added to this package to allow them to be
// shared across packages.
//
// Utility functions which operate on pools should be added to this
// package to allow them to be reused.
package pools

import (
	"bufio"
	"io"
	"sync"

	"github.com/fsouza/go-dockerclient/external/github.com/docker/docker/pkg/ioutils"
)

var (
	// BufioReader32KPool is a pool which returns bufio.Reader with a 32K buffer.
	BufioReader32KPool *BufioReaderPool
	// BufioWriter32KPool is a pool which returns bufio.Writer with a 32K buffer.
	BufioWriter32KPool *BufioWriterPool
)

const buffer32K = 32 * 1024

// BufioReaderPool is a bufio reader that uses sync.Pool.
type BufioReaderPool struct {
	pool sync.Pool
}

func init() {
	BufioReader32KPool = newBufioReaderPoolWithSize(buffer32K)
	BufioWriter32KPool = newBufioWriterPoolWithSize(buffer32K)
}

// newBufioReaderPoolWithSize is unexported because new pools should be
// added here to be shared where required.
func newBufioReaderPoolWithSize(size int) *BufioReaderPool {
	pool := sync.Pool{
		New: func() interface{} { return bufio.NewReaderSize(nil, size) },
	}
	return &BufioReaderPool{pool: pool}
}

// Get returns a bufio.Reader which reads from r. The buffer size is that of the pool.
func (bufPool *BufioReaderPool) Get(r io.Reader) *bufio.Reader {
	buf := bufPool.pool.Get().(*bufio.Reader)
	buf.Reset(r)
	return buf
}

// Put puts the bufio.Reader back into the pool.
func (bufPool *BufioReaderPool) Put(b *bufio.Reader) {
	b.Reset(nil)
	bufPool.pool.Put(b)
}

// Copy is a convenience wrapper which uses a buffer to avoid allocation in io.Copy.
func Copy(dst io.Writer, src io.Reader) (written int64, err error) {
	buf := BufioReader32KPool.Get(src)
	written, err = io.Copy(dst, buf)
	BufioReader32KPool.Put(buf)
	return
}

// NewReadCloserWrapper returns a wrapper which puts the bufio.Reader back
// into the pool and closes the reader if it's an io.ReadCloser.
func (bufPool *BufioReaderPool) NewReadCloserWrapper(buf *bufio.Reader, r io.Reader) io.ReadCloser {
	return ioutils.NewReadCloserWrapper(r, func() error {
		if readCloser, ok := r.(io.ReadCloser); ok {
			readCloser.Close()
		}
		bufPool.Put(buf)
		return nil
	})
}

// BufioWriterPool is a bufio writer that uses sync.Pool.
type BufioWriterPool struct {
	pool sync.Pool
}

// newBufioWriterPoolWithSize is unexported because new pools should be
// added here to be shared where required.
func newBufioWriterPoolWithSize(size int) *BufioWriterPool {
	pool := sync.Pool{
		New: func() interface{} { return bufio.NewWriterSize(nil, size) },
	}
	return &BufioWriterPool{pool: pool}
}

// Get returns a bufio.Writer which writes to w. The buffer size is that of the pool.
func (bufPool *BufioWriterPool) Get(w io.Writer) *bufio.Writer {
	buf := bufPool.pool.Get().(*bufio.Writer)
	buf.Reset(w)
	return buf
}

// Put puts the bufio.Writer back into the pool.
func (bufPool *BufioWriterPool) Put(b *bufio.Writer) {
	b.Reset(nil)
	bufPool.pool.Put(b)
}

// NewWriteCloserWrapper returns a wrapper which puts the bufio.Writer back
// into the pool and closes the writer if it's an io.Writecloser.
func (bufPool *BufioWriterPool) NewWriteCloserWrapper(buf *bufio.Writer, w io.Writer) io.WriteCloser {
	return ioutils.NewWriteCloserWrapper(w, func() error {
		buf.Flush()
		if writeCloser, ok := w.(io.WriteCloser); ok {
			writeCloser.Close()
		}
		bufPool.Put(buf)
		return nil
	})
}
