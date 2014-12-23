// +build !go1.3

package pools

import (
	"bufio"
	"io"

	"github.com/docker/docker/pkg/ioutils"
)

var (
	BufioReader32KPool *BufioReaderPool
	BufioWriter32KPool *BufioWriterPool
)

const buffer32K = 32 * 1024

type BufioReaderPool struct {
	size int
}

func init() {
	BufioReader32KPool = newBufioReaderPoolWithSize(buffer32K)
	BufioWriter32KPool = newBufioWriterPoolWithSize(buffer32K)
}

func newBufioReaderPoolWithSize(size int) *BufioReaderPool {
	return &BufioReaderPool{size: size}
}

func (bufPool *BufioReaderPool) Get(r io.Reader) *bufio.Reader {
	return bufio.NewReaderSize(r, bufPool.size)
}

func (bufPool *BufioReaderPool) Put(b *bufio.Reader) {
	b.Reset(nil)
}

func (bufPool *BufioReaderPool) NewReadCloserWrapper(buf *bufio.Reader, r io.Reader) io.ReadCloser {
	return ioutils.NewReadCloserWrapper(r, func() error {
		if readCloser, ok := r.(io.ReadCloser); ok {
			return readCloser.Close()
		}
		return nil
	})
}

type BufioWriterPool struct {
	size int
}

func newBufioWriterPoolWithSize(size int) *BufioWriterPool {
	return &BufioWriterPool{size: size}
}

func (bufPool *BufioWriterPool) Get(w io.Writer) *bufio.Writer {
	return bufio.NewWriterSize(w, bufPool.size)
}

func (bufPool *BufioWriterPool) Put(b *bufio.Writer) {
	b.Reset(nil)
}

func (bufPool *BufioWriterPool) NewWriteCloserWrapper(buf *bufio.Writer, w io.Writer) io.WriteCloser {
	return ioutils.NewWriteCloserWrapper(w, func() error {
		buf.Flush()
		if writeCloser, ok := w.(io.WriteCloser); ok {
			return writeCloser.Close()
		}
		return nil
	})
}
