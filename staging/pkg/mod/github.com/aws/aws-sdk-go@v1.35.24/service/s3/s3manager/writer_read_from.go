package s3manager

import (
	"bufio"
	"io"
	"sync"

	"github.com/aws/aws-sdk-go/internal/sdkio"
)

// WriterReadFrom defines an interface implementing io.Writer and io.ReaderFrom
type WriterReadFrom interface {
	io.Writer
	io.ReaderFrom
}

// WriterReadFromProvider provides an implementation of io.ReadFrom for the given io.Writer
type WriterReadFromProvider interface {
	GetReadFrom(writer io.Writer) (w WriterReadFrom, cleanup func())
}

type bufferedWriter interface {
	WriterReadFrom
	Flush() error
	Reset(io.Writer)
}

type bufferedReadFrom struct {
	bufferedWriter
}

func (b *bufferedReadFrom) ReadFrom(r io.Reader) (int64, error) {
	n, err := b.bufferedWriter.ReadFrom(r)
	if flushErr := b.Flush(); flushErr != nil && err == nil {
		err = flushErr
	}
	return n, err
}

// PooledBufferedReadFromProvider is a WriterReadFromProvider that uses a sync.Pool
// to manage allocation and reuse of *bufio.Writer structures.
type PooledBufferedReadFromProvider struct {
	pool sync.Pool
}

// NewPooledBufferedWriterReadFromProvider returns a new PooledBufferedReadFromProvider
// Size is used to control the size of the underlying *bufio.Writer created for
// calls to GetReadFrom.
func NewPooledBufferedWriterReadFromProvider(size int) *PooledBufferedReadFromProvider {
	if size < int(32*sdkio.KibiByte) {
		size = int(64 * sdkio.KibiByte)
	}

	return &PooledBufferedReadFromProvider{
		pool: sync.Pool{
			New: func() interface{} {
				return &bufferedReadFrom{bufferedWriter: bufio.NewWriterSize(nil, size)}
			},
		},
	}
}

// GetReadFrom takes an io.Writer and wraps it with a type which satisfies the WriterReadFrom
// interface/ Additionally a cleanup function is provided which must be called after usage of the WriterReadFrom
// has been completed in order to allow the reuse of the *bufio.Writer
func (p *PooledBufferedReadFromProvider) GetReadFrom(writer io.Writer) (r WriterReadFrom, cleanup func()) {
	buffer := p.pool.Get().(*bufferedReadFrom)
	buffer.Reset(writer)
	r = buffer
	cleanup = func() {
		buffer.Reset(nil) // Reset to nil writer to release reference
		p.pool.Put(buffer)
	}
	return r, cleanup
}
