package s3manager

import (
	"io"
	"sync"
)

// ReadSeekerWriteTo defines an interface implementing io.WriteTo and io.ReadSeeker
type ReadSeekerWriteTo interface {
	io.ReadSeeker
	io.WriterTo
}

// BufferedReadSeekerWriteTo wraps a BufferedReadSeeker with an io.WriteAt
// implementation.
type BufferedReadSeekerWriteTo struct {
	*BufferedReadSeeker
}

// WriteTo writes to the given io.Writer from BufferedReadSeeker until there's no more data to write or
// an error occurs. Returns the number of bytes written and any error encountered during the write.
func (b *BufferedReadSeekerWriteTo) WriteTo(writer io.Writer) (int64, error) {
	return io.Copy(writer, b.BufferedReadSeeker)
}

// ReadSeekerWriteToProvider provides an implementation of io.WriteTo for an io.ReadSeeker
type ReadSeekerWriteToProvider interface {
	GetWriteTo(seeker io.ReadSeeker) (r ReadSeekerWriteTo, cleanup func())
}

// BufferedReadSeekerWriteToPool uses a sync.Pool to create and reuse
// []byte slices for buffering parts in memory
type BufferedReadSeekerWriteToPool struct {
	pool sync.Pool
}

// NewBufferedReadSeekerWriteToPool will return a new BufferedReadSeekerWriteToPool that will create
// a pool of reusable buffers . If size is less then < 64 KiB then the buffer
// will default to 64 KiB. Reason: io.Copy from writers or readers that don't support io.WriteTo or io.ReadFrom
// respectively will default to copying 32 KiB.
func NewBufferedReadSeekerWriteToPool(size int) *BufferedReadSeekerWriteToPool {
	if size < 65536 {
		size = 65536
	}

	return &BufferedReadSeekerWriteToPool{
		pool: sync.Pool{New: func() interface{} {
			return make([]byte, size)
		}},
	}
}

// GetWriteTo will wrap the provided io.ReadSeeker with a BufferedReadSeekerWriteTo.
// The provided cleanup must be called after operations have been completed on the
// returned io.ReadSeekerWriteTo in order to signal the return of resources to the pool.
func (p *BufferedReadSeekerWriteToPool) GetWriteTo(seeker io.ReadSeeker) (r ReadSeekerWriteTo, cleanup func()) {
	buffer := p.pool.Get().([]byte)

	r = &BufferedReadSeekerWriteTo{BufferedReadSeeker: NewBufferedReadSeeker(seeker, buffer)}
	cleanup = func() {
		p.pool.Put(buffer)
	}

	return r, cleanup
}
