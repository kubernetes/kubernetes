package internal

import (
	"bytes"
	"sync"
)

var bytesBufferPool = sync.Pool{
	New: func() interface{} {
		return new(bytes.Buffer)
	},
}

// NewBuffer retrieves a [bytes.Buffer] from a pool an re-initialises it.
//
// The returned buffer should be passed to [PutBuffer].
func NewBuffer(buf []byte) *bytes.Buffer {
	wr := bytesBufferPool.Get().(*bytes.Buffer)
	// Reinitialize the Buffer with a new backing slice since it is returned to
	// the caller by wr.Bytes() below. Pooling is faster despite calling
	// NewBuffer. The pooled alloc is still reused, it only needs to be zeroed.
	*wr = *bytes.NewBuffer(buf)
	return wr
}

// PutBuffer releases a buffer to the pool.
func PutBuffer(buf *bytes.Buffer) {
	// Release reference to the backing buffer.
	*buf = *bytes.NewBuffer(nil)
	bytesBufferPool.Put(buf)
}
