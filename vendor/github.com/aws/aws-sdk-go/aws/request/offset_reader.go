package request

import (
	"io"
	"sync"
)

// offsetReader is a thread-safe io.ReadCloser to prevent racing
// with retrying requests
type offsetReader struct {
	buf    io.ReadSeeker
	lock   sync.RWMutex
	closed bool
}

func newOffsetReader(buf io.ReadSeeker, offset int64) *offsetReader {
	reader := &offsetReader{}
	buf.Seek(offset, 0)

	reader.buf = buf
	return reader
}

// Close is a thread-safe close. Uses the write lock.
func (o *offsetReader) Close() error {
	o.lock.Lock()
	defer o.lock.Unlock()
	o.closed = true
	return nil
}

// Read is a thread-safe read using a read lock.
func (o *offsetReader) Read(p []byte) (int, error) {
	o.lock.RLock()
	defer o.lock.RUnlock()

	if o.closed {
		return 0, io.EOF
	}

	return o.buf.Read(p)
}

// CloseAndCopy will return a new offsetReader with a copy of the old buffer
// and close the old buffer.
func (o *offsetReader) CloseAndCopy(offset int64) *offsetReader {
	o.Close()
	return newOffsetReader(o.buf, offset)
}
