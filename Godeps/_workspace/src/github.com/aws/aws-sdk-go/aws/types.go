package aws

import (
	"io"
	"sync"
)

// ReadSeekCloser wraps a io.Reader returning a ReaderSeekerCloser
func ReadSeekCloser(r io.Reader) ReaderSeekerCloser {
	return ReaderSeekerCloser{r}
}

// ReaderSeekerCloser represents a reader that can also delegate io.Seeker and
// io.Closer interfaces to the underlying object if they are available.
type ReaderSeekerCloser struct {
	r io.Reader
}

// Read reads from the reader up to size of p. The number of bytes read, and
// error if it occurred will be returned.
//
// If the reader is not an io.Reader zero bytes read, and nil error will be returned.
//
// Performs the same functionality as io.Reader Read
func (r ReaderSeekerCloser) Read(p []byte) (int, error) {
	switch t := r.r.(type) {
	case io.Reader:
		return t.Read(p)
	}
	return 0, nil
}

// Seek sets the offset for the next Read to offset, interpreted according to
// whence: 0 means relative to the origin of the file, 1 means relative to the
// current offset, and 2 means relative to the end. Seek returns the new offset
// and an error, if any.
//
// If the ReaderSeekerCloser is not an io.Seeker nothing will be done.
func (r ReaderSeekerCloser) Seek(offset int64, whence int) (int64, error) {
	switch t := r.r.(type) {
	case io.Seeker:
		return t.Seek(offset, whence)
	}
	return int64(0), nil
}

// Close closes the ReaderSeekerCloser.
//
// If the ReaderSeekerCloser is not an io.Closer nothing will be done.
func (r ReaderSeekerCloser) Close() error {
	switch t := r.r.(type) {
	case io.Closer:
		return t.Close()
	}
	return nil
}

// A WriteAtBuffer provides a in memory buffer supporting the io.WriterAt interface
// Can be used with the s3manager.Downloader to download content to a buffer
// in memory. Safe to use concurrently.
type WriteAtBuffer struct {
	buf []byte
	m   sync.Mutex
}

// WriteAt writes a slice of bytes to a buffer starting at the position provided
// The number of bytes written will be returned, or error. Can overwrite previous
// written slices if the write ats overlap.
func (b *WriteAtBuffer) WriteAt(p []byte, pos int64) (n int, err error) {
	b.m.Lock()
	defer b.m.Unlock()

	expLen := pos + int64(len(p))
	if int64(len(b.buf)) < expLen {
		newBuf := make([]byte, expLen)
		copy(newBuf, b.buf)
		b.buf = newBuf
	}
	copy(b.buf[pos:], p)
	return len(p), nil
}

// Bytes returns a slice of bytes written to the buffer.
func (b *WriteAtBuffer) Bytes() []byte {
	b.m.Lock()
	defer b.m.Unlock()
	return b.buf[:len(b.buf):len(b.buf)]
}
