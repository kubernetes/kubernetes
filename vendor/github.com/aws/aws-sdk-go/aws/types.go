package aws

import (
	"io"
	"sync"
)

// ReadSeekCloser wraps a io.Reader returning a ReaderSeekerCloser. Should
// only be used with an io.Reader that is also an io.Seeker. Doing so may
// cause request signature errors, or request body's not sent for GET, HEAD
// and DELETE HTTP methods.
//
// Deprecated: Should only be used with io.ReadSeeker. If using for
// S3 PutObject to stream content use s3manager.Uploader instead.
func ReadSeekCloser(r io.Reader) ReaderSeekerCloser {
	return ReaderSeekerCloser{r}
}

// ReaderSeekerCloser represents a reader that can also delegate io.Seeker and
// io.Closer interfaces to the underlying object if they are available.
type ReaderSeekerCloser struct {
	r io.Reader
}

// IsReaderSeekable returns if the underlying reader type can be seeked. A
// io.Reader might not actually be seekable if it is the ReaderSeekerCloser
// type.
func IsReaderSeekable(r io.Reader) bool {
	switch v := r.(type) {
	case ReaderSeekerCloser:
		return v.IsSeeker()
	case *ReaderSeekerCloser:
		return v.IsSeeker()
	case io.ReadSeeker:
		return true
	default:
		return false
	}
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

// IsSeeker returns if the underlying reader is also a seeker.
func (r ReaderSeekerCloser) IsSeeker() bool {
	_, ok := r.r.(io.Seeker)
	return ok
}

// HasLen returns the length of the underlying reader if the value implements
// the Len() int method.
func (r ReaderSeekerCloser) HasLen() (int, bool) {
	type lenner interface {
		Len() int
	}

	if lr, ok := r.r.(lenner); ok {
		return lr.Len(), true
	}

	return 0, false
}

// GetLen returns the length of the bytes remaining in the underlying reader.
// Checks first for Len(), then io.Seeker to determine the size of the
// underlying reader.
//
// Will return -1 if the length cannot be determined.
func (r ReaderSeekerCloser) GetLen() (int64, error) {
	if l, ok := r.HasLen(); ok {
		return int64(l), nil
	}

	if s, ok := r.r.(io.Seeker); ok {
		return seekerLen(s)
	}

	return -1, nil
}

// SeekerLen attempts to get the number of bytes remaining at the seeker's
// current position.  Returns the number of bytes remaining or error.
func SeekerLen(s io.Seeker) (int64, error) {
	// Determine if the seeker is actually seekable. ReaderSeekerCloser
	// hides the fact that a io.Readers might not actually be seekable.
	switch v := s.(type) {
	case ReaderSeekerCloser:
		return v.GetLen()
	case *ReaderSeekerCloser:
		return v.GetLen()
	}

	return seekerLen(s)
}

func seekerLen(s io.Seeker) (int64, error) {
	curOffset, err := s.Seek(0, 1)
	if err != nil {
		return 0, err
	}

	endOffset, err := s.Seek(0, 2)
	if err != nil {
		return 0, err
	}

	_, err = s.Seek(curOffset, 0)
	if err != nil {
		return 0, err
	}

	return endOffset - curOffset, nil
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

	// GrowthCoeff defines the growth rate of the internal buffer. By
	// default, the growth rate is 1, where expanding the internal
	// buffer will allocate only enough capacity to fit the new expected
	// length.
	GrowthCoeff float64
}

// NewWriteAtBuffer creates a WriteAtBuffer with an internal buffer
// provided by buf.
func NewWriteAtBuffer(buf []byte) *WriteAtBuffer {
	return &WriteAtBuffer{buf: buf}
}

// WriteAt writes a slice of bytes to a buffer starting at the position provided
// The number of bytes written will be returned, or error. Can overwrite previous
// written slices if the write ats overlap.
func (b *WriteAtBuffer) WriteAt(p []byte, pos int64) (n int, err error) {
	pLen := len(p)
	expLen := pos + int64(pLen)
	b.m.Lock()
	defer b.m.Unlock()
	if int64(len(b.buf)) < expLen {
		if int64(cap(b.buf)) < expLen {
			if b.GrowthCoeff < 1 {
				b.GrowthCoeff = 1
			}
			newBuf := make([]byte, expLen, int64(b.GrowthCoeff*float64(expLen)))
			copy(newBuf, b.buf)
			b.buf = newBuf
		}
		b.buf = b.buf[:expLen]
	}
	copy(b.buf[pos:], p)
	return pLen, nil
}

// Bytes returns a slice of bytes written to the buffer.
func (b *WriteAtBuffer) Bytes() []byte {
	b.m.Lock()
	defer b.m.Unlock()
	return b.buf
}
