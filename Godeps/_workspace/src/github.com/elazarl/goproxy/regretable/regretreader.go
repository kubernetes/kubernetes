package regretable

import (
	"io"
)

// A RegretableReader will allow you to read from a reader, and then
// to "regret" reading it, and push back everything you've read.
// For example,
//	rb := NewRegretableReader(bytes.NewBuffer([]byte{1,2,3}))
//	var b = make([]byte,1)
//	rb.Read(b) // b[0] = 1
//	rb.Regret()
//	ioutil.ReadAll(rb.Read) // returns []byte{1,2,3},nil
type RegretableReader struct {
	reader   io.Reader
	overflow bool
	r, w     int
	buf      []byte
}

var defaultBufferSize = 500

// Same as RegretableReader, but allows closing the underlying reader
type RegretableReaderCloser struct {
	RegretableReader
	c      io.Closer
}

// Closes the underlying readCloser, you cannot regret after closing the stream
func (rbc *RegretableReaderCloser) Close() error {
	return rbc.c.Close()
}

// initialize a RegretableReaderCloser with underlying readCloser rc
func NewRegretableReaderCloser(rc io.ReadCloser) *RegretableReaderCloser {
	return &RegretableReaderCloser{*NewRegretableReader(rc), rc}
}

// initialize a RegretableReaderCloser with underlying readCloser rc
func NewRegretableReaderCloserSize(rc io.ReadCloser, size int) *RegretableReaderCloser {
	return &RegretableReaderCloser{*NewRegretableReaderSize(rc, size), rc}
}

// The next read from the RegretableReader will be as if the underlying reader
// was never read (or from the last point forget is called).
func (rb *RegretableReader) Regret() {
	if rb.overflow {
		panic("regretting after overflow makes no sense")
	}
	rb.r = 0
}

// Will "forget" everything read so far.
//	rb := NewRegretableReader(bytes.NewBuffer([]byte{1,2,3}))
//	var b = make([]byte,1)
//	rb.Read(b) // b[0] = 1
//	rb.Forget()
//	rb.Read(b) // b[0] = 2
//	rb.Regret()
//	ioutil.ReadAll(rb.Read) // returns []byte{2,3},nil
func (rb *RegretableReader) Forget() {
	if rb.overflow {
		panic("forgetting after overflow makes no sense")
	}
	rb.r = 0
	rb.w = 0
}

// initialize a RegretableReader with underlying reader r, whose buffer is size bytes long
func NewRegretableReaderSize(r io.Reader, size int) *RegretableReader {
	return &RegretableReader{reader: r, buf: make([]byte, size) }
}

// initialize a RegretableReader with underlying reader r
func NewRegretableReader(r io.Reader) *RegretableReader {
	return NewRegretableReaderSize(r, defaultBufferSize)
}

// reads from the underlying reader. Will buffer all input until Regret is called.
func (rb *RegretableReader) Read(p []byte) (n int, err error) {
	if rb.overflow {
		return rb.reader.Read(p)
	}
	if rb.r < rb.w {
		n = copy(p, rb.buf[rb.r:rb.w])
		rb.r += n
		return
	}
	n, err = rb.reader.Read(p)
	bn := copy(rb.buf[rb.w:], p[:n])
	rb.w, rb.r = rb.w + bn, rb.w + n
	if bn < n {
		rb.overflow = true
	}
	return
}
