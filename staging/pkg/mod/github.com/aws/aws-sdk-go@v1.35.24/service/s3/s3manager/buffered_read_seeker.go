package s3manager

import (
	"io"

	"github.com/aws/aws-sdk-go/internal/sdkio"
)

// BufferedReadSeeker is buffered io.ReadSeeker
type BufferedReadSeeker struct {
	r                 io.ReadSeeker
	buffer            []byte
	readIdx, writeIdx int
}

// NewBufferedReadSeeker returns a new BufferedReadSeeker
// if len(b) == 0 then the buffer will be initialized to 64 KiB.
func NewBufferedReadSeeker(r io.ReadSeeker, b []byte) *BufferedReadSeeker {
	if len(b) == 0 {
		b = make([]byte, 64*1024)
	}
	return &BufferedReadSeeker{r: r, buffer: b}
}

func (b *BufferedReadSeeker) reset(r io.ReadSeeker) {
	b.r = r
	b.readIdx, b.writeIdx = 0, 0
}

// Read will read up len(p) bytes into p and will return
// the number of bytes read and any error that occurred.
// If the len(p) > the buffer size then a single read request
// will be issued to the underlying io.ReadSeeker for len(p) bytes.
// A Read request will at most perform a single Read to the underlying
// io.ReadSeeker, and may return < len(p) if serviced from the buffer.
func (b *BufferedReadSeeker) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return n, err
	}

	if b.readIdx == b.writeIdx {
		if len(p) >= len(b.buffer) {
			n, err = b.r.Read(p)
			return n, err
		}
		b.readIdx, b.writeIdx = 0, 0

		n, err = b.r.Read(b.buffer)
		if n == 0 {
			return n, err
		}

		b.writeIdx += n
	}

	n = copy(p, b.buffer[b.readIdx:b.writeIdx])
	b.readIdx += n

	return n, err
}

// Seek will position then underlying io.ReadSeeker to the given offset
// and will clear the buffer.
func (b *BufferedReadSeeker) Seek(offset int64, whence int) (int64, error) {
	n, err := b.r.Seek(offset, whence)

	b.reset(b.r)

	return n, err
}

// ReadAt will read up to len(p) bytes at the given file offset.
// This will result in the buffer being cleared.
func (b *BufferedReadSeeker) ReadAt(p []byte, off int64) (int, error) {
	_, err := b.Seek(off, sdkio.SeekStart)
	if err != nil {
		return 0, err
	}

	return b.Read(p)
}
