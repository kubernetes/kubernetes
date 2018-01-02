package objfile

import (
	"compress/zlib"
	"errors"
	"io"
	"strconv"

	"gopkg.in/src-d/go-git.v4/plumbing"
)

var (
	ErrOverflow = errors.New("objfile: declared data length exceeded (overflow)")
)

// Writer writes and encodes data in compressed objfile format to a provided
// io.Writer. Close should be called when finished with the Writer. Close will
// not close the underlying io.Writer.
type Writer struct {
	raw    io.Writer
	zlib   io.WriteCloser
	hasher plumbing.Hasher
	multi  io.Writer

	closed  bool
	pending int64 // number of unwritten bytes
}

// NewWriter returns a new Writer writing to w.
//
// The returned Writer implements io.WriteCloser. Close should be called when
// finished with the Writer. Close will not close the underlying io.Writer.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		raw:  w,
		zlib: zlib.NewWriter(w),
	}
}

// WriteHeader writes the type and the size and prepares to accept the object's
// contents. If an invalid t is provided, plumbing.ErrInvalidType is returned. If a
// negative size is provided, ErrNegativeSize is returned.
func (w *Writer) WriteHeader(t plumbing.ObjectType, size int64) error {
	if !t.Valid() {
		return plumbing.ErrInvalidType
	}
	if size < 0 {
		return ErrNegativeSize
	}

	b := t.Bytes()
	b = append(b, ' ')
	b = append(b, []byte(strconv.FormatInt(size, 10))...)
	b = append(b, 0)

	defer w.prepareForWrite(t, size)
	_, err := w.zlib.Write(b)

	return err
}

func (w *Writer) prepareForWrite(t plumbing.ObjectType, size int64) {
	w.pending = size

	w.hasher = plumbing.NewHasher(t, size)
	w.multi = io.MultiWriter(w.zlib, w.hasher)
}

// Write writes the object's contents. Write returns the error ErrOverflow if
// more than size bytes are written after WriteHeader.
func (w *Writer) Write(p []byte) (n int, err error) {
	if w.closed {
		return 0, ErrClosed
	}

	overwrite := false
	if int64(len(p)) > w.pending {
		p = p[0:w.pending]
		overwrite = true
	}

	n, err = w.multi.Write(p)
	w.pending -= int64(n)
	if err == nil && overwrite {
		err = ErrOverflow
		return
	}

	return
}

// Hash returns the hash of the object data stream that has been written so far.
// It can be called before or after Close.
func (w *Writer) Hash() plumbing.Hash {
	return w.hasher.Sum() // Not yet closed, return hash of data written so far
}

// Close releases any resources consumed by the Writer.
//
// Calling Close does not close the wrapped io.Writer originally passed to
// NewWriter.
func (w *Writer) Close() error {
	if err := w.zlib.Close(); err != nil {
		return err
	}

	w.closed = true
	return nil
}
