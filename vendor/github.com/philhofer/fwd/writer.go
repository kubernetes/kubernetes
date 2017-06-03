package fwd

import "io"

const (
	// DefaultWriterSize is the
	// default write buffer size.
	DefaultWriterSize = 2048

	minWriterSize = minReaderSize
)

// Writer is a buffered writer
type Writer struct {
	w   io.Writer // writer
	buf []byte    // 0:len(buf) is bufered data
}

// NewWriter returns a new writer
// that writes to 'w' and has a buffer
// that is `DefaultWriterSize` bytes.
func NewWriter(w io.Writer) *Writer {
	if wr, ok := w.(*Writer); ok {
		return wr
	}
	return &Writer{
		w:   w,
		buf: make([]byte, 0, DefaultWriterSize),
	}
}

// NewWriterSize returns a new writer
// that writes to 'w' and has a buffer
// that is 'size' bytes.
func NewWriterSize(w io.Writer, size int) *Writer {
	if wr, ok := w.(*Writer); ok && cap(wr.buf) >= size {
		return wr
	}
	return &Writer{
		w:   w,
		buf: make([]byte, 0, max(size, minWriterSize)),
	}
}

// Buffered returns the number of buffered bytes
// in the reader.
func (w *Writer) Buffered() int { return len(w.buf) }

// BufferSize returns the maximum size of the buffer.
func (w *Writer) BufferSize() int { return cap(w.buf) }

// Flush flushes any buffered bytes
// to the underlying writer.
func (w *Writer) Flush() error {
	l := len(w.buf)
	if l > 0 {
		n, err := w.w.Write(w.buf)

		// if we didn't write the whole
		// thing, copy the unwritten
		// bytes to the beginnning of the
		// buffer.
		if n < l && n > 0 {
			w.pushback(n)
			if err == nil {
				err = io.ErrShortWrite
			}
		}
		if err != nil {
			return err
		}
		w.buf = w.buf[:0]
		return nil
	}
	return nil
}

// Write implements `io.Writer`
func (w *Writer) Write(p []byte) (int, error) {
	c, l, ln := cap(w.buf), len(w.buf), len(p)
	avail := c - l

	// requires flush
	if avail < ln {
		if err := w.Flush(); err != nil {
			return 0, err
		}
		l = len(w.buf)
	}
	// too big to fit in buffer;
	// write directly to w.w
	if c < ln {
		return w.w.Write(p)
	}

	// grow buf slice; copy; return
	w.buf = w.buf[:l+ln]
	return copy(w.buf[l:], p), nil
}

// WriteString is analogous to Write, but it takes a string.
func (w *Writer) WriteString(s string) (int, error) {
	c, l, ln := cap(w.buf), len(w.buf), len(s)
	avail := c - l

	// requires flush
	if avail < ln {
		if err := w.Flush(); err != nil {
			return 0, err
		}
		l = len(w.buf)
	}
	// too big to fit in buffer;
	// write directly to w.w
	//
	// yes, this is unsafe. *but*
	// io.Writer is not allowed
	// to mutate its input or
	// maintain a reference to it,
	// per the spec in package io.
	//
	// plus, if the string is really
	// too big to fit in the buffer, then
	// creating a copy to write it is
	// expensive (and, strictly speaking,
	// unnecessary)
	if c < ln {
		return w.w.Write(unsafestr(s))
	}

	// grow buf slice; copy; return
	w.buf = w.buf[:l+ln]
	return copy(w.buf[l:], s), nil
}

// WriteByte implements `io.ByteWriter`
func (w *Writer) WriteByte(b byte) error {
	if len(w.buf) == cap(w.buf) {
		if err := w.Flush(); err != nil {
			return err
		}
	}
	w.buf = append(w.buf, b)
	return nil
}

// Next returns the next 'n' free bytes
// in the write buffer, flushing the writer
// as necessary. Next will return `io.ErrShortBuffer`
// if 'n' is greater than the size of the write buffer.
// Calls to 'next' increment the write position by
// the size of the returned buffer.
func (w *Writer) Next(n int) ([]byte, error) {
	c, l := cap(w.buf), len(w.buf)
	if n > c {
		return nil, io.ErrShortBuffer
	}
	avail := c - l
	if avail < n {
		if err := w.Flush(); err != nil {
			return nil, err
		}
		l = len(w.buf)
	}
	w.buf = w.buf[:l+n]
	return w.buf[l:], nil
}

// take the bytes from w.buf[n:len(w.buf)]
// and put them at the beginning of w.buf,
// and resize to the length of the copied segment.
func (w *Writer) pushback(n int) {
	w.buf = w.buf[:copy(w.buf, w.buf[n:])]
}

// ReadFrom implements `io.ReaderFrom`
func (w *Writer) ReadFrom(r io.Reader) (int64, error) {
	// anticipatory flush
	if err := w.Flush(); err != nil {
		return 0, err
	}

	w.buf = w.buf[0:cap(w.buf)] // expand buffer

	var nn int64  // written
	var err error // error
	var x int     // read

	// 1:1 reads and writes
	for err == nil {
		x, err = r.Read(w.buf)
		if x > 0 {
			n, werr := w.w.Write(w.buf[:x])
			nn += int64(n)

			if err != nil {
				if n < x && n > 0 {
					w.pushback(n - x)
				}
				return nn, werr
			}
			if n < x {
				w.pushback(n - x)
				return nn, io.ErrShortWrite
			}
		} else if err == nil {
			err = io.ErrNoProgress
			break
		}
	}
	if err != io.EOF {
		return nn, err
	}

	// we only clear here
	// because we are sure
	// the writes have
	// succeeded. otherwise,
	// we retain the data in case
	// future writes succeed.
	w.buf = w.buf[0:0]

	return nn, nil
}
