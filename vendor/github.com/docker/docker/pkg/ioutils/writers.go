package ioutils

import "io"

// NopWriter represents a type which write operation is nop.
type NopWriter struct{}

func (*NopWriter) Write(buf []byte) (int, error) {
	return len(buf), nil
}

type nopWriteCloser struct {
	io.Writer
}

func (w *nopWriteCloser) Close() error { return nil }

// NopWriteCloser returns a nopWriteCloser.
func NopWriteCloser(w io.Writer) io.WriteCloser {
	return &nopWriteCloser{w}
}

// NopFlusher represents a type which flush operation is nop.
type NopFlusher struct{}

// Flush is a nop operation.
func (f *NopFlusher) Flush() {}

type writeCloserWrapper struct {
	io.Writer
	closer func() error
}

func (r *writeCloserWrapper) Close() error {
	return r.closer()
}

// NewWriteCloserWrapper returns a new io.WriteCloser.
func NewWriteCloserWrapper(r io.Writer, closer func() error) io.WriteCloser {
	return &writeCloserWrapper{
		Writer: r,
		closer: closer,
	}
}

// WriteCounter wraps a concrete io.Writer and hold a count of the number
// of bytes written to the writer during a "session".
// This can be convenient when write return is masked
// (e.g., json.Encoder.Encode())
type WriteCounter struct {
	Count  int64
	Writer io.Writer
}

// NewWriteCounter returns a new WriteCounter.
func NewWriteCounter(w io.Writer) *WriteCounter {
	return &WriteCounter{
		Writer: w,
	}
}

func (wc *WriteCounter) Write(p []byte) (count int, err error) {
	count, err = wc.Writer.Write(p)
	wc.Count += int64(count)
	return
}
