package ioutils

import "io"

type NopWriter struct{}

func (*NopWriter) Write(buf []byte) (int, error) {
	return len(buf), nil
}

type nopWriteCloser struct {
	io.Writer
}

func (w *nopWriteCloser) Close() error { return nil }

func NopWriteCloser(w io.Writer) io.WriteCloser {
	return &nopWriteCloser{w}
}

type NopFlusher struct{}

func (f *NopFlusher) Flush() {}

type writeCloserWrapper struct {
	io.Writer
	closer func() error
}

func (r *writeCloserWrapper) Close() error {
	return r.closer()
}

func NewWriteCloserWrapper(r io.Writer, closer func() error) io.WriteCloser {
	return &writeCloserWrapper{
		Writer: r,
		closer: closer,
	}
}

// Wrap a concrete io.Writer and hold a count of the number
// of bytes written to the writer during a "session".
// This can be convenient when write return is masked
// (e.g., json.Encoder.Encode())
type WriteCounter struct {
	Count  int64
	Writer io.Writer
}

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
