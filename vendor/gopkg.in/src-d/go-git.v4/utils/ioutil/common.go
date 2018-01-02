// Package ioutil implements some I/O utility functions.
package ioutil

import (
	"bufio"
	"context"
	"errors"
	"io"

	"github.com/jbenet/go-context/io"
)

type readPeeker interface {
	io.Reader
	Peek(int) ([]byte, error)
}

var (
	ErrEmptyReader = errors.New("reader is empty")
)

// NonEmptyReader takes a reader and returns it if it is not empty, or
// `ErrEmptyReader` if it is empty. If there is an error when reading the first
// byte of the given reader, it will be propagated.
func NonEmptyReader(r io.Reader) (io.Reader, error) {
	pr, ok := r.(readPeeker)
	if !ok {
		pr = bufio.NewReader(r)
	}

	_, err := pr.Peek(1)
	if err == io.EOF {
		return nil, ErrEmptyReader
	}

	if err != nil {
		return nil, err
	}

	return pr, nil
}

type readCloser struct {
	io.Reader
	closer io.Closer
}

func (r *readCloser) Close() error {
	return r.closer.Close()
}

// NewReadCloser creates an `io.ReadCloser` with the given `io.Reader` and
// `io.Closer`.
func NewReadCloser(r io.Reader, c io.Closer) io.ReadCloser {
	return &readCloser{Reader: r, closer: c}
}

type writeCloser struct {
	io.Writer
	closer io.Closer
}

func (r *writeCloser) Close() error {
	return r.closer.Close()
}

// NewWriteCloser creates an `io.WriteCloser` with the given `io.Writer` and
// `io.Closer`.
func NewWriteCloser(w io.Writer, c io.Closer) io.WriteCloser {
	return &writeCloser{Writer: w, closer: c}
}

type writeNopCloser struct {
	io.Writer
}

func (writeNopCloser) Close() error { return nil }

// WriteNopCloser returns a WriteCloser with a no-op Close method wrapping
// the provided Writer w.
func WriteNopCloser(w io.Writer) io.WriteCloser {
	return writeNopCloser{w}
}

// CheckClose calls Close on the given io.Closer. If the given *error points to
// nil, it will be assigned the error returned by Close. Otherwise, any error
// returned by Close will be ignored. CheckClose is usually called with defer.
func CheckClose(c io.Closer, err *error) {
	if cerr := c.Close(); cerr != nil && *err == nil {
		*err = cerr
	}
}

// NewContextWriter wraps a writer to make it respect given Context.
// If there is a blocking write, the returned Writer will return whenever the
// context is cancelled (the return values are n=0 and err=ctx.Err()).
func NewContextWriter(ctx context.Context, w io.Writer) io.Writer {
	return ctxio.NewWriter(ctx, w)
}

// NewContextReader wraps a reader to make it respect given Context.
// If there is a blocking read, the returned Reader will return whenever the
// context is cancelled (the return values are n=0 and err=ctx.Err()).
func NewContextReader(ctx context.Context, r io.Reader) io.Reader {
	return ctxio.NewReader(ctx, r)
}

// NewContextWriteCloser as NewContextWriter but with io.Closer interface.
func NewContextWriteCloser(ctx context.Context, w io.WriteCloser) io.WriteCloser {
	ctxw := ctxio.NewWriter(ctx, w)
	return NewWriteCloser(ctxw, w)
}

// NewContextReadCloser as NewContextReader but with io.Closer interface.
func NewContextReadCloser(ctx context.Context, r io.ReadCloser) io.ReadCloser {
	ctxr := ctxio.NewReader(ctx, r)
	return NewReadCloser(ctxr, r)
}

type readerOnError struct {
	io.Reader
	notify func(error)
}

// NewReaderOnError returns a io.Reader that call the notify function when an
// unexpected (!io.EOF) error happends, after call Read function.
func NewReaderOnError(r io.Reader, notify func(error)) io.Reader {
	return &readerOnError{r, notify}
}

// NewReadCloserOnError returns a io.ReadCloser that call the notify function
// when an unexpected (!io.EOF) error happends, after call Read function.
func NewReadCloserOnError(r io.ReadCloser, notify func(error)) io.ReadCloser {
	return NewReadCloser(NewReaderOnError(r, notify), r)
}

func (r *readerOnError) Read(buf []byte) (n int, err error) {
	n, err = r.Reader.Read(buf)
	if err != nil && err != io.EOF {
		r.notify(err)
	}

	return
}

type writerOnError struct {
	io.Writer
	notify func(error)
}

// NewWriterOnError returns a io.Writer that call the notify function when an
// unexpected (!io.EOF) error happends, after call Write function.
func NewWriterOnError(w io.Writer, notify func(error)) io.Writer {
	return &writerOnError{w, notify}
}

// NewWriteCloserOnError returns a io.WriteCloser that call the notify function
//when an unexpected (!io.EOF) error happends, after call Write function.
func NewWriteCloserOnError(w io.WriteCloser, notify func(error)) io.WriteCloser {
	return NewWriteCloser(NewWriterOnError(w, notify), w)
}

func (r *writerOnError) Write(p []byte) (n int, err error) {
	n, err = r.Writer.Write(p)
	if err != nil && err != io.EOF {
		r.notify(err)
	}

	return
}
