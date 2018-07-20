package framing

import (
	"io"
	"io/ioutil"
)

type Error string

func (err Error) Error() string { return string(err) }

const (
	ErrorUnderrun       = Error("frame underrun, unexpected EOF")
	ErrorBadSize        = Error("bad frame size")
	ErrorOversizedFrame = Error("oversized frame, max size exceeded")
)

type (
	// Reader generates data frames from some source, returning io.EOF when the end of the input stream is
	// detected.
	Reader interface {
		ReadFrame() (frame []byte, err error)
	}

	// ReaderFunc is the functional adaptation of Reader.
	ReaderFunc func() ([]byte, error)

	// Writer sends whole frames to some endpoint; returns io.ErrShortWrite if the frame is only partially written.
	Writer interface {
		WriteFrame(frame []byte) error
	}

	// WriterFunc is the functional adaptation of Writer.
	WriterFunc func([]byte) error
)

func (f ReaderFunc) ReadFrame() ([]byte, error) { return f() }
func (f WriterFunc) WriteFrame(b []byte) error  { return f(b) }

var _ = Reader(ReaderFunc(nil))
var _ = Writer(WriterFunc(nil))

// EOFReaderFunc always returns nil, io.EOF; it implements the ReaderFunc API.
func EOFReaderFunc() ([]byte, error) { return nil, io.EOF }

var _ = ReaderFunc(EOFReaderFunc) // sanity check

// ReadAll returns a reader func that returns the complete contents of `r` in a single frame.
// A zero length frame is treated as an "end of stream" condition, returning io.EOF.
func ReadAll(r io.Reader) ReaderFunc {
	return func() (b []byte, err error) {
		b, err = ioutil.ReadAll(r)
		if len(b) == 0 && err == nil {
			err = io.EOF
		}
		return
	}
}

// WriterFor adapts an io.Writer to the Writer interface. All buffers are written to `w` without decoration or
// modification.
func WriterFor(w io.Writer) WriterFunc {
	return func(b []byte) error {
		n, err := w.Write(b)
		if err == nil && n != len(b) {
			return io.ErrShortWrite
		}
		return err
	}
}
