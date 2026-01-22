package term

import (
	"io"
)

// EscapeError is special error which returned by a TTY proxy reader's Read()
// method in case its detach escape sequence is read.
type EscapeError struct{}

func (EscapeError) Error() string {
	return "read escape sequence"
}

// escapeProxy is used only for attaches with a TTY. It is used to proxy
// stdin keypresses from the underlying reader and look for the passed in
// escape key sequence to signal a detach.
type escapeProxy struct {
	escapeKeys   []byte
	escapeKeyPos int
	r            io.Reader
	buf          []byte
}

// NewEscapeProxy returns a new TTY proxy reader which wraps the given reader
// and detects when the specified escape keys are read, in which case the Read
// method will return an error of type EscapeError.
func NewEscapeProxy(r io.Reader, escapeKeys []byte) io.Reader {
	return &escapeProxy{
		escapeKeys: escapeKeys,
		r:          r,
	}
}

func (r *escapeProxy) Read(buf []byte) (n int, err error) {
	if len(r.escapeKeys) > 0 && r.escapeKeyPos == len(r.escapeKeys) {
		return 0, EscapeError{}
	}

	if len(r.buf) > 0 {
		n = copy(buf, r.buf)
		r.buf = r.buf[n:]
	}

	nr, err := r.r.Read(buf[n:])
	n += nr
	if len(r.escapeKeys) == 0 {
		return n, err
	}

	for i := 0; i < n; i++ {
		if buf[i] == r.escapeKeys[r.escapeKeyPos] {
			r.escapeKeyPos++

			// Check if the full escape sequence is matched.
			if r.escapeKeyPos == len(r.escapeKeys) {
				n = i + 1 - r.escapeKeyPos
				if n < 0 {
					n = 0
				}
				return n, EscapeError{}
			}
			continue
		}

		// If we need to prepend a partial escape sequence from the previous
		// read, make sure the new buffer size doesn't exceed len(buf).
		// Otherwise, preserve any extra data in a buffer for the next read.
		if i < r.escapeKeyPos {
			preserve := make([]byte, 0, r.escapeKeyPos+n)
			preserve = append(preserve, r.escapeKeys[:r.escapeKeyPos]...)
			preserve = append(preserve, buf[:n]...)
			n = copy(buf, preserve)
			i += r.escapeKeyPos
			r.buf = append(r.buf, preserve[n:]...)
		}
		r.escapeKeyPos = 0
	}

	// If we're in the middle of reading an escape sequence, make sure we don't
	// let the caller read it. If later on we find that this is not the escape
	// sequence, we'll prepend it back to buf.
	n -= r.escapeKeyPos
	if n < 0 {
		n = 0
	}
	return n, err
}
