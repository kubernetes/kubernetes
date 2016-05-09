// Taken from $GOROOT/src/pkg/net/http/chunked
// needed to write https responses to client.
package goproxy

import (
	"io"
	"strconv"
)

// newChunkedWriter returns a new chunkedWriter that translates writes into HTTP
// "chunked" format before writing them to w. Closing the returned chunkedWriter
// sends the final 0-length chunk that marks the end of the stream.
//
// newChunkedWriter is not needed by normal applications. The http
// package adds chunking automatically if handlers don't set a
// Content-Length header. Using newChunkedWriter inside a handler
// would result in double chunking or chunking with a Content-Length
// length, both of which are wrong.
func newChunkedWriter(w io.Writer) io.WriteCloser {
	return &chunkedWriter{w}
}

// Writing to chunkedWriter translates to writing in HTTP chunked Transfer
// Encoding wire format to the underlying Wire chunkedWriter.
type chunkedWriter struct {
	Wire io.Writer
}

// Write the contents of data as one chunk to Wire.
// NOTE: Note that the corresponding chunk-writing procedure in Conn.Write has
// a bug since it does not check for success of io.WriteString
func (cw *chunkedWriter) Write(data []byte) (n int, err error) {

	// Don't send 0-length data. It looks like EOF for chunked encoding.
	if len(data) == 0 {
		return 0, nil
	}

	head := strconv.FormatInt(int64(len(data)), 16) + "\r\n"

	if _, err = io.WriteString(cw.Wire, head); err != nil {
		return 0, err
	}
	if n, err = cw.Wire.Write(data); err != nil {
		return
	}
	if n != len(data) {
		err = io.ErrShortWrite
		return
	}
	_, err = io.WriteString(cw.Wire, "\r\n")

	return
}

func (cw *chunkedWriter) Close() error {
	_, err := io.WriteString(cw.Wire, "0\r\n")
	return err
}
