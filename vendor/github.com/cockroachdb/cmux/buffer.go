package cmux

import (
	"bytes"
	"io"
)

// bufferedReader is an optimized implementation of io.Reader that behaves like
// ```
// io.MultiReader(bytes.NewReader(buffer.Bytes()), io.TeeReader(source, buffer))
// ```
// without allocating.
type bufferedReader struct {
	source     io.Reader
	buffer     *bytes.Buffer
	bufferRead int
	bufferSize int
}

func (s *bufferedReader) Read(p []byte) (int, error) {
	// Functionality of bytes.Reader.
	bn := copy(p, s.buffer.Bytes()[s.bufferRead:s.bufferSize])
	s.bufferRead += bn

	p = p[bn:]

	// Funtionality of io.TeeReader.
	sn, sErr := s.source.Read(p)
	if sn > 0 {
		if wn, wErr := s.buffer.Write(p[:sn]); wErr != nil {
			return bn + wn, wErr
		}
	}
	return bn + sn, sErr
}
