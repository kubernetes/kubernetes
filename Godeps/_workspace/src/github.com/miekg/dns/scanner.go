package dns

// Implement a simple scanner, return a byte stream from an io reader.

import (
	"bufio"
	"io"
	"text/scanner"
)

type scan struct {
	src      *bufio.Reader
	position scanner.Position
	eof      bool // Have we just seen a eof
}

func scanInit(r io.Reader) *scan {
	s := new(scan)
	s.src = bufio.NewReader(r)
	s.position.Line = 1
	return s
}

// tokenText returns the next byte from the input
func (s *scan) tokenText() (byte, error) {
	c, err := s.src.ReadByte()
	if err != nil {
		return c, err
	}
	// delay the newline handling until the next token is delivered,
	// fixes off-by-one errors when reporting a parse error.
	if s.eof == true {
		s.position.Line++
		s.position.Column = 0
		s.eof = false
	}
	if c == '\n' {
		s.eof = true
		return c, nil
	}
	s.position.Column++
	return c, nil
}
