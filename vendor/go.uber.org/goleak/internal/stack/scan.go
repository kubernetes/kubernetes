// Copyright (c) 2023 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package stack

import (
	"bufio"
	"io"
)

// scanner provides a bufio.Scanner the ability to Unscan,
// which allows the current token to be read again
// after the next Scan.
type scanner struct {
	*bufio.Scanner

	unscanned bool
}

func newScanner(r io.Reader) *scanner {
	return &scanner{Scanner: bufio.NewScanner(r)}
}

func (s *scanner) Scan() bool {
	if s.unscanned {
		s.unscanned = false
		return true
	}
	return s.Scanner.Scan()
}

// Unscan stops the scanner from advancing its position
// for the next Scan.
//
// Bytes and Text will return the same token after next Scan
// that they do right now.
func (s *scanner) Unscan() {
	s.unscanned = true
}
