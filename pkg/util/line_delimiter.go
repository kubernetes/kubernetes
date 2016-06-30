/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"bytes"
	"io"
	"strings"
)

// A Line Delimiter is a filter that will
type LineDelimiter struct {
	output    io.Writer
	delimiter []byte
	buf       bytes.Buffer
}

// NewLineDelimiter allocates a new io.Writer that will split input on lines
// and bracket each line with the delimiter string.  This can be useful in
// output tests where it is difficult to see and test trailing whitespace.
func NewLineDelimiter(output io.Writer, delimiter string) *LineDelimiter {
	return &LineDelimiter{output: output, delimiter: []byte(delimiter)}
}

// Write writes buf to the LineDelimiter ld. The only errors returned are ones
// encountered while writing to the underlying output stream.
func (ld *LineDelimiter) Write(buf []byte) (n int, err error) {
	return ld.buf.Write(buf)
}

// Flush all lines up until now.  This will assume insert a linebreak at the current point of the stream.
func (ld *LineDelimiter) Flush() (err error) {
	lines := strings.Split(ld.buf.String(), "\n")
	for _, line := range lines {
		if _, err = ld.output.Write(ld.delimiter); err != nil {
			return
		}
		if _, err = ld.output.Write([]byte(line)); err != nil {
			return
		}
		if _, err = ld.output.Write(ld.delimiter); err != nil {
			return
		}
		if _, err = ld.output.Write([]byte("\n")); err != nil {
			return
		}
	}
	return
}
