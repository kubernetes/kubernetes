/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"strings"

	"github.com/go-logr/logr"
)

// LineBuffer is an interface for writing lines of input to a bytes.Buffer
type LineBuffer interface {
	// Write takes a list of arguments, each a string or []string, joins all the
	// individual strings with spaces, terminates with newline, and writes them to the
	// buffer. Any other argument type will panic.
	Write(args ...interface{})

	// WriteBytes writes bytes to the buffer, and terminates with newline.
	WriteBytes(bytes []byte)

	// Reset clears the buffer
	Reset()

	// Bytes returns the contents of the buffer as a []byte
	Bytes() []byte

	// String returns the contents of the buffer as a string
	String() string

	// Lines returns the number of lines in the buffer. Note that more precisely, this
	// returns the number of times Write() or WriteBytes() was called; it assumes that
	// you never wrote any newlines to the buffer yourself.
	Lines() int
}

var _ logr.Marshaler = &realLineBuffer{}

type realLineBuffer struct {
	b     bytes.Buffer
	lines int
}

// NewLineBuffer returns a new "real" LineBuffer
func NewLineBuffer() LineBuffer {
	return &realLineBuffer{}
}

// Write is part of LineBuffer
func (buf *realLineBuffer) Write(args ...interface{}) {
	for i, arg := range args {
		if i > 0 {
			buf.b.WriteByte(' ')
		}
		switch x := arg.(type) {
		case string:
			buf.b.WriteString(x)
		case []string:
			for j, s := range x {
				if j > 0 {
					buf.b.WriteByte(' ')
				}
				buf.b.WriteString(s)
			}
		default:
			panic(fmt.Sprintf("unknown argument type: %T", x))
		}
	}
	buf.b.WriteByte('\n')
	buf.lines++
}

// WriteBytes is part of LineBuffer
func (buf *realLineBuffer) WriteBytes(bytes []byte) {
	buf.b.Write(bytes)
	buf.b.WriteByte('\n')
	buf.lines++
}

// Reset is part of LineBuffer
func (buf *realLineBuffer) Reset() {
	buf.b.Reset()
	buf.lines = 0
}

// Bytes is part of LineBuffer
func (buf *realLineBuffer) Bytes() []byte {
	return buf.b.Bytes()
}

// String is part of LineBuffer
func (buf *realLineBuffer) String() string {
	return buf.b.String()
}

// Lines is part of LineBuffer
func (buf *realLineBuffer) Lines() int {
	return buf.lines
}

// Implements the logs.Marshaler interface
func (buf *realLineBuffer) MarshalLog() any {
	return strings.Split(buf.b.String(), "\n")
}

type discardLineBuffer struct {
	lines int
}

// NewDiscardLineBuffer returns a dummy LineBuffer that counts the number of writes but
// throws away the data. (This is used for iptables proxy partial syncs, to keep track of
// how many rules we managed to avoid having to sync.)
func NewDiscardLineBuffer() LineBuffer {
	return &discardLineBuffer{}
}

// Write is part of LineBuffer
func (buf *discardLineBuffer) Write(args ...interface{}) {
	buf.lines++
}

// WriteBytes is part of LineBuffer
func (buf *discardLineBuffer) WriteBytes(bytes []byte) {
	buf.lines++
}

// Reset is part of LineBuffer
func (buf *discardLineBuffer) Reset() {
	buf.lines = 0
}

// Bytes is part of LineBuffer
func (buf *discardLineBuffer) Bytes() []byte {
	return []byte{}
}

// String is part of LineBuffer
func (buf *discardLineBuffer) String() string {
	return ""
}

// Lines is part of LineBuffer
func (buf *discardLineBuffer) Lines() int {
	return buf.lines
}
