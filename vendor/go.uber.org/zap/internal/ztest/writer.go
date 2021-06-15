// Copyright (c) 2016 Uber Technologies, Inc.
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

package ztest

import (
	"bytes"
	"errors"
	"io/ioutil"
	"strings"
)

// A Syncer is a spy for the Sync portion of zapcore.WriteSyncer.
type Syncer struct {
	err    error
	called bool
}

// SetError sets the error that the Sync method will return.
func (s *Syncer) SetError(err error) {
	s.err = err
}

// Sync records that it was called, then returns the user-supplied error (if
// any).
func (s *Syncer) Sync() error {
	s.called = true
	return s.err
}

// Called reports whether the Sync method was called.
func (s *Syncer) Called() bool {
	return s.called
}

// A Discarder sends all writes to ioutil.Discard.
type Discarder struct{ Syncer }

// Write implements io.Writer.
func (d *Discarder) Write(b []byte) (int, error) {
	return ioutil.Discard.Write(b)
}

// FailWriter is a WriteSyncer that always returns an error on writes.
type FailWriter struct{ Syncer }

// Write implements io.Writer.
func (w FailWriter) Write(b []byte) (int, error) {
	return len(b), errors.New("failed")
}

// ShortWriter is a WriteSyncer whose write method never fails, but
// nevertheless fails to the last byte of the input.
type ShortWriter struct{ Syncer }

// Write implements io.Writer.
func (w ShortWriter) Write(b []byte) (int, error) {
	return len(b) - 1, nil
}

// Buffer is an implementation of zapcore.WriteSyncer that sends all writes to
// a bytes.Buffer. It has convenience methods to split the accumulated buffer
// on newlines.
type Buffer struct {
	bytes.Buffer
	Syncer
}

// Lines returns the current buffer contents, split on newlines.
func (b *Buffer) Lines() []string {
	output := strings.Split(b.String(), "\n")
	return output[:len(output)-1]
}

// Stripped returns the current buffer contents with the last trailing newline
// stripped.
func (b *Buffer) Stripped() string {
	return strings.TrimRight(b.String(), "\n")
}
