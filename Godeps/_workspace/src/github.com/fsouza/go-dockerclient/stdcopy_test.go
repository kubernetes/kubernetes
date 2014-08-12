// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package docker

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"
)

type errorWriter struct {
}

func (errorWriter) Write([]byte) (int, error) {
	return 0, errors.New("something went wrong")
}

func TestStdCopy(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	input.Write([]byte{1, 0, 0, 0, 0, 0, 0, 12})
	input.Write([]byte("just kidding"))
	input.Write([]byte{0, 0, 0, 0, 0, 0, 0, 6})
	input.Write([]byte("\nyeah!"))
	n, err := stdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(19 + 12 + 6); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened!" {
		t.Errorf("stdCopy: wrong stderr. Want %q. Got %q.", "something happened!", got)
	}
	if got := stdout.String(); got != "just kidding\nyeah!" {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q.", "just kidding\nyeah!", got)
	}
}

func TestStdCopyStress(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	value := strings.Repeat("something ", 4096)
	writer := newStdWriter(&input, Stdout)
	writer.Write([]byte(value))
	n, err := stdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if n != 40960 {
		t.Errorf("Wrong number of bytes. Want 40960. Got %d.", n)
	}
	if got := stderr.String(); got != "" {
		t.Errorf("stdCopy: wrong stderr. Want empty string. Got %q", got)
	}
	if got := stdout.String(); got != value {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q", value, got)
	}
}

func TestStdCopyInvalidStdHeader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{3, 0, 0, 0, 0, 0, 0, 19})
	n, err := stdCopy(&stdout, &stderr, &input)
	if n != 0 {
		t.Errorf("stdCopy: wrong number of bytes. Want 0. Got %d", n)
	}
	if err != errInvalidStdHeader {
		t.Errorf("stdCopy: wrong error. Want ErrInvalidStdHeader. Got %#v", err)
	}
}

func TestStdCopyBigFrame(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 18})
	input.Write([]byte("something happened!"))
	n, err := stdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(18); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened" {
		t.Errorf("stdCopy: wrong stderr. Want %q. Got %q.", "something happened", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopySmallFrame(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 20})
	input.Write([]byte("something happened!"))
	n, err := stdCopy(&stdout, &stderr, &input)
	if err != io.ErrShortWrite {
		t.Errorf("stdCopy: wrong error. Want ShortWrite. Got %#v", err)
	}
	if expected := int64(19); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened!" {
		t.Errorf("stdCopy: wrong stderr. Want %q. Got %q.", "something happened", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyEmpty(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	n, err := stdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("stdCopy: wrong number of bytes. Want 0. Got %d.", n)
	}
}

func TestStdCopyCorruptedHeader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0})
	n, err := stdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("stdCopy: wrong number of bytes. Want 0. Got %d.", n)
	}
}

func TestStdCopyTruncateWriter(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	n, err := stdCopy(&stdout, iotest.TruncateWriter(&stderr, 7), &input)
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(19); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "somethi" {
		t.Errorf("stdCopy: wrong stderr. Want %q. Got %q.", "somethi", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyHeaderOnly(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	n, err := stdCopy(&stdout, iotest.TruncateWriter(&stderr, 7), &input)
	if err != io.ErrShortWrite {
		t.Errorf("stdCopy: wrong error. Want ShortWrite. Got %#v", err)
	}
	if n != 0 {
		t.Errorf("Wrong number of bytes. Want 0. Got %d.", n)
	}
	if got := stderr.String(); got != "" {
		t.Errorf("stdCopy: wrong stderr. Want %q. Got %q.", "", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyDataErrReader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	n, err := stdCopy(&stdout, &stderr, iotest.DataErrReader(&input))
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(19); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened!" {
		t.Errorf("stdCopy: wrong stderr. Want %q. Got %q.", "something happened!", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("stdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyTimeoutReader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	_, err := stdCopy(&stdout, &stderr, iotest.TimeoutReader(&input))
	if err != iotest.ErrTimeout {
		t.Errorf("stdCopy: wrong error. Want ErrTimeout. Got %#v.", err)
	}
}

func TestStdCopyWriteError(t *testing.T) {
	var input bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	var stdout, stderr errorWriter
	n, err := stdCopy(stdout, stderr, &input)
	if err.Error() != "something went wrong" {
		t.Errorf("stdCopy: wrong error. Want %q. Got %q", "something went wrong", err)
	}
	if n != 0 {
		t.Errorf("stdCopy: wrong number of bytes. Want 0. Got %d.", n)
	}
}

type StdType [8]byte

var (
	Stdin  = StdType{0: 0}
	Stdout = StdType{0: 1}
	Stderr = StdType{0: 2}
)

type StdWriter struct {
	io.Writer
	prefix  StdType
	sizeBuf []byte
}

func (w *StdWriter) Write(buf []byte) (n int, err error) {
	if w == nil || w.Writer == nil {
		return 0, errors.New("Writer not instanciated")
	}
	binary.BigEndian.PutUint32(w.prefix[4:], uint32(len(buf)))
	buf = append(w.prefix[:], buf...)

	n, err = w.Writer.Write(buf)
	return n - 8, err
}

func newStdWriter(w io.Writer, t StdType) *StdWriter {
	if len(t) != 8 {
		return nil
	}

	return &StdWriter{
		Writer:  w,
		prefix:  t,
		sizeBuf: make([]byte, 4),
	}
}
