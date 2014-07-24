// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package utils

import (
	"bytes"
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
	n, err := StdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(19 + 12 + 6); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened!" {
		t.Errorf("StdCopy: wrong stderr. Want %q. Got %q.", "something happened!", got)
	}
	if got := stdout.String(); got != "just kidding\nyeah!" {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q.", "just kidding\nyeah!", got)
	}
}

func TestStdCopyStress(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	value := strings.Repeat("something ", 4096)
	writer := NewStdWriter(&input, Stdout)
	writer.Write([]byte(value))
	n, err := StdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if n != 40960 {
		t.Errorf("Wrong number of bytes. Want 40960. Got %d.", n)
	}
	if got := stderr.String(); got != "" {
		t.Errorf("StdCopy: wrong stderr. Want empty string. Got %q", got)
	}
	if got := stdout.String(); got != value {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q", value, got)
	}
}

func TestStdCopyInvalidStdHeader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{3, 0, 0, 0, 0, 0, 0, 19})
	n, err := StdCopy(&stdout, &stderr, &input)
	if n != 0 {
		t.Errorf("StdCopy: wrong number of bytes. Want 0. Got %d", n)
	}
	if err != ErrInvalidStdHeader {
		t.Errorf("StdCopy: wrong error. Want ErrInvalidStdHeader. Got %#v", err)
	}
}

func TestStdCopyBigFrame(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 18})
	input.Write([]byte("something happened!"))
	n, err := StdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(18); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened" {
		t.Errorf("StdCopy: wrong stderr. Want %q. Got %q.", "something happened", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopySmallFrame(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 20})
	input.Write([]byte("something happened!"))
	n, err := StdCopy(&stdout, &stderr, &input)
	if err != io.ErrShortWrite {
		t.Errorf("StdCopy: wrong error. Want ShortWrite. Got %#v", err)
	}
	if expected := int64(19); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened!" {
		t.Errorf("StdCopy: wrong stderr. Want %q. Got %q.", "something happened", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyEmpty(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	n, err := StdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("StdCopy: wrong number of bytes. Want 0. Got %d.", n)
	}
}

func TestStdCopyCorruptedHeader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0})
	n, err := StdCopy(&stdout, &stderr, &input)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("StdCopy: wrong number of bytes. Want 0. Got %d.", n)
	}
}

func TestStdCopyTruncateWriter(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	n, err := StdCopy(&stdout, iotest.TruncateWriter(&stderr, 7), &input)
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(19); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "somethi" {
		t.Errorf("StdCopy: wrong stderr. Want %q. Got %q.", "somethi", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyHeaderOnly(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	n, err := StdCopy(&stdout, iotest.TruncateWriter(&stderr, 7), &input)
	if err != io.ErrShortWrite {
		t.Errorf("StdCopy: wrong error. Want ShortWrite. Got %#v", err)
	}
	if n != 0 {
		t.Errorf("Wrong number of bytes. Want 0. Got %d.", n)
	}
	if got := stderr.String(); got != "" {
		t.Errorf("StdCopy: wrong stderr. Want %q. Got %q.", "", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyDataErrReader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	n, err := StdCopy(&stdout, &stderr, iotest.DataErrReader(&input))
	if err != nil {
		t.Fatal(err)
	}
	if expected := int64(19); n != expected {
		t.Errorf("Wrong number of bytes. Want %d. Got %d.", expected, n)
	}
	if got := stderr.String(); got != "something happened!" {
		t.Errorf("StdCopy: wrong stderr. Want %q. Got %q.", "something happened!", got)
	}
	if got := stdout.String(); got != "" {
		t.Errorf("StdCopy: wrong stdout. Want %q. Got %q.", "", got)
	}
}

func TestStdCopyTimeoutReader(t *testing.T) {
	var input, stdout, stderr bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	_, err := StdCopy(&stdout, &stderr, iotest.TimeoutReader(&input))
	if err != iotest.ErrTimeout {
		t.Errorf("StdCopy: wrong error. Want ErrTimeout. Got %#v.", err)
	}
}

func TestStdCopyWriteError(t *testing.T) {
	var input bytes.Buffer
	input.Write([]byte{2, 0, 0, 0, 0, 0, 0, 19})
	input.Write([]byte("something happened!"))
	var stdout, stderr errorWriter
	n, err := StdCopy(stdout, stderr, &input)
	if err.Error() != "something went wrong" {
		t.Errorf("StdCopy: wrong error. Want %q. Got %q", "something went wrong", err)
	}
	if n != 0 {
		t.Errorf("StdCopy: wrong number of bytes. Want 0. Got %d.", n)
	}
}
