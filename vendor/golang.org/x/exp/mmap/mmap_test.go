// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mmap

import (
	"bytes"
	"io"
	"io/ioutil"
	"testing"
)

func TestOpen(t *testing.T) {
	const filename = "mmap_test.go"
	r, err := Open(filename)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	got := make([]byte, r.Len())
	if _, err := r.ReadAt(got, 0); err != nil && err != io.EOF {
		t.Fatalf("ReadAt: %v", err)
	}
	want, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("ioutil.ReadFile: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d bytes, want %d", len(got), len(want))
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("\ngot  %q\nwant %q", string(got), string(want))
	}
}
