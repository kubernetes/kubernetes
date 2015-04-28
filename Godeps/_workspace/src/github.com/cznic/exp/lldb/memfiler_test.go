// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lldb

import (
	"bytes"
	"math/rand"
	"testing"
)

// Test automatic page releasing (hole punching) of zero pages
func TestMemFilerWriteAt(t *testing.T) {
	f := NewMemFiler()

	// Add page index 0
	if _, err := f.WriteAt([]byte{1}, 0); err != nil {
		t.Fatal(err)
	}

	if g, e := len(f.m), 1; g != e {
		t.Fatal(g, e)
	}

	// Add page index 1
	if _, err := f.WriteAt([]byte{2}, pgSize); err != nil {
		t.Fatal(err)
	}

	if g, e := len(f.m), 2; g != e {
		t.Fatal(g, e)
	}

	// Add page index 2
	if _, err := f.WriteAt([]byte{3}, 2*pgSize); err != nil {
		t.Fatal(err)
	}

	if g, e := len(f.m), 3; g != e {
		t.Fatal(g, e)
	}

	// Remove page index 1
	if _, err := f.WriteAt(make([]byte, 2*pgSize), pgSize/2); err != nil {
		t.Fatal(err)
	}

	if g, e := len(f.m), 2; g != e {
		t.Logf("%#v", f.m)
		t.Fatal(g, e)
	}

	if err := f.Truncate(1); err != nil {
		t.Fatal(err)
	}

	if g, e := len(f.m), 1; g != e {
		t.Logf("%#v", f.m)
		t.Fatal(g, e)
	}

	if err := f.Truncate(0); err != nil {
		t.Fatal(err)
	}

	if g, e := len(f.m), 0; g != e {
		t.Logf("%#v", f.m)
		t.Fatal(g, e)
	}
}

func TestMemFilerWriteTo(t *testing.T) {
	const max = 1e5
	var b [max]byte
	rng := rand.New(rand.NewSource(42))
	for sz := 0; sz < 1e5; sz += 2053 {
		for i := range b[:sz] {
			b[i] = byte(rng.Int())
		}
		f := NewMemFiler()
		if n, err := f.WriteAt(b[:sz], 0); n != sz || err != nil {
			t.Fatal(n, err)
		}

		var buf bytes.Buffer
		if n, err := f.WriteTo(&buf); n != int64(sz) || err != nil {
			t.Fatal(n, err)
		}

		if !bytes.Equal(b[:sz], buf.Bytes()) {
			t.Fatal("content differs")
		}
	}
}

func TestMemFilerReadFromWriteTo(t *testing.T) {
	const (
		sz   = 1e2 * pgSize
		hole = 1e1 * pgSize
	)
	rng := rand.New(rand.NewSource(42))
	data := make([]byte, sz)
	for i := range data {
		data[i] = byte(rng.Int())
	}
	f := NewMemFiler()
	buf := bytes.NewBuffer(data)
	if n, err := f.ReadFrom(buf); n != int64(len(data)) || err != nil {
		t.Fatal(n, err)
	}

	buf = bytes.NewBuffer(nil)
	if n, err := f.WriteTo(buf); n != int64(len(data)) || err != nil {
		t.Fatal(n, err)
	}

	rd := buf.Bytes()
	if !bytes.Equal(data, rd) {
		t.Fatal("corrupted data")
	}

	n0 := len(f.m)
	data = make([]byte, hole)
	f.WriteAt(data, sz/2)
	n := len(f.m)
	t.Log(n0, n)
	d := n0 - n
	if d*pgSize < hole-2 || d*pgSize > hole {
		t.Fatal(n0, n, d)
	}
}
