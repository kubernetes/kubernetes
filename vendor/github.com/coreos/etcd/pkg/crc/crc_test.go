// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc

import (
	"hash/crc32"
	"reflect"
	"testing"
)

// TestHash32 tests that Hash32 provided by this package can take an initial
// crc and behaves exactly the same as the standard one in the following calls.
func TestHash32(t *testing.T) {
	stdhash := crc32.New(crc32.IEEETable)
	if _, err := stdhash.Write([]byte("test data")); err != nil {
		t.Fatalf("unexpected write error: %v", err)
	}
	// create a new hash with stdhash.Sum32() as initial crc
	hash := New(stdhash.Sum32(), crc32.IEEETable)

	wsize := stdhash.Size()
	if g := hash.Size(); g != wsize {
		t.Errorf("size = %d, want %d", g, wsize)
	}
	wbsize := stdhash.BlockSize()
	if g := hash.BlockSize(); g != wbsize {
		t.Errorf("block size = %d, want %d", g, wbsize)
	}
	wsum32 := stdhash.Sum32()
	if g := hash.Sum32(); g != wsum32 {
		t.Errorf("Sum32 = %d, want %d", g, wsum32)
	}
	wsum := stdhash.Sum(make([]byte, 32))
	if g := hash.Sum(make([]byte, 32)); !reflect.DeepEqual(g, wsum) {
		t.Errorf("sum = %v, want %v", g, wsum)
	}

	// write something
	if _, err := stdhash.Write([]byte("test data")); err != nil {
		t.Fatalf("unexpected write error: %v", err)
	}
	if _, err := hash.Write([]byte("test data")); err != nil {
		t.Fatalf("unexpected write error: %v", err)
	}
	wsum32 = stdhash.Sum32()
	if g := hash.Sum32(); g != wsum32 {
		t.Errorf("Sum32 after write = %d, want %d", g, wsum32)
	}

	// reset
	stdhash.Reset()
	hash.Reset()
	wsum32 = stdhash.Sum32()
	if g := hash.Sum32(); g != wsum32 {
		t.Errorf("Sum32 after reset = %d, want %d", g, wsum32)
	}
}
