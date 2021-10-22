// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x
// +build zos,s390x

// This test is based on mmap_unix_test, but tweaked for z/OS, which does not support memadvise
// or anonymous mmapping.

package unix_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestMmap(t *testing.T) {
	tmpdir := mktmpdir(t)
	filename := filepath.Join(filepath.Join(tmpdir, "testdata"), "memmapped_file")
	destination, err := os.Create(filename)
	if err != nil {
		t.Fatal("os.Create:", err)
		return
	}
	defer os.RemoveAll(tmpdir)

	fmt.Fprintf(destination, "%s\n", "0 <- Flipped between 0 and 1 when test runs successfully")
	fmt.Fprintf(destination, "%s\n", "//Do not change contents - mmap test relies on this")
	destination.Close()
	fd, err := unix.Open(filename, unix.O_RDWR, 0777)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	b, err := unix.Mmap(fd, 0, 8, unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		t.Fatalf("Mmap: %v", err)
	}

	if err := unix.Mprotect(b, unix.PROT_READ|unix.PROT_WRITE); err != nil {
		t.Fatalf("Mprotect: %v", err)
	}

	// Flip flag in test file via mapped memory
	flagWasZero := true
	if b[0] == '0' {
		b[0] = '1'
	} else if b[0] == '1' {
		b[0] = '0'
		flagWasZero = false
	}

	if err := unix.Msync(b, unix.MS_SYNC); err != nil {
		t.Fatalf("Msync: %v", err)
	}

	// Read file from FS to ensure flag flipped after msync
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("Could not read mmapped file from disc for test: %v", err)
	}
	if flagWasZero && buf[0] != '1' || !flagWasZero && buf[0] != '0' {
		t.Error("Flag flip in MAP_SHARED mmapped file not visible")
	}

	if err := unix.Munmap(b); err != nil {
		t.Fatalf("Munmap: %v", err)
	}
}

func mktmpdir(t *testing.T) string {
	tmpdir, err := ioutil.TempDir("", "memmapped_file")
	if err != nil {
		t.Fatal("mktmpdir:", err)
	}
	if err := os.Mkdir(filepath.Join(tmpdir, "testdata"), 0700); err != nil {
		os.RemoveAll(tmpdir)
		t.Fatal("mktmpdir:", err)
	}
	return tmpdir
}
