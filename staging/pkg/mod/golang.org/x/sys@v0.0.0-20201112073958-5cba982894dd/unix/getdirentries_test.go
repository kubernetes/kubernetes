// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd openbsd netbsd

package unix_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func TestGetdirentries(t *testing.T) {
	for _, count := range []int{10, 1000} {
		t.Run(fmt.Sprintf("n=%d", count), func(t *testing.T) {
			testGetdirentries(t, count)
		})
	}
}
func testGetdirentries(t *testing.T, count int) {
	if count > 100 && testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		t.Skip("skipping in -short mode")
	}
	d, err := ioutil.TempDir("", "getdirentries-test")
	if err != nil {
		t.Fatalf("Tempdir: %v", err)
	}
	defer os.RemoveAll(d)
	var names []string
	for i := 0; i < count; i++ {
		names = append(names, fmt.Sprintf("file%03d", i))
	}

	// Make files in the temp directory
	for _, name := range names {
		err := ioutil.WriteFile(filepath.Join(d, name), []byte("data"), 0)
		if err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
	}

	// Read files using Getdirentries
	fd, err := unix.Open(d, unix.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer unix.Close(fd)
	var base uintptr
	var buf [2048]byte
	names2 := make([]string, 0, count)
	for {
		n, err := unix.Getdirentries(fd, buf[:], &base)
		if err != nil {
			t.Fatalf("Getdirentries: %v", err)
		}
		if n == 0 {
			break
		}
		data := buf[:n]
		for len(data) > 0 {
			var bc int
			bc, _, names2 = unix.ParseDirent(data, -1, names2)
			if bc == 0 && len(data) > 0 {
				t.Fatal("no progress")
			}
			data = data[bc:]
		}
	}

	sort.Strings(names)
	sort.Strings(names2)
	if strings.Join(names, ":") != strings.Join(names2, ":") {
		t.Errorf("names don't match\n names: %q\nnames2: %q", names, names2)
	}
}
