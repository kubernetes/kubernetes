// Copyright (c) 2011 CZ.NIC z.s.p.o. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// blame: jnml, labs.nic.cz

package storage

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func newfile(t *testing.T) (string, string, Accessor) {
	dir, err := ioutil.TempDir("", "test-storage-")
	if err != nil {
		panic(err)
	}

	name := filepath.Join(dir, "test.tmp")
	f, err := NewFile(name, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0666)
	if err != nil {
		t.Fatal("newfile", err)
	}

	return dir, name, f
}

func readfile(t *testing.T, name string) (b []byte) {
	var err error
	if b, err = ioutil.ReadFile(name); err != nil {
		t.Fatal("readfile")
	}

	return
}

func newcache(t *testing.T) (dir, name string, c *Cache) {
	dir, name, f := newfile(t)
	var err error
	if c, err = NewCache(f, 1<<20, nil); err != nil {
		t.Fatal("newCache", err)
	}

	return
}

func TestCache0(t *testing.T) {
	dir, name, c := newcache(t)
	defer os.RemoveAll(dir)

	if err := c.Close(); err != nil {
		t.Fatal(10, err)
	}

	if b := readfile(t, name); len(b) != 0 {
		t.Fatal(20, len(b), 0)
	}
}

func TestCache1(t *testing.T) {
	dir, name, c := newcache(t)
	defer os.RemoveAll(dir)

	if n, err := c.WriteAt([]byte{0xa5}, 0); n != 1 {
		t.Fatal(20, n, err)
	}

	if err := c.Close(); err != nil {
		t.Fatal(10, err)
	}

	b := readfile(t, name)
	if len(b) != 1 {
		t.Fatal(30, len(b), 1)
	}

	if b[0] != 0xa5 {
		t.Fatal(40, b[0], 0xa5)
	}
}
