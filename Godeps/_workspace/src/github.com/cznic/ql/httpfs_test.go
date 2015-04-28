// Copyright (c) 2014 ql Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ql

import (
	"io"
	"os"
	"path"
	"testing"
)

func TestHTTP(t *testing.T) {
	db, err := OpenMem()
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err = db.Run(
		NewRWCtx(),
		`
		BEGIN TRANSACTION;
			CREATE TABLE t (path string);
			INSERT INTO t VALUES
				("/a/b/c/1"),
				("/a/b/c/2"),
				("/a/b/3"),
				("/a/b/4"),
				("/a/5"),
				("/a/6"),
			;
		COMMIT;
		`,
	); err != nil {
		t.Fatal(err)
	}

	fs, err := db.NewHTTPFS("SELECT path, blob(path+`-c`) AS content FROM t")
	if err != nil {
		t.Fatal(err)
	}

	for _, nm := range []string{"/a/b/c/1", "/a/b/4", "/a/5"} {
		f, err := fs.Open(nm)
		if err != nil {
			t.Fatal(err)
		}

		stat, err := f.Stat()
		if err != nil {
			t.Fatal(err)
		}

		b := make([]byte, 100)
		n, err := f.Read(b)
		if err != nil {
			t.Fatal(nm, n, err)
		}

		g := string(b[:n])
		if e := nm + "-c"; g != e {
			t.Fatal(g, e)
		}

		if g, e := stat.Name(), path.Base(nm); g != e {
			t.Fatal(g, e)
		}

		if g, e := stat.Size(), int64(len(g)); g != e {
			t.Fatal(g, e)
		}

		if g, e := stat.IsDir(), false; g != e {
			t.Fatal(g, e)
		}

		b = make([]byte, 100)
		n, err = f.Read(b)
		if n != 0 {
			t.Error(n)
		}

		if err != io.EOF {
			t.Fatal(err)
		}

		if n, err := f.Seek(0, 0); err != nil || n != 0 {
			t.Fatal(n, err)
		}

		exp := []byte(nm + "-c")
		b = make([]byte, 1)
		for _, e := range exp {
			n, err := f.Read(b)
			if n != 1 || err != nil {
				t.Fatal(n, err)
			}

			if g := b[0]; g != e {
				t.Fatal(g, e)
			}
		}
		if n, err := f.Read(b); n != 0 || err != io.EOF {
			t.Fatal(n, err)
		}

		if err = f.Close(); err != nil {
			t.Fatal(err)
		}

		if n, err := f.Seek(0, 0); err != os.ErrInvalid {
			t.Fatal(n, err)
		}
	}

	if _, err = fs.Open("nonexistent"); err != os.ErrNotExist {
		t.Fatal(err)
	}

	// ------------------------------------------------------------------ /
	d, err := fs.Open("")
	if err != nil {
		t.Fatal(err)
	}

	stat, err := d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err := d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 1; g != e {
		t.Fatal(g, e)
	}

	var a bool
	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "a":
				a = true
			default:
				t.Fatal(v.Name())
			}
		default:
			t.Fatal(v.IsDir())
		}
	}
	if !a {
		t.Fatal(a)
	}

	// ------------------------------------------------------------------ a
	d, err = fs.Open("a")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	var aB, a5, a6 bool
	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "b":
				aB = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "5":
				a5 = true
			case "6":
				a6 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB && a5 && a6) {
		t.Fatal(aB, a5, a6)
	}

	// ----------------------------------------------------------------- a/
	d, err = fs.Open("a/")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "b":
				aB = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "5":
				a5 = true
			case "6":
				a6 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB && a5 && a6) {
		t.Fatal(aB, a5, a6)
	}

	// ----------------------------------------------------------------- /a
	d, err = fs.Open("/a")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "b":
				aB = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "5":
				a5 = true
			case "6":
				a6 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB && a5 && a6) {
		t.Fatal(aB, a5, a6)
	}

	// ---------------------------------------------------------------- /a/
	d, err = fs.Open("/a/")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "b":
				aB = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "5":
				a5 = true
			case "6":
				a6 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB && a5 && a6) {
		t.Fatal(aB, a5, a6)
	}

	// ---------------------------------------------------------------- a/b
	d, err = fs.Open("a/b")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	var aB3, aB4, aBC bool
	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "c":
				aBC = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "3":
				aB3 = true
			case "4":
				aB4 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB3 && aB4 && aBC) {
		t.Fatal(aB4, aB4, aBC)
	}

	// --------------------------------------------------------------- a/b/
	d, err = fs.Open("a/b/")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "c":
				aBC = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "3":
				aB3 = true
			case "4":
				aB4 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB3 && aB4 && aBC) {
		t.Fatal(aB4, aB4, aBC)
	}

	// -------------------------------------------------------------- /a/b/
	d, err = fs.Open("/a/b/")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "c":
				aBC = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "3":
				aB3 = true
			case "4":
				aB4 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB3 && aB4 && aBC) {
		t.Fatal(aB4, aB4, aBC)
	}

	// --------------------------------------------------------------- /a/b
	d, err = fs.Open("/a/b")
	if err != nil {
		t.Fatal(err)
	}

	stat, err = d.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if g, e := stat.IsDir(), true; g != e {
		t.Fatal(g, e)
	}

	list, err = d.Readdir(0)
	if err != nil {
		t.Fatal(err)
	}

	if g, e := len(list), 3; g != e {
		t.Fatal(g, e)
	}

	for _, v := range list {
		switch v.IsDir() {
		case true:
			switch v.Name() {
			case "c":
				aBC = true
			default:
				t.Fatal(v.Name())
			}
		default:
			switch v.Name() {
			case "3":
				aB3 = true
			case "4":
				aB4 = true
			default:
				t.Fatal(v.Name())
			}
		}
	}
	if !(aB3 && aB4 && aBC) {
		t.Fatal(aB4, aB4, aBC)
	}
}
