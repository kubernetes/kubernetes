// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"hash/crc64"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

/*
The on-disk index is a text file.
The first 3 lines are header information containing CurrentVersion,
the value of GOMODCACHE, and the validity date of the index.
(This is when the code started building the index.)
Following the header are sections of lines, one section for each
import path. These sections are sorted by package name.
The first line of each section, marked by a leading :, contains
the package name, the import path, the name of the directory relative
to GOMODCACHE, and its semantic version.
The rest of each section consists of one line per exported symbol.
The lines are sorted by the symbol's name and contain the name,
an indication of its lexical type (C, T, V, F), and if it is the
name of a function, information about the signature.

The fields in the section header lines are separated by commas, and
in the unlikely event this would be confusing, the csv package is used
to write (and read) them.

In the lines containing exported names, C=const, V=var, T=type, F=func.
If it is a func, the next field is the number of returned values,
followed by pairs consisting of formal parameter names and types.
All these fields are separated by spaces. Any spaces in a type
(e.g., chan struct{}) are replaced by $s on the disk. The $s are
turned back into spaces when read.

Here is an index header (the comments are not part of the index):
0                                      // version (of the index format)
/usr/local/google/home/pjw/go/pkg/mod  // GOMODCACHE
2024-09-11 18:55:09                    // validity date of the index

Here is an index section:
:yaml,gopkg.in/yaml.v1,gopkg.in/yaml.v1@v1.0.0-20140924161607-9f9df34309c0,v1.0.0-20140924161607-9f9df34309c0
Getter T
Marshal F 2 in interface{}
Setter T
Unmarshal F 1 in []byte out interface{}

The package name is yaml, the import path is gopkg.in/yaml.v1.
Getter and Setter are types, and Marshal and Unmarshal are functions.
The latter returns one value and has two arguments, 'in' and 'out'
whose types are []byte and interface{}.
*/

// CurrentVersion tells readers about the format of the index.
const CurrentVersion int = 0

// Index is returned by ReadIndex().
type Index struct {
	Version  int
	Cachedir Abspath   // The directory containing the module cache
	Changed  time.Time // The index is up to date as of Changed
	Entries  []Entry
}

// An Entry contains information for an import path.
type Entry struct {
	Dir        Relpath // directory in modcache
	ImportPath string
	PkgName    string
	Version    string
	//ModTime    STime    // is this useful?
	Names []string // exported names and information
}

// IndexDir is where the module index is stored.
var IndexDir string

// Set IndexDir
func init() {
	var dir string
	var err error
	if testing.Testing() {
		dir = os.TempDir()
	} else {
		dir, err = os.UserCacheDir()
		// shouldn't happen, but TempDir is better than
		// creating ./go/imports
		if err != nil {
			dir = os.TempDir()
		}
	}
	dir = filepath.Join(dir, "go", "imports")
	os.MkdirAll(dir, 0777)
	IndexDir = dir
}

// ReadIndex reads the latest version of the on-disk index
// for the cache directory cd.
// It returns (nil, nil) if there is no index, but returns
// a non-nil error if the index exists but could not be read.
func ReadIndex(cachedir string) (*Index, error) {
	cachedir, err := filepath.Abs(cachedir)
	if err != nil {
		return nil, err
	}
	cd := Abspath(cachedir)
	dir := IndexDir
	base := indexNameBase(cd)
	iname := filepath.Join(dir, base)
	buf, err := os.ReadFile(iname)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("cannot read %s: %w", iname, err)
	}
	fname := filepath.Join(dir, string(buf))
	fd, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer fd.Close()
	r := bufio.NewReader(fd)
	ix, err := readIndexFrom(cd, r)
	if err != nil {
		return nil, err
	}
	return ix, nil
}

func readIndexFrom(cd Abspath, bx io.Reader) (*Index, error) {
	b := bufio.NewScanner(bx)
	var ans Index
	// header
	ok := b.Scan()
	if !ok {
		return nil, fmt.Errorf("unexpected scan error")
	}
	l := b.Text()
	var err error
	ans.Version, err = strconv.Atoi(l)
	if err != nil {
		return nil, err
	}
	if ans.Version != CurrentVersion {
		return nil, fmt.Errorf("got version %d, expected %d", ans.Version, CurrentVersion)
	}
	if ok := b.Scan(); !ok {
		return nil, fmt.Errorf("scanner error reading cachedir")
	}
	ans.Cachedir = Abspath(b.Text())
	if ok := b.Scan(); !ok {
		return nil, fmt.Errorf("scanner error reading index creation time")
	}
	// TODO(pjw): need to check that this is the expected cachedir
	// so the tag should be passed in to this function
	ans.Changed, err = time.ParseInLocation(time.DateTime, b.Text(), time.Local)
	if err != nil {
		return nil, err
	}
	var curEntry *Entry
	for b.Scan() {
		v := b.Text()
		if v[0] == ':' {
			if curEntry != nil {
				ans.Entries = append(ans.Entries, *curEntry)
			}
			// as directories may contain commas and quotes, they need to be read as csv.
			rdr := strings.NewReader(v[1:])
			cs := csv.NewReader(rdr)
			flds, err := cs.Read()
			if err != nil {
				return nil, err
			}
			if len(flds) != 4 {
				return nil, fmt.Errorf("header contains %d fields, not 4: %q", len(v), v)
			}
			curEntry = &Entry{PkgName: flds[0], ImportPath: flds[1], Dir: toRelpath(cd, flds[2]), Version: flds[3]}
			continue
		}
		curEntry.Names = append(curEntry.Names, v)
	}
	if curEntry != nil {
		ans.Entries = append(ans.Entries, *curEntry)
	}
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("scanner failed %v", err)
	}
	return &ans, nil
}

// write the index as a text file
func writeIndex(cachedir Abspath, ix *Index) error {
	ipat := fmt.Sprintf("index-%d-*", CurrentVersion)
	fd, err := os.CreateTemp(IndexDir, ipat)
	if err != nil {
		return err // can this happen?
	}
	defer fd.Close()
	if err := writeIndexToFile(ix, fd); err != nil {
		return err
	}
	content := fd.Name()
	content = filepath.Base(content)
	base := indexNameBase(cachedir)
	nm := filepath.Join(IndexDir, base)
	err = os.WriteFile(nm, []byte(content), 0666)
	if err != nil {
		return err
	}
	return nil
}

func writeIndexToFile(x *Index, fd *os.File) error {
	cnt := 0
	w := bufio.NewWriter(fd)
	fmt.Fprintf(w, "%d\n", x.Version)
	fmt.Fprintf(w, "%s\n", x.Cachedir)
	// round the time down
	tm := x.Changed.Add(-time.Second / 2)
	fmt.Fprintf(w, "%s\n", tm.Format(time.DateTime))
	for _, e := range x.Entries {
		if e.ImportPath == "" {
			continue // shouldn't happen
		}
		// PJW: maybe always write these headers as csv?
		if strings.ContainsAny(string(e.Dir), ",\"") {
			log.Printf("DIR: %s", e.Dir)
			cw := csv.NewWriter(w)
			cw.Write([]string{":" + e.PkgName, e.ImportPath, string(e.Dir), e.Version})
			cw.Flush()
		} else {
			fmt.Fprintf(w, ":%s,%s,%s,%s\n", e.PkgName, e.ImportPath, e.Dir, e.Version)
		}
		for _, x := range e.Names {
			fmt.Fprintf(w, "%s\n", x)
			cnt++
		}
	}
	if err := w.Flush(); err != nil {
		return err
	}
	return nil
}

// return the base name of the file containing the name of the current index
func indexNameBase(cachedir Abspath) string {
	// crc64 is a way to convert path names into 16 hex digits.
	h := crc64.Checksum([]byte(cachedir), crc64.MakeTable(crc64.ECMA))
	fname := fmt.Sprintf("index-name-%d-%016x", CurrentVersion, h)
	return fname
}
