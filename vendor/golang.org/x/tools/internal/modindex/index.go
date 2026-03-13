// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"bufio"
	"crypto/sha256"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

/*
The on-disk index ("payload") is a text file.
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

// Index is returned by [Read].
type Index struct {
	Version    int
	GOMODCACHE string    // absolute path of Go module cache dir
	ValidAt    time.Time // moment at which the index was up to date
	Entries    []Entry
}

func (ix *Index) String() string {
	return fmt.Sprintf("Index(%s v%d has %d entries at %v)",
		ix.GOMODCACHE, ix.Version, len(ix.Entries), ix.ValidAt)
}

// An Entry contains information for an import path.
type Entry struct {
	Dir        string // package directory relative to GOMODCACHE; uses OS path separator
	ImportPath string
	PkgName    string
	Version    string
	Names      []string // exported names and information
}

// IndexDir is where the module index is stored.
// Each logical index entry consists of a pair of files:
//
//   - the "payload" (index-VERSION-XXX), whose name is
//     randomized, holds the actual index; and
//   - the "link" (index-name-VERSION-HASH),
//     whose name is predictable, contains the
//     name of the payload file.
//
// Since the link file is small (<512B),
// reads and writes to it may be assumed atomic.
var IndexDir string = func() string {
	var dir string
	if testing.Testing() {
		dir = os.TempDir()
	} else {
		var err error
		dir, err = os.UserCacheDir()
		// shouldn't happen, but TempDir is better than
		// creating ./goimports
		if err != nil {
			dir = os.TempDir()
		}
	}
	dir = filepath.Join(dir, "goimports")
	if err := os.MkdirAll(dir, 0777); err != nil {
		dir = "" // #75505, people complain about the error message
	}
	return dir
}()

// Read reads the latest version of the on-disk index
// for the specified Go module cache directory.
// If there is no index, it returns a nil Index and an fs.ErrNotExist error.
func Read(gomodcache string) (*Index, error) {
	gomodcache, err := filepath.Abs(gomodcache)
	if err != nil {
		return nil, err
	}
	if IndexDir == "" {
		return nil, os.ErrNotExist
	}

	// Read the "link" file for the specified gomodcache directory.
	// It names the payload file.
	content, err := os.ReadFile(filepath.Join(IndexDir, linkFileBasename(gomodcache)))
	if err != nil {
		return nil, err
	}
	payloadFile := filepath.Join(IndexDir, string(content))

	// Read the index out of the payload file.
	f, err := os.Open(payloadFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readIndexFrom(gomodcache, bufio.NewReader(f))
}

func readIndexFrom(gomodcache string, r io.Reader) (*Index, error) {
	scan := bufio.NewScanner(r)

	// version
	if !scan.Scan() {
		return nil, fmt.Errorf("unexpected scan error: %v", scan.Err())
	}
	version, err := strconv.Atoi(scan.Text())
	if err != nil {
		return nil, err
	}
	if version != CurrentVersion {
		return nil, fmt.Errorf("got version %d, expected %d", version, CurrentVersion)
	}

	// gomodcache
	if !scan.Scan() {
		return nil, fmt.Errorf("scanner error reading module cache dir: %v", scan.Err())
	}
	// TODO(pjw): need to check that this is the expected cache dir
	// so the tag should be passed in to this function
	if dir := string(scan.Text()); dir != gomodcache {
		return nil, fmt.Errorf("index file GOMODCACHE mismatch: got %q, want %q", dir, gomodcache)
	}

	// changed
	if !scan.Scan() {
		return nil, fmt.Errorf("scanner error reading index creation time: %v", scan.Err())
	}
	changed, err := time.ParseInLocation(time.DateTime, scan.Text(), time.Local)
	if err != nil {
		return nil, err
	}

	// entries
	var (
		curEntry *Entry
		entries  []Entry
	)
	for scan.Scan() {
		v := scan.Text()
		if v[0] == ':' {
			if curEntry != nil {
				entries = append(entries, *curEntry)
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
			curEntry = &Entry{
				PkgName:    flds[0],
				ImportPath: flds[1],
				Dir:        relative(gomodcache, flds[2]),
				Version:    flds[3],
			}
			continue
		}
		curEntry.Names = append(curEntry.Names, v)
	}
	if err := scan.Err(); err != nil {
		return nil, fmt.Errorf("scanner failed while reading modindex entry: %v", err)
	}
	if curEntry != nil {
		entries = append(entries, *curEntry)
	}

	return &Index{
		Version:    version,
		GOMODCACHE: gomodcache,
		ValidAt:    changed,
		Entries:    entries,
	}, nil
}

// write writes the index file and updates the index directory to refer to it.
func write(gomodcache string, ix *Index) error {
	if IndexDir == "" {
		return os.ErrNotExist
	}
	// Write the index into a payload file with a fresh name.
	f, err := os.CreateTemp(IndexDir, fmt.Sprintf("index-%d-*", CurrentVersion))
	if err != nil {
		return err // e.g. disk full, or index dir deleted
	}
	if err := writeIndexToFile(ix, bufio.NewWriter(f)); err != nil {
		_ = f.Close() // ignore error
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	// Write the name of the payload file into a link file.
	indexDirFile := filepath.Join(IndexDir, linkFileBasename(gomodcache))
	content := []byte(filepath.Base(f.Name()))
	return os.WriteFile(indexDirFile, content, 0666)
}

func writeIndexToFile(x *Index, w *bufio.Writer) error {
	fmt.Fprintf(w, "%d\n", x.Version)
	fmt.Fprintf(w, "%s\n", x.GOMODCACHE)
	tm := x.ValidAt.Truncate(time.Second) // round the time down
	fmt.Fprintf(w, "%s\n", tm.Format(time.DateTime))
	for _, e := range x.Entries {
		if e.ImportPath == "" {
			continue // shouldn't happen
		}
		// PJW: maybe always write these headers as csv?
		if strings.ContainsAny(string(e.Dir), ",\"") {
			cw := csv.NewWriter(w)
			cw.Write([]string{":" + e.PkgName, e.ImportPath, string(e.Dir), e.Version})
			cw.Flush()
		} else {
			fmt.Fprintf(w, ":%s,%s,%s,%s\n", e.PkgName, e.ImportPath, e.Dir, e.Version)
		}
		for _, x := range e.Names {
			fmt.Fprintf(w, "%s\n", x)
		}
	}
	return w.Flush()
}

// linkFileBasename returns the base name of the link file in the
// index directory that holds the name of the payload file for the
// specified (absolute) Go module cache dir.
func linkFileBasename(gomodcache string) string {
	// Note: coupled to logic in ./gomodindex/cmd.go. TODO: factor.
	h := sha256.Sum256([]byte(gomodcache)) // collision-resistant hash
	return fmt.Sprintf("index-name-%d-%032x", CurrentVersion, h)
}

func relative(base, file string) string {
	if rel, err := filepath.Rel(base, file); err == nil {
		return rel
	}
	return file
}
