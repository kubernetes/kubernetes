// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package txtar implements a trivial text-based file archive format.
//
// The goals for the format are:
//
//   - be trivial enough to create and edit by hand.
//   - be able to store trees of text files describing go command test cases.
//   - diff nicely in git history and code reviews.
//
// Non-goals include being a completely general archive format,
// storing binary data, storing file modes, storing special files like
// symbolic links, and so on.
//
// # Txtar format
//
// A txtar archive is zero or more comment lines and then a sequence of file entries.
// Each file entry begins with a file marker line of the form "-- FILENAME --"
// and is followed by zero or more file content lines making up the file data.
// The comment or file content ends at the next file marker line.
// The file marker line must begin with the three-byte sequence "-- "
// and end with the three-byte sequence " --", but the enclosed
// file name can be surrounding by additional white space,
// all of which is stripped.
//
// If the txtar file is missing a trailing newline on the final line,
// parsers should consider a final newline to be present anyway.
//
// There are no possible syntax errors in a txtar archive.
package txtar

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"golang.org/x/tools/txtar"
)

// An Archive is a collection of files.
type Archive = txtar.Archive

// A File is a single file in an archive.
type File = txtar.File

// Format returns the serialized form of an Archive.
// It is assumed that the Archive data structure is well-formed:
// a.Comment and all a.File[i].Data contain no file marker lines,
// and all a.File[i].Name is non-empty.
func Format(a *Archive) []byte {
	return txtar.Format(a)
}

// ParseFile parses the named file as an archive.
func ParseFile(file string) (*Archive, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	return Parse(data), nil
}

// Parse parses the serialized form of an Archive.
// The returned Archive holds slices of data.
//
// TODO use golang.org/x/tools/txtar.Parse when https://github.com/golang/go/issues/59264
// is fixed.
func Parse(data []byte) *Archive {
	a := new(Archive)
	var name string
	a.Comment, name, data = findFileMarker(data)
	for name != "" {
		f := File{name, nil}
		f.Data, name, data = findFileMarker(data)
		a.Files = append(a.Files, f)
	}
	return a
}

// NeedsQuote reports whether the given data needs to
// be quoted before it's included as a txtar file.
func NeedsQuote(data []byte) bool {
	_, _, after := findFileMarker(data)
	return after != nil
}

// Quote quotes the data so that it can be safely stored in a txtar
// file. This copes with files that contain lines that look like txtar
// separators.
//
// The original data can be recovered with Unquote. It returns an error
// if the data cannot be quoted (for example because it has no final
// newline or it holds unprintable characters)
func Quote(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, nil
	}
	if data[len(data)-1] != '\n' {
		return nil, errors.New("data has no final newline")
	}
	if !utf8.Valid(data) {
		return nil, fmt.Errorf("data contains non-UTF-8 characters")
	}
	var nd []byte
	prev := byte('\n')
	for _, b := range data {
		if prev == '\n' {
			nd = append(nd, '>')
		}
		nd = append(nd, b)
		prev = b
	}
	return nd, nil
}

// Unquote unquotes data as quoted by Quote.
func Unquote(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return nil, nil
	}
	if data[0] != '>' || data[len(data)-1] != '\n' {
		return nil, errors.New("data does not appear to be quoted")
	}
	data = bytes.Replace(data, []byte("\n>"), []byte("\n"), -1)
	data = bytes.TrimPrefix(data, []byte(">"))
	return data, nil
}

var (
	newlineMarker = []byte("\n-- ")
	marker        = []byte("-- ")
	markerEnd     = []byte(" --")
)

// findFileMarker finds the next file marker in data,
// extracts the file name, and returns the data before the marker,
// the file name, and the data after the marker.
// If there is no next marker, findFileMarker returns before = fixNL(data), name = "", after = nil.
func findFileMarker(data []byte) (before []byte, name string, after []byte) {
	var i int
	for {
		if name, after = isMarker(data[i:]); name != "" {
			return data[:i], name, after
		}
		j := bytes.Index(data[i:], newlineMarker)
		if j < 0 {
			return fixNL(data), "", nil
		}
		i += j + 1 // positioned at start of new possible marker
	}
}

// isMarker checks whether data begins with a file marker line.
// If so, it returns the name from the line and the data after the line.
// Otherwise it returns name == "" with an unspecified after.
func isMarker(data []byte) (name string, after []byte) {
	if !bytes.HasPrefix(data, marker) {
		return "", nil
	}
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		data, after = data[:i], data[i+1:]
		if data[i-1] == '\r' {
			data = data[:len(data)-1]
		}
	}
	if !bytes.HasSuffix(data, markerEnd) {
		return "", nil
	}
	return strings.TrimSpace(string(data[len(marker) : len(data)-len(markerEnd)])), after
}

// If data is empty or ends in \n, fixNL returns data.
// Otherwise fixNL returns a new slice consisting of data with a final \n added.
func fixNL(data []byte) []byte {
	if len(data) == 0 || data[len(data)-1] == '\n' {
		return data
	}
	d := make([]byte, len(data)+1)
	copy(d, data)
	d[len(data)] = '\n'
	return d
}

// Write writes each File in an Archive to the given directory, returning any
// errors encountered. An error is also returned in the event a file would be
// written outside of dir.
func Write(a *Archive, dir string) error {
	for _, f := range a.Files {
		fp := filepath.Clean(filepath.FromSlash(f.Name))
		if isAbs(fp) || strings.HasPrefix(fp, ".."+string(filepath.Separator)) {
			return fmt.Errorf("%q: outside parent directory", f.Name)
		}
		fp = filepath.Join(dir, fp)

		if err := os.MkdirAll(filepath.Dir(fp), 0o777); err != nil {
			return err
		}
		// Avoid overwriting existing files by using O_EXCL.
		out, err := os.OpenFile(fp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o666)
		if err != nil {
			return err
		}

		_, err = out.Write(f.Data)
		cerr := out.Close()
		if err != nil {
			return err
		}
		if cerr != nil {
			return cerr
		}
	}
	return nil
}

func isAbs(p string) bool {
	// Note: under Windows, filepath.IsAbs(`\foo`) returns false,
	// so we need to check for that case specifically.
	return filepath.IsAbs(p) || strings.HasPrefix(p, string(filepath.Separator))
}
