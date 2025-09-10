// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package modindex contains code for building and searching an
// [Index] of the Go module cache.
package modindex

// The directory containing the index, returned by
// [IndexDir], contains a file index-name-<ver> that contains the name
// of the current index. We believe writing that short file is atomic.
// [Read] reads that file to get the file name of the index.
// WriteIndex writes an index with a unique name and then
// writes that name into a new version of index-name-<ver>.
// (<ver> stands for the CurrentVersion of the index format.)

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"golang.org/x/mod/semver"
)

// Update updates the index for the specified Go
// module cache directory, creating it as needed.
// On success it returns the current index.
func Update(gomodcache string) (*Index, error) {
	prev, err := Read(gomodcache)
	if err != nil {
		if !os.IsNotExist(err) {
			return nil, err
		}
		prev = nil
	}
	return update(gomodcache, prev)
}

// update builds, writes, and returns the current index.
//
// If old is nil, the new index is built from all of GOMODCACHE;
// otherwise it is built from the old index plus cache updates
// since the previous index's time.
func update(gomodcache string, old *Index) (*Index, error) {
	gomodcache, err := filepath.Abs(gomodcache)
	if err != nil {
		return nil, err
	}
	new, changed, err := build(gomodcache, old)
	if err != nil {
		return nil, err
	}
	if old == nil || changed {
		if err := write(gomodcache, new); err != nil {
			return nil, err
		}
	}
	return new, nil
}

// build returns a new index for the specified Go module cache (an
// absolute path).
//
// If an old index is provided, only directories more recent than it
// that it are scanned; older directories are provided by the old
// Index.
//
// The boolean result indicates whether new entries were found.
func build(gomodcache string, old *Index) (*Index, bool, error) {
	// Set the time window.
	var start time.Time // = dawn of time
	if old != nil {
		start = old.ValidAt
	}
	now := time.Now()
	end := now.Add(24 * time.Hour) // safely in the future

	// Enumerate GOMODCACHE package directories.
	// Choose the best (latest) package for each import path.
	pkgDirs := findDirs(gomodcache, start, end)
	dirByPath, err := bestDirByImportPath(pkgDirs)
	if err != nil {
		return nil, false, err
	}

	// For each import path it might occur only in
	// dirByPath, only in old, or in both.
	// If both, use the semantically later one.
	var entries []Entry
	if old != nil {
		for _, entry := range old.Entries {
			dir, ok := dirByPath[entry.ImportPath]
			if !ok || semver.Compare(dir.version, entry.Version) <= 0 {
				// New dir is missing or not more recent; use old entry.
				entries = append(entries, entry)
				delete(dirByPath, entry.ImportPath)
			}
		}
	}

	// Extract symbol information for all the new directories.
	newEntries := extractSymbols(gomodcache, maps.Values(dirByPath))
	entries = append(entries, newEntries...)
	slices.SortFunc(entries, func(x, y Entry) int {
		if n := strings.Compare(x.PkgName, y.PkgName); n != 0 {
			return n
		}
		return strings.Compare(x.ImportPath, y.ImportPath)
	})

	return &Index{
		GOMODCACHE: gomodcache,
		ValidAt:    now, // time before the directories were scanned
		Entries:    entries,
	}, len(newEntries) > 0, nil
}
