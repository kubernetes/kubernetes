// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gopathwalk is like filepath.Walk but specialized for finding Go
// packages, particularly in $GOPATH and $GOROOT.
package gopathwalk

import (
	"bufio"
	"bytes"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Options controls the behavior of a Walk call.
type Options struct {
	// If Logf is non-nil, debug logging is enabled through this function.
	Logf func(format string, args ...interface{})
	// Search module caches. Also disables legacy goimports ignore rules.
	ModulesEnabled bool
}

// RootType indicates the type of a Root.
type RootType int

const (
	RootUnknown RootType = iota
	RootGOROOT
	RootGOPATH
	RootCurrentModule
	RootModuleCache
	RootOther
)

// A Root is a starting point for a Walk.
type Root struct {
	Path string
	Type RootType
}

// Walk walks Go source directories ($GOROOT, $GOPATH, etc) to find packages.
// For each package found, add will be called with the absolute
// paths of the containing source directory and the package directory.
func Walk(roots []Root, add func(root Root, dir string), opts Options) {
	WalkSkip(roots, add, func(Root, string) bool { return false }, opts)
}

// WalkSkip walks Go source directories ($GOROOT, $GOPATH, etc) to find packages.
// For each package found, add will be called with the absolute
// paths of the containing source directory and the package directory.
// For each directory that will be scanned, skip will be called
// with the absolute paths of the containing source directory and the directory.
// If skip returns false on a directory it will be processed.
func WalkSkip(roots []Root, add func(root Root, dir string), skip func(root Root, dir string) bool, opts Options) {
	for _, root := range roots {
		walkDir(root, add, skip, opts)
	}
}

// walkDir creates a walker and starts fastwalk with this walker.
func walkDir(root Root, add func(Root, string), skip func(root Root, dir string) bool, opts Options) {
	if _, err := os.Stat(root.Path); os.IsNotExist(err) {
		if opts.Logf != nil {
			opts.Logf("skipping nonexistent directory: %v", root.Path)
		}
		return
	}
	start := time.Now()
	if opts.Logf != nil {
		opts.Logf("scanning %s", root.Path)
	}

	w := &walker{
		root:  root,
		add:   add,
		skip:  skip,
		opts:  opts,
		added: make(map[string]bool),
	}
	w.init()

	// Add a trailing path separator to cause filepath.WalkDir to traverse symlinks.
	path := root.Path
	if len(path) == 0 {
		path = "." + string(filepath.Separator)
	} else if !os.IsPathSeparator(path[len(path)-1]) {
		path = path + string(filepath.Separator)
	}

	if err := filepath.WalkDir(path, w.walk); err != nil {
		logf := opts.Logf
		if logf == nil {
			logf = log.Printf
		}
		logf("scanning directory %v: %v", root.Path, err)
	}

	if opts.Logf != nil {
		opts.Logf("scanned %s in %v", root.Path, time.Since(start))
	}
}

// walker is the callback for fastwalk.Walk.
type walker struct {
	root Root                    // The source directory to scan.
	add  func(Root, string)      // The callback that will be invoked for every possible Go package dir.
	skip func(Root, string) bool // The callback that will be invoked for every dir. dir is skipped if it returns true.
	opts Options                 // Options passed to Walk by the user.

	pathSymlinks []os.FileInfo
	ignoredDirs  []string

	added map[string]bool
}

// init initializes the walker based on its Options
func (w *walker) init() {
	var ignoredPaths []string
	if w.root.Type == RootModuleCache {
		ignoredPaths = []string{"cache"}
	}
	if !w.opts.ModulesEnabled && w.root.Type == RootGOPATH {
		ignoredPaths = w.getIgnoredDirs(w.root.Path)
		ignoredPaths = append(ignoredPaths, "v", "mod")
	}

	for _, p := range ignoredPaths {
		full := filepath.Join(w.root.Path, p)
		w.ignoredDirs = append(w.ignoredDirs, full)
		if w.opts.Logf != nil {
			w.opts.Logf("Directory added to ignore list: %s", full)
		}
	}
}

// getIgnoredDirs reads an optional config file at <path>/.goimportsignore
// of relative directories to ignore when scanning for go files.
// The provided path is one of the $GOPATH entries with "src" appended.
func (w *walker) getIgnoredDirs(path string) []string {
	file := filepath.Join(path, ".goimportsignore")
	slurp, err := os.ReadFile(file)
	if w.opts.Logf != nil {
		if err != nil {
			w.opts.Logf("%v", err)
		} else {
			w.opts.Logf("Read %s", file)
		}
	}
	if err != nil {
		return nil
	}

	var ignoredDirs []string
	bs := bufio.NewScanner(bytes.NewReader(slurp))
	for bs.Scan() {
		line := strings.TrimSpace(bs.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		ignoredDirs = append(ignoredDirs, line)
	}
	return ignoredDirs
}

// shouldSkipDir reports whether the file should be skipped or not.
func (w *walker) shouldSkipDir(dir string) bool {
	for _, ignoredDir := range w.ignoredDirs {
		if dir == ignoredDir {
			return true
		}
	}
	if w.skip != nil {
		// Check with the user specified callback.
		return w.skip(w.root, dir)
	}
	return false
}

// walk walks through the given path.
//
// Errors are logged if w.opts.Logf is non-nil, but otherwise ignored:
// walk returns only nil or fs.SkipDir.
func (w *walker) walk(path string, d fs.DirEntry, err error) error {
	if err != nil {
		// We have no way to report errors back through Walk or WalkSkip,
		// so just log and ignore them.
		if w.opts.Logf != nil {
			w.opts.Logf("%v", err)
		}
		if d == nil {
			// Nothing more to do: the error prevents us from knowing
			// what path even represents.
			return nil
		}
	}

	if d.Type().IsRegular() {
		if !strings.HasSuffix(path, ".go") {
			return nil
		}

		dir := filepath.Dir(path)
		if dir == w.root.Path && (w.root.Type == RootGOROOT || w.root.Type == RootGOPATH) {
			// Doesn't make sense to have regular files
			// directly in your $GOPATH/src or $GOROOT/src.
			return nil
		}

		if !w.added[dir] {
			w.add(w.root, dir)
			w.added[dir] = true
		}
		return nil
	}

	if d.IsDir() {
		base := filepath.Base(path)
		if base == "" || base[0] == '.' || base[0] == '_' ||
			base == "testdata" ||
			(w.root.Type == RootGOROOT && w.opts.ModulesEnabled && base == "vendor") ||
			(!w.opts.ModulesEnabled && base == "node_modules") {
			return fs.SkipDir
		}
		if w.shouldSkipDir(path) {
			return fs.SkipDir
		}
		return nil
	}

	if d.Type()&os.ModeSymlink != 0 {
		// TODO(bcmills): 'go list all' itself ignores symlinks within GOROOT/src
		// and GOPATH/src. Do we really need to traverse them here? If so, why?

		fi, err := os.Stat(path)
		if err != nil || !fi.IsDir() {
			// Not a directory. Just walk the file (or broken link) and be done.
			return w.walk(path, fs.FileInfoToDirEntry(fi), err)
		}

		// Avoid walking symlink cycles: if we have already followed a symlink to
		// this directory as a parent of itself, don't follow it again.
		//
		// This doesn't catch the first time through a cycle, but it also minimizes
		// the number of extra stat calls we make if we *don't* encounter a cycle.
		// Since we don't actually expect to encounter symlink cycles in practice,
		// this seems like the right tradeoff.
		for _, parent := range w.pathSymlinks {
			if os.SameFile(fi, parent) {
				return nil
			}
		}

		w.pathSymlinks = append(w.pathSymlinks, fi)
		defer func() {
			w.pathSymlinks = w.pathSymlinks[:len(w.pathSymlinks)-1]
		}()

		// On some platforms the OS (or the Go os package) sometimes fails to
		// resolve directory symlinks before a trailing slash
		// (even though POSIX requires it to do so).
		//
		// On macOS that failure may be caused by a known libc/kernel bug;
		// see https://go.dev/issue/59586.
		//
		// On Windows before Go 1.21, it may be caused by a bug in
		// os.Lstat (fixed in https://go.dev/cl/463177).
		//
		// Since we need to handle this explicitly on broken platforms anyway,
		// it is simplest to just always do that and not rely on POSIX pathname
		// resolution to walk the directory (such as by calling WalkDir with
		// a trailing slash appended to the path).
		//
		// Instead, we make a sequence of walk calls — directly and through
		// recursive calls to filepath.WalkDir — simulating what WalkDir would do
		// if the symlink were a regular directory.

		// First we call walk on the path as a directory
		// (instead of a symlink).
		err = w.walk(path, fs.FileInfoToDirEntry(fi), nil)
		if err == fs.SkipDir {
			return nil
		} else if err != nil {
			// This should be impossible, but handle it anyway in case
			// walk is changed to return other errors.
			return err
		}

		// Now read the directory and walk its entries.
		ents, err := os.ReadDir(path)
		if err != nil {
			// Report the ReadDir error, as filepath.WalkDir would do.
			err = w.walk(path, fs.FileInfoToDirEntry(fi), err)
			if err == fs.SkipDir {
				return nil
			} else if err != nil {
				return err // Again, should be impossible.
			}
			// Fall through and iterate over whatever entries we did manage to get.
		}

		for _, d := range ents {
			nextPath := filepath.Join(path, d.Name())
			if d.IsDir() {
				// We want to walk the whole directory tree rooted at nextPath,
				// not just the single entry for the directory.
				err := filepath.WalkDir(nextPath, w.walk)
				if err != nil && w.opts.Logf != nil {
					w.opts.Logf("%v", err)
				}
			} else {
				err := w.walk(nextPath, d, nil)
				if err == fs.SkipDir {
					// Skip the rest of the entries in the parent directory of nextPath
					// (that is, path itself).
					break
				} else if err != nil {
					return err // Again, should be impossible.
				}
			}
		}
		return nil
	}

	// Not a file, regular directory, or symlink; skip.
	return nil
}
