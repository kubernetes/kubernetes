// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gopathwalk is like filepath.Walk but specialized for finding Go
// packages, particularly in $GOPATH and $GOROOT.
package gopathwalk

import (
	"bufio"
	"bytes"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
)

// Options controls the behavior of a Walk call.
type Options struct {
	// If Logf is non-nil, debug logging is enabled through this function.
	Logf func(format string, args ...any)

	// Search module caches. Also disables legacy goimports ignore rules.
	ModulesEnabled bool

	// Maximum number of concurrent calls to user-provided callbacks,
	// or 0 for GOMAXPROCS.
	Concurrency int
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

// Walk concurrently walks Go source directories ($GOROOT, $GOPATH, etc) to find packages.
//
// For each package found, add will be called with the absolute
// paths of the containing source directory and the package directory.
//
// Unlike filepath.WalkDir, Walk follows symbolic links
// (while guarding against cycles).
func Walk(roots []Root, add func(root Root, dir string), opts Options) {
	WalkSkip(roots, add, func(Root, string) bool { return false }, opts)
}

// WalkSkip concurrently walks Go source directories ($GOROOT, $GOPATH, etc) to
// find packages.
//
// For each package found, add will be called with the absolute
// paths of the containing source directory and the package directory.
// For each directory that will be scanned, skip will be called
// with the absolute paths of the containing source directory and the directory.
// If skip returns false on a directory it will be processed.
//
// Unlike filepath.WalkDir, WalkSkip follows symbolic links
// (while guarding against cycles).
func WalkSkip(roots []Root, add func(root Root, dir string), skip func(root Root, dir string) bool, opts Options) {
	for _, root := range roots {
		walkDir(root, add, skip, opts)
	}
}

// walkDir creates a walker and starts fastwalk with this walker.
func walkDir(root Root, add func(Root, string), skip func(root Root, dir string) bool, opts Options) {
	if opts.Logf == nil {
		opts.Logf = func(format string, args ...any) {}
	}
	if _, err := os.Stat(root.Path); os.IsNotExist(err) {
		opts.Logf("skipping nonexistent directory: %v", root.Path)
		return
	}
	start := time.Now()
	opts.Logf("scanning %s", root.Path)

	concurrency := opts.Concurrency
	if concurrency == 0 {
		// The walk be either CPU-bound or I/O-bound, depending on what the
		// caller-supplied add function does and the details of the user's platform
		// and machine. Rather than trying to fine-tune the concurrency level for a
		// specific environment, we default to GOMAXPROCS: it is likely to be a good
		// choice for a CPU-bound add function, and if it is instead I/O-bound, then
		// dealing with I/O saturation is arguably the job of the kernel and/or
		// runtime. (Oversaturating I/O seems unlikely to harm performance as badly
		// as failing to saturate would.)
		concurrency = runtime.GOMAXPROCS(0)
	}
	w := &walker{
		root: root,
		add:  add,
		skip: skip,
		opts: opts,
		sem:  make(chan struct{}, concurrency),
	}
	w.init()

	w.sem <- struct{}{}
	path := root.Path
	if path == "" {
		path = "."
	}
	if fi, err := os.Lstat(path); err == nil {
		w.walk(path, nil, fs.FileInfoToDirEntry(fi))
	} else {
		w.opts.Logf("scanning directory %v: %v", root.Path, err)
	}
	<-w.sem
	w.walking.Wait()

	opts.Logf("scanned %s in %v", root.Path, time.Since(start))
}

// walker is the callback for fastwalk.Walk.
type walker struct {
	root Root                    // The source directory to scan.
	add  func(Root, string)      // The callback that will be invoked for every possible Go package dir.
	skip func(Root, string) bool // The callback that will be invoked for every dir. dir is skipped if it returns true.
	opts Options                 // Options passed to Walk by the user.

	walking     sync.WaitGroup
	sem         chan struct{} // Channel of semaphore tokens; send to acquire, receive to release.
	ignoredDirs []string

	added sync.Map // map[string]bool
}

// A symlinkList is a linked list of os.FileInfos for parent directories
// reached via symlinks.
type symlinkList struct {
	info os.FileInfo
	prev *symlinkList
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
		w.opts.Logf("Directory added to ignore list: %s", full)
	}
}

// getIgnoredDirs reads an optional config file at <path>/.goimportsignore
// of relative directories to ignore when scanning for go files.
// The provided path is one of the $GOPATH entries with "src" appended.
func (w *walker) getIgnoredDirs(path string) []string {
	file := filepath.Join(path, ".goimportsignore")
	slurp, err := os.ReadFile(file)
	if err != nil {
		w.opts.Logf("%v", err)
	} else {
		w.opts.Logf("Read %s", file)
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
	if slices.Contains(w.ignoredDirs, dir) {
		return true
	}
	if w.skip != nil {
		// Check with the user specified callback.
		return w.skip(w.root, dir)
	}
	return false
}

// walk walks through the given path.
//
// Errors are logged if w.opts.Logf is non-nil, but otherwise ignored.
func (w *walker) walk(path string, pathSymlinks *symlinkList, d fs.DirEntry) {
	if d.Type()&os.ModeSymlink != 0 {
		// Walk the symlink's target rather than the symlink itself.
		//
		// (Note that os.Stat, unlike the lower-lever os.Readlink,
		// follows arbitrarily many layers of symlinks, so it will eventually
		// reach either a non-symlink or a nonexistent target.)
		//
		// TODO(bcmills): 'go list all' itself ignores symlinks within GOROOT/src
		// and GOPATH/src. Do we really need to traverse them here? If so, why?

		fi, err := os.Stat(path)
		if err != nil {
			w.opts.Logf("%v", err)
			return
		}

		// Avoid walking symlink cycles: if we have already followed a symlink to
		// this directory as a parent of itself, don't follow it again.
		//
		// This doesn't catch the first time through a cycle, but it also minimizes
		// the number of extra stat calls we make if we *don't* encounter a cycle.
		// Since we don't actually expect to encounter symlink cycles in practice,
		// this seems like the right tradeoff.
		for parent := pathSymlinks; parent != nil; parent = parent.prev {
			if os.SameFile(fi, parent.info) {
				return
			}
		}

		pathSymlinks = &symlinkList{
			info: fi,
			prev: pathSymlinks,
		}
		d = fs.FileInfoToDirEntry(fi)
	}

	if d.Type().IsRegular() {
		if !strings.HasSuffix(path, ".go") {
			return
		}

		dir := filepath.Dir(path)
		if dir == w.root.Path && (w.root.Type == RootGOROOT || w.root.Type == RootGOPATH) {
			// Doesn't make sense to have regular files
			// directly in your $GOPATH/src or $GOROOT/src.
			//
			// TODO(bcmills): there are many levels of directory within
			// RootModuleCache where this also wouldn't make sense,
			// Can we generalize this to any directory without a corresponding
			// import path?
			return
		}

		if _, dup := w.added.LoadOrStore(dir, true); !dup {
			w.add(w.root, dir)
		}
	}

	if !d.IsDir() {
		return
	}

	base := filepath.Base(path)
	if base == "" || base[0] == '.' || base[0] == '_' ||
		base == "testdata" ||
		(w.root.Type == RootGOROOT && w.opts.ModulesEnabled && base == "vendor") ||
		(!w.opts.ModulesEnabled && base == "node_modules") ||
		w.shouldSkipDir(path) {
		return
	}

	// Read the directory and walk its entries.

	f, err := os.Open(path)
	if err != nil {
		w.opts.Logf("%v", err)
		return
	}
	defer f.Close()

	for {
		// We impose an arbitrary limit on the number of ReadDir results per
		// directory to limit the amount of memory consumed for stale or upcoming
		// directory entries. The limit trades off CPU (number of syscalls to read
		// the whole directory) against RAM (reachable directory entries other than
		// the one currently being processed).
		//
		// Since we process the directories recursively, we will end up maintaining
		// a slice of entries for each level of the directory tree.
		// (Compare https://go.dev/issue/36197.)
		ents, err := f.ReadDir(1024)
		if err != nil {
			if err != io.EOF {
				w.opts.Logf("%v", err)
			}
			break
		}

		for _, d := range ents {
			nextPath := filepath.Join(path, d.Name())
			if d.IsDir() {
				select {
				case w.sem <- struct{}{}:
					// Got a new semaphore token, so we can traverse the directory concurrently.
					d := d
					w.walking.Add(1)
					go func() {
						defer func() {
							<-w.sem
							w.walking.Done()
						}()
						w.walk(nextPath, pathSymlinks, d)
					}()
					continue

				default:
					// No tokens available, so traverse serially.
				}
			}

			w.walk(nextPath, pathSymlinks, d)
		}
	}
}
