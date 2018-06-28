// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgtree

import (
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

// DirWalkFunc is the type of the function called for each file system node
// visited by DirWalk. The path argument contains the argument to DirWalk as a
// prefix; that is, if DirWalk is called with "dir", which is a directory
// containing the file "a", the walk function will be called with the argument
// "dir/a", using the correct os.PathSeparator for the Go Operating System
// architecture, GOOS. The info argument is the os.FileInfo for the named path.
//
// If there was a problem walking to the file or directory named by path, the
// incoming error will describe the problem and the function can decide how to
// handle that error (and DirWalk will not descend into that directory). If an
// error is returned, processing stops. The sole exception is when the function
// returns the special value filepath.SkipDir. If the function returns
// filepath.SkipDir when invoked on a directory, DirWalk skips the directory's
// contents entirely. If the function returns filepath.SkipDir when invoked on a
// non-directory file system node, DirWalk skips the remaining files in the
// containing directory.
type DirWalkFunc func(osPathname string, info os.FileInfo, err error) error

// DirWalk walks the file tree rooted at osDirname, calling for each file system
// node in the tree, including root. All errors that arise visiting nodes are
// filtered by walkFn. The nodes are walked in lexical order, which makes the
// output deterministic but means that for very large directories DirWalk can be
// inefficient. Unlike filepath.Walk, DirWalk does follow symbolic links.
func DirWalk(osDirname string, walkFn DirWalkFunc) error {
	osDirname = filepath.Clean(osDirname)

	// Ensure parameter is a directory
	fi, err := os.Stat(osDirname)
	if err != nil {
		return errors.Wrap(err, "cannot read node")
	}
	if !fi.IsDir() {
		return errors.Errorf("cannot walk non directory: %q", osDirname)
	}

	// Initialize a work queue with the empty string, which signifies the
	// starting directory itself.
	queue := []string{""}

	var osRelative string // os-specific relative pathname under directory name

	// As we enumerate over the queue and encounter a directory, its children
	// will be added to the work queue.
	for len(queue) > 0 {
		// Unshift a pathname from the queue (breadth-first traversal of
		// hierarchy)
		osRelative, queue = queue[0], queue[1:]
		osPathname := filepath.Join(osDirname, osRelative)

		// walkFn needs to choose how to handle symbolic links, therefore obtain
		// lstat rather than stat.
		fi, err = os.Lstat(osPathname)
		if err == nil {
			err = walkFn(osPathname, fi, nil)
		} else {
			err = walkFn(osPathname, nil, errors.Wrap(err, "cannot read node"))
		}

		if err != nil {
			if err == filepath.SkipDir {
				if fi.Mode()&os.ModeSymlink > 0 {
					// Resolve symbolic link referent to determine whether node
					// is directory or not.
					fi, err = os.Stat(osPathname)
					if err != nil {
						return errors.Wrap(err, "cannot visit node")
					}
				}
				// If current node is directory, then skip this
				// directory. Otherwise, skip all nodes in the same parent
				// directory.
				if !fi.IsDir() {
					// Consume nodes from queue while they have the same parent
					// as the current node.
					osParent := filepath.Dir(osPathname) + osPathSeparator
					for len(queue) > 0 && strings.HasPrefix(queue[0], osParent) {
						queue = queue[1:] // drop sibling from queue
					}
				}

				continue
			}
			return errors.Wrap(err, "DirWalkFunction") // wrap error returned by walkFn
		}

		if fi.IsDir() {
			osChildrenNames, err := sortedChildrenFromDirname(osPathname)
			if err != nil {
				return errors.Wrap(err, "cannot get list of directory children")
			}
			for _, osChildName := range osChildrenNames {
				switch osChildName {
				case ".", "..":
					// skip
				default:
					queue = append(queue, filepath.Join(osRelative, osChildName))
				}
			}
		}
	}
	return nil
}

// sortedChildrenFromDirname returns a lexicographically sorted list of child
// nodes for the specified directory.
func sortedChildrenFromDirname(osDirname string) ([]string, error) {
	fh, err := os.Open(osDirname)
	if err != nil {
		return nil, errors.Wrap(err, "cannot Open")
	}

	osChildrenNames, err := fh.Readdirnames(0) // 0: read names of all children
	if err != nil {
		return nil, errors.Wrap(err, "cannot Readdirnames")
	}
	sort.Strings(osChildrenNames)

	// Close the file handle to the open directory without masking possible
	// previous error value.
	if er := fh.Close(); err == nil {
		err = errors.Wrap(er, "cannot Close")
	}
	return osChildrenNames, err
}
