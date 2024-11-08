//go:build linux

// Copyright (C) 2024 SUSE LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package securejoin

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/sys/unix"
)

var hasOpenat2 = sync.OnceValue(func() bool {
	fd, err := unix.Openat2(unix.AT_FDCWD, ".", &unix.OpenHow{
		Flags:   unix.O_PATH | unix.O_CLOEXEC,
		Resolve: unix.RESOLVE_NO_SYMLINKS | unix.RESOLVE_IN_ROOT,
	})
	if err != nil {
		return false
	}
	_ = unix.Close(fd)
	return true
})

func scopedLookupShouldRetry(how *unix.OpenHow, err error) bool {
	// RESOLVE_IN_ROOT (and RESOLVE_BENEATH) can return -EAGAIN if we resolve
	// ".." while a mount or rename occurs anywhere on the system. This could
	// happen spuriously, or as the result of an attacker trying to mess with
	// us during lookup.
	//
	// In addition, scoped lookups have a "safety check" at the end of
	// complete_walk which will return -EXDEV if the final path is not in the
	// root.
	return how.Resolve&(unix.RESOLVE_IN_ROOT|unix.RESOLVE_BENEATH) != 0 &&
		(errors.Is(err, unix.EAGAIN) || errors.Is(err, unix.EXDEV))
}

const scopedLookupMaxRetries = 10

func openat2File(dir *os.File, path string, how *unix.OpenHow) (*os.File, error) {
	fullPath := dir.Name() + "/" + path
	// Make sure we always set O_CLOEXEC.
	how.Flags |= unix.O_CLOEXEC
	var tries int
	for tries < scopedLookupMaxRetries {
		fd, err := unix.Openat2(int(dir.Fd()), path, how)
		if err != nil {
			if scopedLookupShouldRetry(how, err) {
				// We retry a couple of times to avoid the spurious errors, and
				// if we are being attacked then returning -EAGAIN is the best
				// we can do.
				tries++
				continue
			}
			return nil, &os.PathError{Op: "openat2", Path: fullPath, Err: err}
		}
		// If we are using RESOLVE_IN_ROOT, the name we generated may be wrong.
		// NOTE: The procRoot code MUST NOT use RESOLVE_IN_ROOT, otherwise
		//       you'll get infinite recursion here.
		if how.Resolve&unix.RESOLVE_IN_ROOT == unix.RESOLVE_IN_ROOT {
			if actualPath, err := rawProcSelfFdReadlink(fd); err == nil {
				fullPath = actualPath
			}
		}
		return os.NewFile(uintptr(fd), fullPath), nil
	}
	return nil, &os.PathError{Op: "openat2", Path: fullPath, Err: errPossibleAttack}
}

func lookupOpenat2(root *os.File, unsafePath string, partial bool) (*os.File, string, error) {
	if !partial {
		file, err := openat2File(root, unsafePath, &unix.OpenHow{
			Flags:   unix.O_PATH | unix.O_CLOEXEC,
			Resolve: unix.RESOLVE_IN_ROOT | unix.RESOLVE_NO_MAGICLINKS,
		})
		return file, "", err
	}
	return partialLookupOpenat2(root, unsafePath)
}

// partialLookupOpenat2 is an alternative implementation of
// partialLookupInRoot, using openat2(RESOLVE_IN_ROOT) to more safely get a
// handle to the deepest existing child of the requested path within the root.
func partialLookupOpenat2(root *os.File, unsafePath string) (*os.File, string, error) {
	// TODO: Implement this as a git-bisect-like binary search.

	unsafePath = filepath.ToSlash(unsafePath) // noop
	endIdx := len(unsafePath)
	var lastError error
	for endIdx > 0 {
		subpath := unsafePath[:endIdx]

		handle, err := openat2File(root, subpath, &unix.OpenHow{
			Flags:   unix.O_PATH | unix.O_CLOEXEC,
			Resolve: unix.RESOLVE_IN_ROOT | unix.RESOLVE_NO_MAGICLINKS,
		})
		if err == nil {
			// Jump over the slash if we have a non-"" remainingPath.
			if endIdx < len(unsafePath) {
				endIdx += 1
			}
			// We found a subpath!
			return handle, unsafePath[endIdx:], lastError
		}
		if errors.Is(err, unix.ENOENT) || errors.Is(err, unix.ENOTDIR) {
			// That path doesn't exist, let's try the next directory up.
			endIdx = strings.LastIndexByte(subpath, '/')
			lastError = err
			continue
		}
		return nil, "", fmt.Errorf("open subpath: %w", err)
	}
	// If we couldn't open anything, the whole subpath is missing. Return a
	// copy of the root fd so that the caller doesn't close this one by
	// accident.
	rootClone, err := dupFile(root)
	if err != nil {
		return nil, "", err
	}
	return rootClone, unsafePath, lastError
}
