// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

package gopathrs

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/fd"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/procfs"
)

func openat2(dir fd.Fd, path string, how *unix.OpenHow) (*os.File, error) {
	file, err := fd.Openat2(dir, path, how)
	if err != nil {
		return nil, err
	}
	// If we are using RESOLVE_IN_ROOT, the name we generated may be wrong.
	if how.Resolve&unix.RESOLVE_IN_ROOT == unix.RESOLVE_IN_ROOT {
		if actualPath, err := procfs.ProcSelfFdReadlink(file); err == nil {
			// TODO: Ideally we would not need to dup the fd, but you cannot
			//       easily just swap an *os.File with one from the same fd
			//       (the GC will close the old one, and you cannot clear the
			//       finaliser easily because it is associated with an internal
			//       field of *os.File not *os.File itself).
			newFile, err := fd.DupWithName(file, actualPath)
			if err != nil {
				return nil, err
			}
			_ = file.Close()
			file = newFile
		}
	}
	return file, nil
}

func lookupOpenat2(root fd.Fd, unsafePath string, partial bool) (*os.File, string, error) {
	if !partial {
		file, err := openat2(root, unsafePath, &unix.OpenHow{
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
func partialLookupOpenat2(root fd.Fd, unsafePath string) (*os.File, string, error) {
	// TODO: Implement this as a git-bisect-like binary search.

	unsafePath = filepath.ToSlash(unsafePath) // noop
	endIdx := len(unsafePath)
	var lastError error
	for endIdx > 0 {
		subpath := unsafePath[:endIdx]

		handle, err := openat2(root, subpath, &unix.OpenHow{
			Flags:   unix.O_PATH | unix.O_CLOEXEC,
			Resolve: unix.RESOLVE_IN_ROOT | unix.RESOLVE_NO_MAGICLINKS,
		})
		if err == nil {
			// Jump over the slash if we have a non-"" remainingPath.
			if endIdx < len(unsafePath) {
				endIdx++
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
	rootClone, err := fd.Dup(root)
	if err != nil {
		return nil, "", err
	}
	return rootClone, unsafePath, lastError
}
