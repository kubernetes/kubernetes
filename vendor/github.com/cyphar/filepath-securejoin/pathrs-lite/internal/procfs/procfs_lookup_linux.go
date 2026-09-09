// SPDX-License-Identifier: MPL-2.0

//go:build linux

// Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
// Copyright (C) 2024-2025 SUSE LLC
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// This code is adapted to be a minimal version of the libpathrs proc resolver
// <https://github.com/opensuse/libpathrs/blob/v0.1.3/src/resolvers/procfs.rs>.
// As we only need O_PATH|O_NOFOLLOW support, this is not too much to port.

package procfs

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/internal/consts"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/fd"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gocompat"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/linux"
)

// procfsLookupInRoot is a stripped down version of completeLookupInRoot,
// entirely designed to support the very small set of features necessary to
// make procfs handling work. Unlike completeLookupInRoot, we always have
// O_PATH|O_NOFOLLOW behaviour for trailing symlinks.
//
// The main restrictions are:
//
//   - ".." is not supported (as it requires either os.Root-style replays,
//     which is more bug-prone; or procfs verification, which is not possible
//     due to re-entrancy issues).
//   - Absolute symlinks for the same reason (and all absolute symlinks in
//     procfs are magic-links, which we want to skip anyway).
//   - If statx is supported (checkSymlinkOvermount), any mount-point crossings
//     (which is the main attack of concern against /proc).
//   - Partial lookups are not supported, so the symlink stack is not needed.
//   - Trailing slash special handling is not necessary in most cases (if we
//     operating on procfs, it's usually with programmer-controlled strings
//     that will then be re-opened), so we skip it since whatever re-opens it
//     can deal with it. It's a creature comfort anyway.
//
// If the system supports openat2(), this is implemented using equivalent flags
// (RESOLVE_BENEATH | RESOLVE_NO_XDEV | RESOLVE_NO_MAGICLINKS).
func procfsLookupInRoot(procRoot fd.Fd, unsafePath string) (Handle *os.File, _ error) {
	unsafePath = filepath.ToSlash(unsafePath) // noop

	// Make sure that an empty unsafe path still returns something sane, even
	// with openat2 (which doesn't have AT_EMPTY_PATH semantics yet).
	if unsafePath == "" {
		unsafePath = "."
	}

	// This is already checked by getProcRoot, but make sure here since the
	// core security of this lookup is based on this assumption.
	if err := verifyProcRoot(procRoot); err != nil {
		return nil, err
	}

	if linux.HasOpenat2() {
		// We prefer being able to use RESOLVE_NO_XDEV if we can, to be
		// absolutely sure we are operating on a clean /proc handle that
		// doesn't have any cheeky overmounts that could trick us (including
		// symlink mounts on top of /proc/thread-self). RESOLVE_BENEATH isn't
		// strictly needed, but just use it since we have it.
		//
		// NOTE: /proc/self is technically a magic-link (the contents of the
		//       symlink are generated dynamically), but it doesn't use
		//       nd_jump_link() so RESOLVE_NO_MAGICLINKS allows it.
		//
		// TODO: It would be nice to have RESOLVE_NO_DOTDOT, purely for
		//       self-consistency with the backup O_PATH resolver.
		handle, err := fd.Openat2(procRoot, unsafePath, &unix.OpenHow{
			Flags:   unix.O_PATH | unix.O_NOFOLLOW | unix.O_CLOEXEC,
			Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_XDEV | unix.RESOLVE_NO_MAGICLINKS,
		})
		if err != nil {
			// TODO: Once we bump the minimum Go version to 1.20, we can use
			// multiple %w verbs for this wrapping. For now we need to use a
			// compatibility shim for older Go versions.
			// err = fmt.Errorf("%w: %w", errUnsafeProcfs, err)
			return nil, gocompat.WrapBaseError(err, errUnsafeProcfs)
		}
		return handle, nil
	}

	// To mirror openat2(RESOLVE_BENEATH), we need to return an error if the
	// path is absolute.
	if path.IsAbs(unsafePath) {
		return nil, fmt.Errorf("%w: cannot resolve absolute paths in procfs resolver", internal.ErrPossibleBreakout)
	}

	currentDir, err := fd.Dup(procRoot)
	if err != nil {
		return nil, fmt.Errorf("clone root fd: %w", err)
	}
	defer func() {
		// If a handle is not returned, close the internal handle.
		if Handle == nil {
			_ = currentDir.Close()
		}
	}()

	var (
		linksWalked   int
		currentPath   string
		remainingPath = unsafePath
	)
	for remainingPath != "" {
		// Get the next path component.
		var part string
		if i := strings.IndexByte(remainingPath, '/'); i == -1 {
			part, remainingPath = remainingPath, ""
		} else {
			part, remainingPath = remainingPath[:i], remainingPath[i+1:]
		}
		if part == "" {
			// no-op component, but treat it the same as "."
			part = "."
		}
		if part == ".." {
			// not permitted
			return nil, fmt.Errorf("%w: cannot walk into '..' in procfs resolver", internal.ErrPossibleBreakout)
		}

		// Apply the component lexically to the path we are building.
		// currentPath does not contain any symlinks, and we are lexically
		// dealing with a single component, so it's okay to do a filepath.Clean
		// here. (Not to mention that ".." isn't allowed.)
		nextPath := path.Join("/", currentPath, part)
		// If we logically hit the root, just clone the root rather than
		// opening the part and doing all of the other checks.
		if nextPath == "/" {
			// Jump to root.
			rootClone, err := fd.Dup(procRoot)
			if err != nil {
				return nil, fmt.Errorf("clone root fd: %w", err)
			}
			_ = currentDir.Close()
			currentDir = rootClone
			currentPath = nextPath
			continue
		}

		// Try to open the next component.
		nextDir, err := fd.Openat(currentDir, part, unix.O_PATH|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		if err != nil {
			return nil, err
		}

		// Make sure we are still on procfs and haven't crossed mounts.
		if err := verifyProcHandle(nextDir); err != nil {
			_ = nextDir.Close()
			return nil, fmt.Errorf("check %q component is on procfs: %w", part, err)
		}
		if err := checkSubpathOvermount(procRoot, nextDir, ""); err != nil {
			_ = nextDir.Close()
			return nil, fmt.Errorf("check %q component is not overmounted: %w", part, err)
		}

		// We are emulating O_PATH|O_NOFOLLOW, so we only need to traverse into
		// trailing symlinks if we are not the final component. Otherwise we
		// can just return the currentDir.
		if remainingPath != "" {
			st, err := nextDir.Stat()
			if err != nil {
				_ = nextDir.Close()
				return nil, fmt.Errorf("stat component %q: %w", part, err)
			}

			if st.Mode()&os.ModeType == os.ModeSymlink {
				// readlinkat implies AT_EMPTY_PATH since Linux 2.6.39. See
				// Linux commit 65cfc6722361 ("readlinkat(), fchownat() and
				// fstatat() with empty relative pathnames").
				linkDest, err := fd.Readlinkat(nextDir, "")
				// We don't need the handle anymore.
				_ = nextDir.Close()
				if err != nil {
					return nil, err
				}

				linksWalked++
				if linksWalked > consts.MaxSymlinkLimit {
					return nil, &os.PathError{Op: "securejoin.procfsLookupInRoot", Path: "/proc/" + unsafePath, Err: unix.ELOOP}
				}

				// Update our logical remaining path.
				remainingPath = linkDest + "/" + remainingPath
				// Absolute symlinks are probably magiclinks, we reject them.
				if path.IsAbs(linkDest) {
					return nil, fmt.Errorf("%w: cannot jump to / in procfs resolver -- possible magiclink", internal.ErrPossibleBreakout)
				}
				continue
			}
		}

		// Walk into the next component.
		_ = currentDir.Close()
		currentDir = nextDir
		currentPath = nextPath
	}

	// One final sanity-check.
	if err := verifyProcHandle(currentDir); err != nil {
		return nil, fmt.Errorf("check final handle is on procfs: %w", err)
	}
	if err := checkSubpathOvermount(procRoot, currentDir, ""); err != nil {
		return nil, fmt.Errorf("check final handle is not overmounted: %w", err)
	}
	return currentDir, nil
}
