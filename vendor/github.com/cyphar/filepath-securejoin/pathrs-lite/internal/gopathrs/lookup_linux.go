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
	"path"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/cyphar/filepath-securejoin/internal/consts"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/fd"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/gocompat"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/linux"
	"github.com/cyphar/filepath-securejoin/pathrs-lite/internal/procfs"
)

type symlinkStackEntry struct {
	// (dir, remainingPath) is what we would've returned if the link didn't
	// exist. This matches what openat2(RESOLVE_IN_ROOT) would return in
	// this case.
	dir           *os.File
	remainingPath string
	// linkUnwalked is the remaining path components from the original
	// Readlink which we have yet to walk. When this slice is empty, we
	// drop the link from the stack.
	linkUnwalked []string
}

func (se symlinkStackEntry) String() string {
	return fmt.Sprintf("<%s>/%s [->%s]", se.dir.Name(), se.remainingPath, strings.Join(se.linkUnwalked, "/"))
}

func (se symlinkStackEntry) Close() {
	_ = se.dir.Close()
}

type symlinkStack []*symlinkStackEntry

func (s *symlinkStack) IsEmpty() bool {
	return s == nil || len(*s) == 0
}

func (s *symlinkStack) Close() {
	if s != nil {
		for _, link := range *s {
			link.Close()
		}
		// TODO: Switch to clear once we switch to Go 1.21.
		*s = nil
	}
}

var (
	errEmptyStack         = errors.New("[internal] stack is empty")
	errBrokenSymlinkStack = errors.New("[internal error] broken symlink stack")
)

func (s *symlinkStack) popPart(part string) error {
	if s == nil || s.IsEmpty() {
		// If there is nothing in the symlink stack, then the part was from the
		// real path provided by the user, and this is a no-op.
		return errEmptyStack
	}
	if part == "." {
		// "." components are no-ops -- we drop them when doing SwapLink.
		return nil
	}

	tailEntry := (*s)[len(*s)-1]

	// Double-check that we are popping the component we expect.
	if len(tailEntry.linkUnwalked) == 0 {
		return fmt.Errorf("%w: trying to pop component %q of empty stack entry %s", errBrokenSymlinkStack, part, tailEntry)
	}
	headPart := tailEntry.linkUnwalked[0]
	if headPart != part {
		return fmt.Errorf("%w: trying to pop component %q but the last stack entry is %s (%q)", errBrokenSymlinkStack, part, tailEntry, headPart)
	}

	// Drop the component, but keep the entry around in case we are dealing
	// with a "tail-chained" symlink.
	tailEntry.linkUnwalked = tailEntry.linkUnwalked[1:]
	return nil
}

func (s *symlinkStack) PopPart(part string) error {
	if err := s.popPart(part); err != nil {
		if errors.Is(err, errEmptyStack) {
			// Skip empty stacks.
			err = nil
		}
		return err
	}

	// Clean up any of the trailing stack entries that are empty.
	for lastGood := len(*s) - 1; lastGood >= 0; lastGood-- {
		entry := (*s)[lastGood]
		if len(entry.linkUnwalked) > 0 {
			break
		}
		entry.Close()
		(*s) = (*s)[:lastGood]
	}
	return nil
}

func (s *symlinkStack) push(dir *os.File, remainingPath, linkTarget string) error {
	if s == nil {
		return nil
	}
	// Split the link target and clean up any "" parts.
	linkTargetParts := gocompat.SlicesDeleteFunc(
		strings.Split(linkTarget, "/"),
		func(part string) bool { return part == "" || part == "." })

	// Copy the directory so the caller doesn't close our copy.
	dirCopy, err := fd.Dup(dir)
	if err != nil {
		return err
	}

	// Add to the stack.
	*s = append(*s, &symlinkStackEntry{
		dir:           dirCopy,
		remainingPath: remainingPath,
		linkUnwalked:  linkTargetParts,
	})
	return nil
}

func (s *symlinkStack) SwapLink(linkPart string, dir *os.File, remainingPath, linkTarget string) error {
	// If we are currently inside a symlink resolution, remove the symlink
	// component from the last symlink entry, but don't remove the entry even
	// if it's empty. If we are a "tail-chained" symlink (a trailing symlink we
	// hit during a symlink resolution) we need to keep the old symlink until
	// we finish the resolution.
	if err := s.popPart(linkPart); err != nil {
		if !errors.Is(err, errEmptyStack) {
			return err
		}
		// Push the component regardless of whether the stack was empty.
	}
	return s.push(dir, remainingPath, linkTarget)
}

func (s *symlinkStack) PopTopSymlink() (*os.File, string, bool) {
	if s == nil || s.IsEmpty() {
		return nil, "", false
	}
	tailEntry := (*s)[0]
	*s = (*s)[1:]
	return tailEntry.dir, tailEntry.remainingPath, true
}

// PartialLookupInRoot tries to lookup as much of the request path as possible
// within the provided root (a-la RESOLVE_IN_ROOT) and opens the final existing
// component of the requested path, returning a file handle to the final
// existing component and a string containing the remaining path components.
func PartialLookupInRoot(root fd.Fd, unsafePath string) (*os.File, string, error) {
	return lookupInRoot(root, unsafePath, true)
}

func completeLookupInRoot(root fd.Fd, unsafePath string) (*os.File, error) {
	handle, remainingPath, err := lookupInRoot(root, unsafePath, false)
	if remainingPath != "" && err == nil {
		// should never happen
		err = fmt.Errorf("[bug] non-empty remaining path when doing a non-partial lookup: %q", remainingPath)
	}
	// lookupInRoot(partial=false) will always close the handle if an error is
	// returned, so no need to double-check here.
	return handle, err
}

func lookupInRoot(root fd.Fd, unsafePath string, partial bool) (Handle *os.File, _ string, _ error) {
	unsafePath = filepath.ToSlash(unsafePath) // noop

	// This is very similar to SecureJoin, except that we operate on the
	// components using file descriptors. We then return the last component we
	// managed open, along with the remaining path components not opened.

	// Try to use openat2 if possible.
	//
	// NOTE: If openat2(2) works normally but fails for this lookup, it is
	// probably not a good idea to fall-back to the O_PATH resolver. An
	// attacker could find a bug in the O_PATH resolver and uncontionally
	// falling back to the O_PATH resolver would form a downgrade attack.
	if handle, remainingPath, err := lookupOpenat2(root, unsafePath, partial); err == nil || linux.HasOpenat2() {
		return handle, remainingPath, err
	}

	// Get the "actual" root path from /proc/self/fd. This is necessary if the
	// root is some magic-link like /proc/$pid/root, in which case we want to
	// make sure when we do procfs.CheckProcSelfFdPath that we are using the
	// correct root path.
	logicalRootPath, err := procfs.ProcSelfFdReadlink(root)
	if err != nil {
		return nil, "", fmt.Errorf("get real root path: %w", err)
	}

	currentDir, err := fd.Dup(root)
	if err != nil {
		return nil, "", fmt.Errorf("clone root fd: %w", err)
	}
	defer func() {
		// If a handle is not returned, close the internal handle.
		if Handle == nil {
			_ = currentDir.Close()
		}
	}()

	// symlinkStack is used to emulate how openat2(RESOLVE_IN_ROOT) treats
	// dangling symlinks. If we hit a non-existent path while resolving a
	// symlink, we need to return the (dir, remainingPath) that we had when we
	// hit the symlink (treating the symlink as though it were a regular file).
	// The set of (dir, remainingPath) sets is stored within the symlinkStack
	// and we add and remove parts when we hit symlink and non-symlink
	// components respectively. We need a stack because of recursive symlinks
	// (symlinks that contain symlink components in their target).
	//
	// Note that the stack is ONLY used for book-keeping. All of the actual
	// path walking logic is still based on currentPath/remainingPath and
	// currentDir (as in SecureJoin).
	var symStack *symlinkStack
	if partial {
		symStack = new(symlinkStack)
		defer symStack.Close()
	}

	var (
		linksWalked   int
		currentPath   string
		remainingPath = unsafePath
	)
	for remainingPath != "" {
		// Save the current remaining path so if the part is not real we can
		// return the path including the component.
		oldRemainingPath := remainingPath

		// Get the next path component.
		var part string
		if i := strings.IndexByte(remainingPath, '/'); i == -1 {
			part, remainingPath = remainingPath, ""
		} else {
			part, remainingPath = remainingPath[:i], remainingPath[i+1:]
		}
		// If we hit an empty component, we need to treat it as though it is
		// "." so that trailing "/" and "//" components on a non-directory
		// correctly return the right error code.
		if part == "" {
			part = "."
		}

		// Apply the component lexically to the path we are building.
		// currentPath does not contain any symlinks, and we are lexically
		// dealing with a single component, so it's okay to do a filepath.Clean
		// here.
		nextPath := path.Join("/", currentPath, part)
		// If we logically hit the root, just clone the root rather than
		// opening the part and doing all of the other checks.
		if nextPath == "/" {
			if err := symStack.PopPart(part); err != nil {
				return nil, "", fmt.Errorf("walking into root with part %q failed: %w", part, err)
			}
			// Jump to root.
			rootClone, err := fd.Dup(root)
			if err != nil {
				return nil, "", fmt.Errorf("clone root fd: %w", err)
			}
			_ = currentDir.Close()
			currentDir = rootClone
			currentPath = nextPath
			continue
		}

		// Try to open the next component.
		nextDir, err := fd.Openat(currentDir, part, unix.O_PATH|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		switch err {
		case nil:
			st, err := nextDir.Stat()
			if err != nil {
				_ = nextDir.Close()
				return nil, "", fmt.Errorf("stat component %q: %w", part, err)
			}

			switch st.Mode() & os.ModeType { //nolint:exhaustive // just a glorified if statement
			case os.ModeSymlink:
				// readlinkat implies AT_EMPTY_PATH since Linux 2.6.39. See
				// Linux commit 65cfc6722361 ("readlinkat(), fchownat() and
				// fstatat() with empty relative pathnames").
				linkDest, err := fd.Readlinkat(nextDir, "")
				// We don't need the handle anymore.
				_ = nextDir.Close()
				if err != nil {
					return nil, "", err
				}

				linksWalked++
				if linksWalked > consts.MaxSymlinkLimit {
					return nil, "", &os.PathError{Op: "securejoin.lookupInRoot", Path: logicalRootPath + "/" + unsafePath, Err: unix.ELOOP}
				}

				// Swap out the symlink's component for the link entry itself.
				if err := symStack.SwapLink(part, currentDir, oldRemainingPath, linkDest); err != nil {
					return nil, "", fmt.Errorf("walking into symlink %q failed: push symlink: %w", part, err)
				}

				// Update our logical remaining path.
				remainingPath = linkDest + "/" + remainingPath
				// Absolute symlinks reset any work we've already done.
				if path.IsAbs(linkDest) {
					// Jump to root.
					rootClone, err := fd.Dup(root)
					if err != nil {
						return nil, "", fmt.Errorf("clone root fd: %w", err)
					}
					_ = currentDir.Close()
					currentDir = rootClone
					currentPath = "/"
				}

			default:
				// If we are dealing with a directory, simply walk into it.
				_ = currentDir.Close()
				currentDir = nextDir
				currentPath = nextPath

				// The part was real, so drop it from the symlink stack.
				if err := symStack.PopPart(part); err != nil {
					return nil, "", fmt.Errorf("walking into directory %q failed: %w", part, err)
				}

				// If we are operating on a .., make sure we haven't escaped.
				// We only have to check for ".." here because walking down
				// into a regular component component cannot cause you to
				// escape. This mirrors the logic in RESOLVE_IN_ROOT, except we
				// have to check every ".." rather than only checking after a
				// rename or mount on the system.
				if part == ".." {
					// Make sure the root hasn't moved.
					if err := procfs.CheckProcSelfFdPath(logicalRootPath, root); err != nil {
						return nil, "", fmt.Errorf("root path moved during lookup: %w", err)
					}
					// Make sure the path is what we expect.
					fullPath := logicalRootPath + nextPath
					if err := procfs.CheckProcSelfFdPath(fullPath, currentDir); err != nil {
						return nil, "", fmt.Errorf("walking into %q had unexpected result: %w", part, err)
					}
				}
			}

		default:
			if !partial {
				return nil, "", err
			}
			// If there are any remaining components in the symlink stack, we
			// are still within a symlink resolution and thus we hit a dangling
			// symlink. So pretend that the first symlink in the stack we hit
			// was an ENOENT (to match openat2).
			if oldDir, remainingPath, ok := symStack.PopTopSymlink(); ok {
				_ = currentDir.Close()
				return oldDir, remainingPath, err
			}
			// We have hit a final component that doesn't exist, so we have our
			// partial open result. Note that we have to use the OLD remaining
			// path, since the lookup failed.
			return currentDir, oldRemainingPath, err
		}
	}

	// If the unsafePath had a trailing slash, we need to make sure we try to
	// do a relative "." open so that we will correctly return an error when
	// the final component is a non-directory (to match openat2). In the
	// context of openat2, a trailing slash and a trailing "/." are completely
	// equivalent.
	if strings.HasSuffix(unsafePath, "/") {
		nextDir, err := fd.Openat(currentDir, ".", unix.O_PATH|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		if err != nil {
			if !partial {
				_ = currentDir.Close()
				currentDir = nil
			}
			return currentDir, "", err
		}
		_ = currentDir.Close()
		currentDir = nextDir
	}

	// All of the components existed!
	return currentDir, "", nil
}
