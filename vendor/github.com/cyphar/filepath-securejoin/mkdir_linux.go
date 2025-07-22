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

	"golang.org/x/sys/unix"
)

var (
	errInvalidMode    = errors.New("invalid permission mode")
	errPossibleAttack = errors.New("possible attack detected")
)

// modePermExt is like os.ModePerm except that it also includes the set[ug]id
// and sticky bits.
const modePermExt = os.ModePerm | os.ModeSetuid | os.ModeSetgid | os.ModeSticky

//nolint:cyclop // this function needs to handle a lot of cases
func toUnixMode(mode os.FileMode) (uint32, error) {
	sysMode := uint32(mode.Perm())
	if mode&os.ModeSetuid != 0 {
		sysMode |= unix.S_ISUID
	}
	if mode&os.ModeSetgid != 0 {
		sysMode |= unix.S_ISGID
	}
	if mode&os.ModeSticky != 0 {
		sysMode |= unix.S_ISVTX
	}
	// We don't allow file type bits.
	if mode&os.ModeType != 0 {
		return 0, fmt.Errorf("%w %+.3o (%s): type bits not permitted", errInvalidMode, mode, mode)
	}
	// We don't allow other unknown modes.
	if mode&^modePermExt != 0 || sysMode&unix.S_IFMT != 0 {
		return 0, fmt.Errorf("%w %+.3o (%s): unknown mode bits", errInvalidMode, mode, mode)
	}
	return sysMode, nil
}

// MkdirAllHandle is equivalent to [MkdirAll], except that it is safer to use
// in two respects:
//
//   - The caller provides the root directory as an *[os.File] (preferably O_PATH)
//     handle. This means that the caller can be sure which root directory is
//     being used. Note that this can be emulated by using /proc/self/fd/... as
//     the root path with [os.MkdirAll].
//
//   - Once all of the directories have been created, an *[os.File] O_PATH handle
//     to the directory at unsafePath is returned to the caller. This is done in
//     an effectively-race-free way (an attacker would only be able to swap the
//     final directory component), which is not possible to emulate with
//     [MkdirAll].
//
// In addition, the returned handle is obtained far more efficiently than doing
// a brand new lookup of unsafePath (such as with [SecureJoin] or openat2) after
// doing [MkdirAll]. If you intend to open the directory after creating it, you
// should use MkdirAllHandle.
func MkdirAllHandle(root *os.File, unsafePath string, mode os.FileMode) (_ *os.File, Err error) {
	unixMode, err := toUnixMode(mode)
	if err != nil {
		return nil, err
	}
	// On Linux, mkdirat(2) (and os.Mkdir) silently ignore the suid and sgid
	// bits. We could also silently ignore them but since we have very few
	// users it seems more prudent to return an error so users notice that
	// these bits will not be set.
	if unixMode&^0o1777 != 0 {
		return nil, fmt.Errorf("%w for mkdir %+.3o: suid and sgid are ignored by mkdir", errInvalidMode, mode)
	}

	// Try to open as much of the path as possible.
	currentDir, remainingPath, err := partialLookupInRoot(root, unsafePath)
	defer func() {
		if Err != nil {
			_ = currentDir.Close()
		}
	}()
	if err != nil && !errors.Is(err, unix.ENOENT) {
		return nil, fmt.Errorf("find existing subpath of %q: %w", unsafePath, err)
	}

	// If there is an attacker deleting directories as we walk into them,
	// detect this proactively. Note this is guaranteed to detect if the
	// attacker deleted any part of the tree up to currentDir.
	//
	// Once we walk into a dead directory, partialLookupInRoot would not be
	// able to walk further down the tree (directories must be empty before
	// they are deleted), and if the attacker has removed the entire tree we
	// can be sure that anything that was originally inside a dead directory
	// must also be deleted and thus is a dead directory in its own right.
	//
	// This is mostly a quality-of-life check, because mkdir will simply fail
	// later if the attacker deletes the tree after this check.
	if err := isDeadInode(currentDir); err != nil {
		return nil, fmt.Errorf("finding existing subpath of %q: %w", unsafePath, err)
	}

	// Re-open the path to match the O_DIRECTORY reopen loop later (so that we
	// always return a non-O_PATH handle). We also check that we actually got a
	// directory.
	if reopenDir, err := Reopen(currentDir, unix.O_DIRECTORY|unix.O_CLOEXEC); errors.Is(err, unix.ENOTDIR) {
		return nil, fmt.Errorf("cannot create subdirectories in %q: %w", currentDir.Name(), unix.ENOTDIR)
	} else if err != nil {
		return nil, fmt.Errorf("re-opening handle to %q: %w", currentDir.Name(), err)
	} else {
		_ = currentDir.Close()
		currentDir = reopenDir
	}

	remainingParts := strings.Split(remainingPath, string(filepath.Separator))
	if slices_Contains(remainingParts, "..") {
		// The path contained ".." components after the end of the "real"
		// components. We could try to safely resolve ".." here but that would
		// add a bunch of extra logic for something that it's not clear even
		// needs to be supported. So just return an error.
		//
		// If we do filepath.Clean(remainingPath) then we end up with the
		// problem that ".." can erase a trailing dangling symlink and produce
		// a path that doesn't quite match what the user asked for.
		return nil, fmt.Errorf("%w: yet-to-be-created path %q contains '..' components", unix.ENOENT, remainingPath)
	}

	// Create the remaining components.
	for _, part := range remainingParts {
		switch part {
		case "", ".":
			// Skip over no-op paths.
			continue
		}

		// NOTE: mkdir(2) will not follow trailing symlinks, so we can safely
		// create the final component without worrying about symlink-exchange
		// attacks.
		//
		// If we get -EEXIST, it's possible that another program created the
		// directory at the same time as us. In that case, just continue on as
		// if we created it (if the created inode is not a directory, the
		// following open call will fail).
		if err := unix.Mkdirat(int(currentDir.Fd()), part, unixMode); err != nil && !errors.Is(err, unix.EEXIST) {
			err = &os.PathError{Op: "mkdirat", Path: currentDir.Name() + "/" + part, Err: err}
			// Make the error a bit nicer if the directory is dead.
			if deadErr := isDeadInode(currentDir); deadErr != nil {
				// TODO: Once we bump the minimum Go version to 1.20, we can use
				// multiple %w verbs for this wrapping. For now we need to use a
				// compatibility shim for older Go versions.
				//err = fmt.Errorf("%w (%w)", err, deadErr)
				err = wrapBaseError(err, deadErr)
			}
			return nil, err
		}

		// Get a handle to the next component. O_DIRECTORY means we don't need
		// to use O_PATH.
		var nextDir *os.File
		if hasOpenat2() {
			nextDir, err = openat2File(currentDir, part, &unix.OpenHow{
				Flags:   unix.O_NOFOLLOW | unix.O_DIRECTORY | unix.O_CLOEXEC,
				Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_SYMLINKS | unix.RESOLVE_NO_XDEV,
			})
		} else {
			nextDir, err = openatFile(currentDir, part, unix.O_NOFOLLOW|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
		}
		if err != nil {
			return nil, err
		}
		_ = currentDir.Close()
		currentDir = nextDir

		// It's possible that the directory we just opened was swapped by an
		// attacker. Unfortunately there isn't much we can do to protect
		// against this, and MkdirAll's behaviour is that we will reuse
		// existing directories anyway so the need to protect against this is
		// incredibly limited (and arguably doesn't even deserve mention here).
		//
		// Ideally we might want to check that the owner and mode match what we
		// would've created -- unfortunately, it is non-trivial to verify that
		// the owner and mode of the created directory match. While plain Unix
		// DAC rules seem simple enough to emulate, there are a bunch of other
		// factors that can change the mode or owner of created directories
		// (default POSIX ACLs, mount options like uid=1,gid=2,umask=0 on
		// filesystems like vfat, etc etc). We used to try to verify this but
		// it just lead to a series of spurious errors.
		//
		// We could also check that the directory is non-empty, but
		// unfortunately some pseduofilesystems (like cgroupfs) create
		// non-empty directories, which would result in different spurious
		// errors.
	}
	return currentDir, nil
}

// MkdirAll is a race-safe alternative to the [os.MkdirAll] function,
// where the new directory is guaranteed to be within the root directory (if an
// attacker can move directories from inside the root to outside the root, the
// created directory tree might be outside of the root but the key constraint
// is that at no point will we walk outside of the directory tree we are
// creating).
//
// Effectively, MkdirAll(root, unsafePath, mode) is equivalent to
//
//	path, _ := securejoin.SecureJoin(root, unsafePath)
//	err := os.MkdirAll(path, mode)
//
// But is much safer. The above implementation is unsafe because if an attacker
// can modify the filesystem tree between [SecureJoin] and [os.MkdirAll], it is
// possible for MkdirAll to resolve unsafe symlink components and create
// directories outside of the root.
//
// If you plan to open the directory after you have created it or want to use
// an open directory handle as the root, you should use [MkdirAllHandle] instead.
// This function is a wrapper around [MkdirAllHandle].
func MkdirAll(root, unsafePath string, mode os.FileMode) error {
	rootDir, err := os.OpenFile(root, unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer rootDir.Close()

	f, err := MkdirAllHandle(rootDir, unsafePath, mode)
	if err != nil {
		return err
	}
	_ = f.Close()
	return nil
}
