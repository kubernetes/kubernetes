//go:build linux || freebsd
// +build linux freebsd

/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package subpath

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
)

const (
	// place for subpath mounts
	// TODO: pass in directory using kubelet_getters instead
	containerSubPathDirectoryName = "volume-subpaths"
	// syscall.Openat flags used to traverse directories not following symlinks
	nofollowFlags = unix.O_RDONLY | unix.O_NOFOLLOW
	// flags for getting file descriptor without following the symlink
	openFDFlags = unix.O_NOFOLLOW | O_PATH_PORTABLE
)

type subpath struct {
	mounter mount.Interface
}

func (sp *subpath) CleanSubPaths(podDir string, volumeName string) error {
	return doCleanSubPaths(sp.mounter, podDir, volumeName)
}

func (sp *subpath) SafeMakeDir(subdir string, base string, perm os.FileMode) error {
	realBase, err := filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("error resolving symlinks in %s: %s", base, err)
	}

	realFullPath := filepath.Join(realBase, subdir)

	return doSafeMakeDir(realFullPath, realBase, perm)
}

// This implementation is shared between Linux and NsEnter
func doCleanSubPaths(mounter mount.Interface, podDir string, volumeName string) error {
	// scan /var/lib/kubelet/pods/<uid>/volume-subpaths/<volume>/*
	subPathDir := filepath.Join(podDir, containerSubPathDirectoryName, volumeName)
	klog.V(4).Infof("Cleaning up subpath mounts for %s", subPathDir)

	containerDirs, err := ioutil.ReadDir(subPathDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("error reading %s: %s", subPathDir, err)
	}

	for _, containerDir := range containerDirs {
		if !containerDir.IsDir() {
			klog.V(4).Infof("Container file is not a directory: %s", containerDir.Name())
			continue
		}
		klog.V(4).Infof("Cleaning up subpath mounts for container %s", containerDir.Name())

		// scan /var/lib/kubelet/pods/<uid>/volume-subpaths/<volume>/<container name>/*
		fullContainerDirPath := filepath.Join(subPathDir, containerDir.Name())
		// The original traversal method here was ReadDir, which was not so robust to handle some error such as "stale NFS file handle",
		// so it was replaced with filepath.Walk in a later patch, which can pass through error and handled by the callback WalkFunc.
		// After go 1.16, WalkDir was introduced, it's more effective than Walk because the callback WalkDirFunc is called before
		// reading a directory, making it save some time when a container's subPath contains lots of dirs.
		// See https://github.com/kubernetes/kubernetes/pull/71804 and https://github.com/kubernetes/kubernetes/issues/107667 for more details.
		err = filepath.WalkDir(fullContainerDirPath, func(path string, info os.DirEntry, _ error) error {
			if path == fullContainerDirPath {
				// Skip top level directory
				return nil
			}

			// pass through errors and let doCleanSubPath handle them
			if err = doCleanSubPath(mounter, fullContainerDirPath, filepath.Base(path)); err != nil {
				return err
			}

			// We need to check that info is not nil. This may happen when the incoming err is not nil due to stale mounts or permission errors.
			if info != nil && info.IsDir() {
				// skip subdirs of the volume: it only matters the first level to unmount, otherwise it would try to unmount subdir of the volume
				return filepath.SkipDir
			}

			return nil
		})
		if err != nil {
			return fmt.Errorf("error processing %s: %s", fullContainerDirPath, err)
		}

		// Whole container has been processed, remove its directory.
		if err := os.Remove(fullContainerDirPath); err != nil {
			return fmt.Errorf("error deleting %s: %s", fullContainerDirPath, err)
		}
		klog.V(5).Infof("Removed %s", fullContainerDirPath)
	}
	// Whole pod volume subpaths have been cleaned up, remove its subpath directory.
	if err := os.Remove(subPathDir); err != nil {
		return fmt.Errorf("error deleting %s: %s", subPathDir, err)
	}
	klog.V(5).Infof("Removed %s", subPathDir)

	// Remove entire subpath directory if it's the last one
	podSubPathDir := filepath.Join(podDir, containerSubPathDirectoryName)
	if err := os.Remove(podSubPathDir); err != nil && !os.IsExist(err) {
		return fmt.Errorf("error deleting %s: %s", podSubPathDir, err)
	}
	klog.V(5).Infof("Removed %s", podSubPathDir)
	return nil
}

// doCleanSubPath tears down the single subpath bind mount
func doCleanSubPath(mounter mount.Interface, fullContainerDirPath, subPathIndex string) error {
	// process /var/lib/kubelet/pods/<uid>/volume-subpaths/<volume>/<container name>/<subPathName>
	klog.V(4).Infof("Cleaning up subpath mounts for subpath %v", subPathIndex)
	fullSubPath := filepath.Join(fullContainerDirPath, subPathIndex)

	if err := mount.CleanupMountPoint(fullSubPath, mounter, true); err != nil {
		return fmt.Errorf("error cleaning subpath mount %s: %s", fullSubPath, err)
	}

	klog.V(4).Infof("Successfully cleaned subpath directory %s", fullSubPath)
	return nil
}

// This implementation is shared between Linux and NsEnterMounter. Both pathname
// and base must be either already resolved symlinks or thet will be resolved in
// kubelet's mount namespace (in case it runs containerized).
func doSafeMakeDir(pathname string, base string, perm os.FileMode) error {
	klog.V(4).Infof("Creating directory %q within base %q", pathname, base)

	if !mount.PathWithinBase(pathname, base) {
		return fmt.Errorf("path %s is outside of allowed base %s", pathname, base)
	}

	// Quick check if the directory already exists
	s, err := os.Stat(pathname)
	if err == nil {
		// Path exists
		if s.IsDir() {
			// The directory already exists. It can be outside of the parent,
			// but there is no race-proof check.
			klog.V(4).Infof("Directory %s already exists", pathname)
			return nil
		}
		return &os.PathError{Op: "mkdir", Path: pathname, Err: syscall.ENOTDIR}
	}

	// Find all existing directories
	existingPath, toCreate, err := findExistingPrefix(base, pathname)
	if err != nil {
		return fmt.Errorf("error opening directory %s: %s", pathname, err)
	}
	// Ensure the existing directory is inside allowed base
	fullExistingPath, err := filepath.EvalSymlinks(existingPath)
	if err != nil {
		return fmt.Errorf("error opening directory %s: %s", existingPath, err)
	}
	if !mount.PathWithinBase(fullExistingPath, base) {
		return fmt.Errorf("path %s is outside of allowed base %s", fullExistingPath, err)
	}

	klog.V(4).Infof("%q already exists, %q to create", fullExistingPath, filepath.Join(toCreate...))
	parentFD, err := doSafeOpen(fullExistingPath, base)
	if err != nil {
		return fmt.Errorf("cannot open directory %s: %s", existingPath, err)
	}
	childFD := -1
	defer func() {
		if parentFD != -1 {
			if err = syscall.Close(parentFD); err != nil {
				klog.V(4).Infof("Closing FD %v failed for safemkdir(%v): %v", parentFD, pathname, err)
			}
		}
		if childFD != -1 {
			if err = syscall.Close(childFD); err != nil {
				klog.V(4).Infof("Closing FD %v failed for safemkdir(%v): %v", childFD, pathname, err)
			}
		}
	}()

	currentPath := fullExistingPath
	// create the directories one by one, making sure nobody can change
	// created directory into symlink.
	for _, dir := range toCreate {
		currentPath = filepath.Join(currentPath, dir)
		klog.V(4).Infof("Creating %s", dir)
		err = doMkdirat(parentFD, currentPath, uint32(perm))
		if err != nil {
			return fmt.Errorf("cannot create directory %s: %s", currentPath, err)
		}
		// Dive into the created directory
		childFD, err = doOpenat(parentFD, dir, nofollowFlags|unix.O_CLOEXEC, 0)
		if err != nil {
			return fmt.Errorf("cannot open %s: %s", currentPath, err)
		}
		// We can be sure that childFD is safe to use. It could be changed
		// by user after Mkdirat() and before Openat(), however:
		// - it could not be changed to symlink - we use nofollowFlags
		// - it could be changed to a file (or device, pipe, socket, ...)
		//   but either subsequent Mkdirat() fails or we mount this file
		//   to user's container. Security is no violated in both cases
		//   and user either gets error or the file that it can already access.

		if err = syscall.Close(parentFD); err != nil {
			klog.V(4).Infof("Closing FD %v failed for safemkdir(%v): %v", parentFD, pathname, err)
		}
		parentFD = childFD
		childFD = -1

		// Everything was created. mkdirat(..., perm) above was affected by current
		// umask and we must apply the right permissions to the all created directory.
		// (that's the one that will be available to the container as subpath)
		// so user can read/write it.
		// parentFD is the last created directory.

		// Translate perm (os.FileMode) to uint32 that fchmod() expects
		kernelPerm := uint32(perm & os.ModePerm)
		if perm&os.ModeSetgid > 0 {
			kernelPerm |= syscall.S_ISGID
		}
		if perm&os.ModeSetuid > 0 {
			kernelPerm |= syscall.S_ISUID
		}
		if perm&os.ModeSticky > 0 {
			kernelPerm |= syscall.S_ISVTX
		}
		if err = syscall.Fchmod(parentFD, kernelPerm); err != nil {
			return fmt.Errorf("chmod %q failed: %s", currentPath, err)
		}
	}

	return nil
}

// findExistingPrefix finds prefix of pathname that exists. In addition, it
// returns list of remaining directories that don't exist yet.
func findExistingPrefix(base, pathname string) (string, []string, error) {
	rel, err := filepath.Rel(base, pathname)
	if err != nil {
		return base, nil, err
	}
	dirs := strings.Split(rel, string(filepath.Separator))

	// Do OpenAt in a loop to find the first non-existing dir. Resolve symlinks.
	// This should be faster than looping through all dirs and calling os.Stat()
	// on each of them, as the symlinks are resolved only once with OpenAt().
	currentPath := base
	fd, err := syscall.Open(currentPath, syscall.O_RDONLY|syscall.O_CLOEXEC, 0)
	if err != nil {
		return pathname, nil, fmt.Errorf("error opening %s: %s", currentPath, err)
	}
	defer func() {
		if err = syscall.Close(fd); err != nil {
			klog.V(4).Infof("Closing FD %v failed for findExistingPrefix(%v): %v", fd, pathname, err)
		}
	}()
	for i, dir := range dirs {
		// Using O_PATH here will prevent hangs in case user replaces directory with
		// fifo
		childFD, err := doOpenat(fd, dir, O_PATH_PORTABLE|unix.O_CLOEXEC, 0)
		if err != nil {
			if os.IsNotExist(err) {
				return currentPath, dirs[i:], nil
			}
			return base, nil, err
		}
		if err = syscall.Close(fd); err != nil {
			klog.V(4).Infof("Closing FD %v failed for findExistingPrefix(%v): %v", fd, pathname, err)
		}
		fd = childFD
		currentPath = filepath.Join(currentPath, dir)
	}
	return pathname, []string{}, nil
}

// This implementation is shared between Linux and NsEnterMounter
// Open path and return its fd.
// Symlinks are disallowed (pathname must already resolve symlinks),
// and the path must be within the base directory.
func doSafeOpen(pathname string, base string) (int, error) {
	pathname = filepath.Clean(pathname)
	base = filepath.Clean(base)

	// Calculate segments to follow
	subpath, err := filepath.Rel(base, pathname)
	if err != nil {
		return -1, err
	}
	segments := strings.Split(subpath, string(filepath.Separator))

	// Assumption: base is the only directory that we have under control.
	// Base dir is not allowed to be a symlink.
	parentFD, err := syscall.Open(base, nofollowFlags|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, fmt.Errorf("cannot open directory %s: %s", base, err)
	}
	defer func() {
		if parentFD != -1 {
			if err = syscall.Close(parentFD); err != nil {
				klog.V(4).Infof("Closing FD %v failed for safeopen(%v): %v", parentFD, pathname, err)
			}
		}
	}()

	childFD := -1
	defer func() {
		if childFD != -1 {
			if err = syscall.Close(childFD); err != nil {
				klog.V(4).Infof("Closing FD %v failed for safeopen(%v): %v", childFD, pathname, err)
			}
		}
	}()

	currentPath := base

	// Follow the segments one by one using openat() to make
	// sure the user cannot change already existing directories into symlinks.
	for _, seg := range segments {
		var deviceStat unix.Stat_t

		currentPath = filepath.Join(currentPath, seg)
		if !mount.PathWithinBase(currentPath, base) {
			return -1, fmt.Errorf("path %s is outside of allowed base %s", currentPath, base)
		}

		// Trigger auto mount if it's an auto-mounted directory, ignore error if not a directory.
		// Notice the trailing slash is mandatory, see "automount" in openat(2) and open_by_handle_at(2).
		unix.Fstatat(parentFD, seg+"/", &deviceStat, unix.AT_SYMLINK_NOFOLLOW)

		klog.V(5).Infof("Opening path %s", currentPath)
		childFD, err = doOpenat(parentFD, seg, openFDFlags|unix.O_CLOEXEC, 0)
		if err != nil {
			return -1, fmt.Errorf("cannot open %s: %s", currentPath, err)
		}

		err := unix.Fstat(childFD, &deviceStat)
		if err != nil {
			return -1, fmt.Errorf("error running fstat on %s with %v", currentPath, err)
		}
		fileFmt := deviceStat.Mode & syscall.S_IFMT
		if fileFmt == syscall.S_IFLNK {
			return -1, fmt.Errorf("unexpected symlink found %s", currentPath)
		}

		// Close parentFD
		if err = syscall.Close(parentFD); err != nil {
			return -1, fmt.Errorf("closing fd for %q failed: %v", filepath.Dir(currentPath), err)
		}
		// Set child to new parent
		parentFD = childFD
		childFD = -1
	}

	// We made it to the end, return this fd, don't close it
	finalFD := parentFD
	parentFD = -1

	return finalFD, nil
}
