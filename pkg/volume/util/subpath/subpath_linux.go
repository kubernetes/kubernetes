// +build linux

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
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
	"k8s.io/utils/mount"
)

const (
	// syscall.Openat flags used to traverse directories not following symlinks
	nofollowFlags = unix.O_RDONLY | unix.O_NOFOLLOW
	// flags for getting file descriptor without following the symlink
	openFDFlags = unix.O_NOFOLLOW | unix.O_PATH
)

func (sp *subpath) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	newHostPath, err = doBindSubPath(sp.mounter, subPath)

	// There is no action when the container starts. Bind-mount will be cleaned
	// when container stops by CleanSubPaths.
	cleanupAction = nil
	return newHostPath, cleanupAction, err
}

// This implementation is shared between Linux and NsEnter
func safeOpenSubPath(mounter mount.Interface, subpath Subpath) (int, error) {
	if !mount.PathWithinBase(subpath.Path, subpath.VolumePath) {
		return -1, fmt.Errorf("subpath %q not within volume path %q", subpath.Path, subpath.VolumePath)
	}
	fd, err := doSafeOpen(subpath.Path, subpath.VolumePath)
	if err != nil {
		return -1, fmt.Errorf("error opening subpath %v: %v", subpath.Path, err)
	}
	return fd, nil
}

func bindMount(mounter mount.Interface, subpath Subpath, bindPathTarget string) error {
	fd, err := safeOpenSubPath(mounter, subpath)
	if err != nil {
		return err
	}
	defer syscall.Close(fd)

	kubeletPid := os.Getpid()
	mountSource := fmt.Sprintf("/proc/%d/fd/%v", kubeletPid, fd)

	options := []string{"bind"}
	klog.V(5).Infof("bind mounting %q at %q", mountSource, bindPathTarget)
	if err = mounter.Mount(mountSource, bindPathTarget, "" /*fstype*/, options); err != nil {
		return fmt.Errorf("error mounting %s: %s", subpath.Path, err)
	}

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
		err = syscall.Mkdirat(parentFD, currentPath, uint32(perm))
		if err != nil {
			return fmt.Errorf("cannot create directory %s: %s", currentPath, err)
		}
		// Dive into the created directory
		childFD, err = syscall.Openat(parentFD, dir, nofollowFlags|unix.O_CLOEXEC, 0)
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
	}

	// Everything was created. mkdirat(..., perm) above was affected by current
	// umask and we must apply the right permissions to the last directory
	// (that's the one that will be available to the container as subpath)
	// so user can read/write it. This is the behavior of previous code.
	// TODO: chmod all created directories, not just the last one.
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
		childFD, err := syscall.Openat(fd, dir, unix.O_PATH|unix.O_CLOEXEC, 0)
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
		currentPath = filepath.Join(currentPath, seg)
		if !mount.PathWithinBase(currentPath, base) {
			return -1, fmt.Errorf("path %s is outside of allowed base %s", currentPath, base)
		}

		klog.V(5).Infof("Opening path %s", currentPath)
		childFD, err = syscall.Openat(parentFD, seg, openFDFlags|unix.O_CLOEXEC, 0)
		if err != nil {
			return -1, fmt.Errorf("cannot open %s: %s", currentPath, err)
		}

		var deviceStat unix.Stat_t
		err := unix.Fstat(childFD, &deviceStat)
		if err != nil {
			return -1, fmt.Errorf("Error running fstat on %s with %v", currentPath, err)
		}
		fileFmt := deviceStat.Mode & syscall.S_IFMT
		if fileFmt == syscall.S_IFLNK {
			return -1, fmt.Errorf("Unexpected symlink found %s", currentPath)
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
