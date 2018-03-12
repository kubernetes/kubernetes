// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package mount

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/golang/glog"
)

type Mounter struct {
	mounterPath string
}

func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *Mounter) Unmount(target string) error {
	return nil
}

func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

func (mounter *Mounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return (mp.Path == dir)
}

func (mounter *Mounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(mounter, dir)
}

func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

func (mounter *Mounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", nil
}

func (mounter *Mounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

func (mounter *Mounter) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	return true, nil
}

// check whether hostPath is within volume path
// this func will lock all intermediate subpath directories, need to close handle outside of this func after container started
func lockAndCheckSubPath(volumePath, hostPath string) ([]uintptr, error) {
	if len(volumePath) == 0 || len(hostPath) == 0 {
		return []uintptr{}, nil
	}

	finalSubPath, err := filepath.EvalSymlinks(hostPath)
	if err != nil {
		return []uintptr{}, fmt.Errorf("cannot read link %s: %s", hostPath, err)
	}
	finalVolumePath, err := filepath.EvalSymlinks(volumePath)
	if err != nil {
		return []uintptr{}, fmt.Errorf("cannot read link %s: %s", volumePath, err)
	}

	return lockAndCheckSubPathWithoutSymlink(finalVolumePath, finalSubPath)
}

// lock all intermediate subPath directories and check they are all within volumePath
// volumePath & subPath should not contain any symlink, otherwise it will return error
func lockAndCheckSubPathWithoutSymlink(volumePath, subPath string) ([]uintptr, error) {
	if len(volumePath) == 0 || len(subPath) == 0 {
		return []uintptr{}, nil
	}

	// get relative path to volumePath
	relSubPath, err := filepath.Rel(volumePath, subPath)
	if err != nil {
		return []uintptr{}, fmt.Errorf("Rel(%s, %s) error: %v", volumePath, subPath, err)
	}
	if strings.HasPrefix(relSubPath, "..") {
		return []uintptr{}, fmt.Errorf("SubPath %q not within volume path %q", subPath, volumePath)
	}

	if relSubPath == "." {
		// volumePath and subPath are equal
		return []uintptr{}, nil
	}

	fileHandles := []uintptr{}
	var errorResult error

	currentFullPath := volumePath
	dirs := strings.Split(relSubPath, string(os.PathSeparator))
	for _, dir := range dirs {
		// lock intermediate subPath directory first
		currentFullPath = filepath.Join(currentFullPath, dir)
		handle, err := lockPath(currentFullPath)
		if err != nil {
			errorResult = fmt.Errorf("cannot lock path %s: %s", currentFullPath, err)
			break
		}
		fileHandles = append(fileHandles, handle)

		// make sure intermediate subPath directory does not contain symlink any more
		stat, err := os.Lstat(currentFullPath)
		if err != nil {
			errorResult = fmt.Errorf("Lstat(%q) error: %v", currentFullPath, err)
			break
		}
		if stat.Mode()&os.ModeSymlink != 0 {
			errorResult = fmt.Errorf("subpath %q is an unexpected symlink after EvalSymlinks", currentFullPath)
			break
		}

		if !pathWithinBase(currentFullPath, volumePath) {
			errorResult = fmt.Errorf("SubPath %q not within volume path %q", currentFullPath, volumePath)
			break
		}
	}

	return fileHandles, errorResult
}

// unlockPath unlock directories
func unlockPath(fileHandles []uintptr) {
	if fileHandles != nil {
		for _, handle := range fileHandles {
			syscall.CloseHandle(syscall.Handle(handle))
		}
	}
}

// lockPath locks a directory or symlink, return handle, exec "syscall.CloseHandle(handle)" to unlock the path
func lockPath(path string) (uintptr, error) {
	if len(path) == 0 {
		return uintptr(syscall.InvalidHandle), syscall.ERROR_FILE_NOT_FOUND
	}
	pathp, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return uintptr(syscall.InvalidHandle), err
	}
	access := uint32(syscall.GENERIC_READ)
	sharemode := uint32(syscall.FILE_SHARE_READ)
	createmode := uint32(syscall.OPEN_EXISTING)
	flags := uint32(syscall.FILE_FLAG_BACKUP_SEMANTICS | syscall.FILE_FLAG_OPEN_REPARSE_POINT)
	fd, err := syscall.CreateFile(pathp, access, sharemode, nil, createmode, flags, 0)
	return uintptr(fd), err
}

// Lock all directories in subPath and check they're not symlinks.
func (mounter *Mounter) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	handles, err := lockAndCheckSubPath(subPath.VolumePath, subPath.Path)

	// Unlock the directories when the container starts
	cleanupAction = func() {
		unlockPath(handles)
	}
	return subPath.Path, cleanupAction, err
}

// No bind-mounts for subpaths are necessary on Windows
func (mounter *Mounter) CleanSubPaths(podDir string, volumeName string) error {
	return nil
}

// SafeMakeDir makes sure that the created directory does not escape given base directory mis-using symlinks.
func (mounter *Mounter) SafeMakeDir(pathname string, base string, perm os.FileMode) error {
	return doSafeMakeDir(pathname, base, perm)
}

func doSafeMakeDir(pathname string, base string, perm os.FileMode) error {
	glog.V(4).Infof("Creating directory %q within base %q", pathname, base)

	if !pathWithinBase(pathname, base) {
		return fmt.Errorf("path %s is outside of allowed base %s", pathname, base)
	}

	// Quick check if the directory already exists
	s, err := os.Stat(pathname)
	if err == nil {
		// Path exists
		if s.IsDir() {
			// The directory already exists. It can be outside of the parent,
			// but there is no race-proof check.
			glog.V(4).Infof("Directory %s already exists", pathname)
			return nil
		}
		return &os.PathError{Op: "mkdir", Path: pathname, Err: syscall.ENOTDIR}
	}

	// Find all existing directories
	existingPath, toCreate, err := findExistingPrefix(base, pathname)
	if err != nil {
		return fmt.Errorf("error opening directory %s: %s", pathname, err)
	}
	if len(toCreate) == 0 {
		return nil
	}

	// Ensure the existing directory is inside allowed base
	fullExistingPath, err := filepath.EvalSymlinks(existingPath)
	if err != nil {
		return fmt.Errorf("error opening existing directory %s: %s", existingPath, err)
	}
	fullBasePath, err := filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("cannot read link %s: %s", base, err)
	}
	if !pathWithinBase(fullExistingPath, fullBasePath) {
		return fmt.Errorf("path %s is outside of allowed base %s", fullExistingPath, err)
	}

	// lock all intermediate directories from fullBasePath to fullExistingPath (top to bottom)
	fileHandles, err := lockAndCheckSubPathWithoutSymlink(fullBasePath, fullExistingPath)
	defer unlockPath(fileHandles)
	if err != nil {
		return err
	}

	glog.V(4).Infof("%q already exists, %q to create", fullExistingPath, filepath.Join(toCreate...))
	currentPath := fullExistingPath
	// create the directories one by one, making sure nobody can change
	// created directory into symlink by lock that directory immediately
	for _, dir := range toCreate {
		currentPath = filepath.Join(currentPath, dir)
		glog.V(4).Infof("Creating %s", dir)
		if err := os.Mkdir(currentPath, perm); err != nil {
			return fmt.Errorf("cannot create directory %s: %s", currentPath, err)
		}
		handle, err := lockPath(currentPath)
		if err != nil {
			return fmt.Errorf("cannot lock path %s: %s", currentPath, err)
		}
		defer syscall.CloseHandle(syscall.Handle(handle))
		// make sure newly created directory does not contain symlink after lock
		stat, err := os.Lstat(currentPath)
		if err != nil {
			return fmt.Errorf("Lstat(%q) error: %v", currentPath, err)
		}
		if stat.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("subpath %q is an unexpected symlink after Mkdir", currentPath)
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

	if strings.HasPrefix(rel, "..") {
		return base, nil, fmt.Errorf("pathname(%s) is not within base(%s)", pathname, base)
	}

	if rel == "." {
		// base and pathname are equal
		return pathname, []string{}, nil
	}

	dirs := strings.Split(rel, string(filepath.Separator))

	parent := base
	currentPath := base
	for i, dir := range dirs {
		parent = currentPath
		currentPath = filepath.Join(parent, dir)
		if _, err := os.Lstat(currentPath); err != nil {
			if os.IsNotExist(err) {
				return parent, dirs[i:], nil
			}
			return base, nil, err
		}
	}

	return pathname, []string{}, nil
}
