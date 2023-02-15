//go:build linux
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
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"syscall"

	"golang.org/x/sys/unix"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
)

const (
	O_PATH_PORTABLE = unix.O_PATH
)

// New returns a subpath.Interface for the current system
func New(mounter mount.Interface) Interface {
	return &subpath{
		mounter: mounter,
	}
}

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

// prepareSubpathTarget creates target for bind-mount of subpath. It returns
// "true" when the target already exists and something is mounted there.
// Given Subpath must have all paths with already resolved symlinks and with
// paths relevant to kubelet (when it runs in a container).
// This function is called also by NsEnterMounter. It works because
// /var/lib/kubelet is mounted from the host into the container with Kubelet as
// /var/lib/kubelet too.
func prepareSubpathTarget(mounter mount.Interface, subpath Subpath) (bool, string, error) {
	// Early check for already bind-mounted subpath.
	bindPathTarget := getSubpathBindTarget(subpath)
	notMount, err := mount.IsNotMountPoint(mounter, bindPathTarget)
	if err != nil {
		if !os.IsNotExist(err) {
			return false, "", fmt.Errorf("error checking path %s for mount: %s", bindPathTarget, err)
		}
		// Ignore ErrorNotExist: the file/directory will be created below if it does not exist yet.
		notMount = true
	}
	if !notMount {
		// It's already mounted, so check if it's bind-mounted to the same path
		samePath, err := checkSubPathFileEqual(subpath, bindPathTarget)
		if err != nil {
			return false, "", fmt.Errorf("error checking subpath mount info for %s: %s", bindPathTarget, err)
		}
		if !samePath {
			// It's already mounted but not what we want, unmount it
			if err = mounter.Unmount(bindPathTarget); err != nil {
				return false, "", fmt.Errorf("error ummounting %s: %s", bindPathTarget, err)
			}
		} else {
			// It's already mounted
			klog.V(5).Infof("Skipping bind-mounting subpath %s: already mounted", bindPathTarget)
			return true, bindPathTarget, nil
		}
	}

	// bindPathTarget is in /var/lib/kubelet and thus reachable without any
	// translation even to containerized kubelet.
	bindParent := filepath.Dir(bindPathTarget)
	err = os.MkdirAll(bindParent, 0750)
	if err != nil && !os.IsExist(err) {
		return false, "", fmt.Errorf("error creating directory %s: %s", bindParent, err)
	}

	t, err := os.Lstat(subpath.Path)
	if err != nil {
		return false, "", fmt.Errorf("lstat %s failed: %s", subpath.Path, err)
	}

	if t.Mode()&os.ModeDir > 0 {
		if err = os.Mkdir(bindPathTarget, 0750); err != nil && !os.IsExist(err) {
			return false, "", fmt.Errorf("error creating directory %s: %s", bindPathTarget, err)
		}
	} else {
		// "/bin/touch <bindPathTarget>".
		// A file is enough for all possible targets (symlink, device, pipe,
		// socket, ...), bind-mounting them into a file correctly changes type
		// of the target file.
		if err = ioutil.WriteFile(bindPathTarget, []byte{}, 0640); err != nil {
			return false, "", fmt.Errorf("error creating file %s: %s", bindPathTarget, err)
		}
	}
	return false, bindPathTarget, nil
}

func checkSubPathFileEqual(subpath Subpath, bindMountTarget string) (bool, error) {
	s, err := os.Lstat(subpath.Path)
	if err != nil {
		return false, fmt.Errorf("stat %s failed: %s", subpath.Path, err)
	}

	t, err := os.Lstat(bindMountTarget)
	if err != nil {
		return false, fmt.Errorf("lstat %s failed: %s", bindMountTarget, err)
	}

	if !os.SameFile(s, t) {
		return false, nil
	}
	return true, nil
}

func getSubpathBindTarget(subpath Subpath) string {
	// containerName is DNS label, i.e. safe as a directory name.
	return filepath.Join(subpath.PodDir, containerSubPathDirectoryName, subpath.VolumeName, subpath.ContainerName, strconv.Itoa(subpath.VolumeMountIndex))
}

func doBindSubPath(mounter mount.Interface, subpath Subpath) (hostPath string, err error) {
	// Linux, kubelet runs on the host:
	// - safely open the subpath
	// - bind-mount /proc/<pid of kubelet>/fd/<fd> to subpath target
	// User can't change /proc/<pid of kubelet>/fd/<fd> to point to a bad place.

	// Evaluate all symlinks here once for all subsequent functions.
	newVolumePath, err := filepath.EvalSymlinks(subpath.VolumePath)
	if err != nil {
		return "", fmt.Errorf("error resolving symlinks in %q: %v", subpath.VolumePath, err)
	}
	newPath, err := filepath.EvalSymlinks(subpath.Path)
	if err != nil {
		return "", fmt.Errorf("error resolving symlinks in %q: %v", subpath.Path, err)
	}
	klog.V(5).Infof("doBindSubPath %q (%q) for volumepath %q", subpath.Path, newPath, subpath.VolumePath)
	subpath.VolumePath = newVolumePath
	subpath.Path = newPath

	fd, err := safeOpenSubPath(mounter, subpath)
	if err != nil {
		return "", err
	}
	defer syscall.Close(fd)

	alreadyMounted, bindPathTarget, err := prepareSubpathTarget(mounter, subpath)
	if err != nil {
		return "", err
	}
	if alreadyMounted {
		return bindPathTarget, nil
	}

	success := false
	defer func() {
		// Cleanup subpath on error
		if !success {
			klog.V(4).Infof("doBindSubPath() failed for %q, cleaning up subpath", bindPathTarget)
			if cleanErr := cleanSubPath(mounter, subpath); cleanErr != nil {
				klog.Errorf("Failed to clean subpath %q: %v", bindPathTarget, cleanErr)
			}
		}
	}()

	kubeletPid := os.Getpid()
	mountSource := fmt.Sprintf("/proc/%d/fd/%v", kubeletPid, fd)

	// Do the bind mount
	options := []string{"bind"}
	mountFlags := []string{"--no-canonicalize"}
	klog.V(5).Infof("bind mounting %q at %q", mountSource, bindPathTarget)
	if err = mounter.MountSensitiveWithoutSystemdWithMountFlags(mountSource, bindPathTarget, "" /*fstype*/, options, nil /* sensitiveOptions */, mountFlags); err != nil {
		return "", fmt.Errorf("error mounting %s: %s", subpath.Path, err)
	}
	success = true

	klog.V(3).Infof("Bound SubPath %s into %s", subpath.Path, bindPathTarget)
	return bindPathTarget, nil
}

// cleanSubPath will teardown the subpath bind mount and any remove any directories if empty
func cleanSubPath(mounter mount.Interface, subpath Subpath) error {
	containerDir := filepath.Join(subpath.PodDir, containerSubPathDirectoryName, subpath.VolumeName, subpath.ContainerName)

	// Clean subdir bindmount
	if err := doCleanSubPath(mounter, containerDir, strconv.Itoa(subpath.VolumeMountIndex)); err != nil && !os.IsNotExist(err) {
		return err
	}

	// Recusively remove directories if empty
	if err := removeEmptyDirs(subpath.PodDir, containerDir); err != nil {
		return err
	}

	return nil
}

// removeEmptyDirs works backwards from endDir to baseDir and removes each directory
// if it is empty.  It stops once it encounters a directory that has content
func removeEmptyDirs(baseDir, endDir string) error {
	if !mount.PathWithinBase(endDir, baseDir) {
		return fmt.Errorf("endDir %q is not within baseDir %q", endDir, baseDir)
	}

	for curDir := endDir; curDir != baseDir; curDir = filepath.Dir(curDir) {
		s, err := os.Stat(curDir)
		if err != nil {
			if os.IsNotExist(err) {
				klog.V(5).Infof("curDir %q doesn't exist, skipping", curDir)
				continue
			}
			return fmt.Errorf("error stat %q: %v", curDir, err)
		}
		if !s.IsDir() {
			return fmt.Errorf("path %q not a directory", curDir)
		}

		err = os.Remove(curDir)
		if os.IsExist(err) {
			klog.V(5).Infof("Directory %q not empty, not removing", curDir)
			break
		} else if err != nil {
			return fmt.Errorf("error removing directory %q: %v", curDir, err)
		}
		klog.V(5).Infof("Removed directory %q", curDir)
	}
	return nil
}

func doMkdirat(dirfd int, path string, mode uint32) (err error) {
	return syscall.Mkdirat(dirfd, path, mode)
}

func doOpenat(fd int, path string, flags int, mode uint32) (int, error) {
	return syscall.Openat(fd, path, flags, mode)
}
