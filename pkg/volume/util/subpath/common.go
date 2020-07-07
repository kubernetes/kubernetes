/*
Copyright 2020 The Kubernetes Authors.

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
	"runtime"
	"strconv"

	"k8s.io/klog/v2"
	"k8s.io/utils/mount"
)

const (
	// place for subpath mounts
	// TODO: pass in directory using kubelet_getters instead
	containerSubPathDirectoryName = "volume-subpaths"
)

type subpath struct {
	mounter mount.Interface
}

// New returns a subpath.Interface for the current system
func New(mounter mount.Interface) Interface {
	return &subpath{
		mounter: mounter,
	}
}

func (sp *subpath) CleanSubPaths(podDir string, volumeName string) error {
	return doCleanSubPaths(sp.mounter, podDir, volumeName)
}

// SafeMakeDir makes sure that the created directory does not escape given base directory mis-using symlinks.
func (sp *subpath) SafeMakeDir(subdir string, base string, perm os.FileMode) error {
	realBase, err := filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("error resolving symlinks in %s: %s", base, err)
	}

	realFullPath := filepath.Join(realBase, subdir)
	return doSafeMakeDir(realFullPath, realBase, perm)
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
		// It's already mounted
		klog.V(5).Infof("Skipping bind-mounting subpath %s: already mounted", bindPathTarget)
		return true, bindPathTarget, nil
	}

	// bindPathTarget is in /var/lib/kubelet and thus reachable without any
	// translation even to containerized kubelet.
	bindParent := filepath.Dir(bindPathTarget)
	err = os.MkdirAll(bindParent, 0750)
	if err != nil && !os.IsExist(err) {
		return false, "", fmt.Errorf("error creating directory %s: %s", bindParent, err)
	}

	// On Windows, a symlink cannot be created if the target already exists.
	// We've already created the parent folder, nothing else to do.
	if runtime.GOOS == "windows" {
		return false, bindPathTarget, nil
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

	// Do the bind mount
	if err = bindMount(mounter, subpath, bindPathTarget); err != nil {
		return "", err
	}
	success = true

	klog.V(3).Infof("Bound SubPath %s into %s", subpath.Path, bindPathTarget)
	return bindPathTarget, nil
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
		err = filepath.Walk(fullContainerDirPath, func(path string, info os.FileInfo, _ error) error {
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
