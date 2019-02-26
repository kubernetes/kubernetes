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
	"syscall"

	"golang.org/x/sys/unix"

	"k8s.io/klog"
	"k8s.io/utils/nsenter"

	"k8s.io/kubernetes/pkg/util/mount"
)

type subpathNSE struct {
	mounter mount.Interface
	ne      *nsenter.Nsenter
	rootDir string
}

// Compile time-check for all implementers of subpath interface
var _ Interface = &subpathNSE{}

// NewNSEnter returns a subpath.Interface that is to be used with the NsenterMounter
// It is only valid on Linux systems
func NewNSEnter(mounter mount.Interface, ne *nsenter.Nsenter, rootDir string) Interface {
	return &subpathNSE{
		mounter: mounter,
		ne:      ne,
		rootDir: rootDir,
	}
}

func (sp *subpathNSE) CleanSubPaths(podDir string, volumeName string) error {
	return doCleanSubPaths(sp.mounter, podDir, volumeName)
}

func (sp *subpathNSE) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	// Bind-mount the subpath to avoid using symlinks in subpaths.
	newHostPath, err = sp.doNsEnterBindSubPath(subPath)

	// There is no action when the container starts. Bind-mount will be cleaned
	// when container stops by CleanSubPaths.
	cleanupAction = nil
	return newHostPath, cleanupAction, err
}

func (sp *subpathNSE) SafeMakeDir(subdir string, base string, perm os.FileMode) error {
	fullSubdirPath := filepath.Join(base, subdir)
	evaluatedSubdirPath, err := sp.ne.EvalSymlinks(fullSubdirPath, false /* mustExist */)
	if err != nil {
		return fmt.Errorf("error resolving symlinks in %s: %s", fullSubdirPath, err)
	}
	evaluatedSubdirPath = filepath.Clean(evaluatedSubdirPath)

	evaluatedBase, err := sp.ne.EvalSymlinks(base, true /* mustExist */)
	if err != nil {
		return fmt.Errorf("error resolving symlinks in %s: %s", base, err)
	}
	evaluatedBase = filepath.Clean(evaluatedBase)

	rootDir := filepath.Clean(sp.rootDir)
	if mount.PathWithinBase(evaluatedBase, rootDir) {
		// Base is in /var/lib/kubelet. This directory is shared between the
		// container with kubelet and the host. We don't need to add '/rootfs'.
		// This is useful when /rootfs is mounted as read-only - we can still
		// create subpaths for paths in /var/lib/kubelet.
		return doSafeMakeDir(evaluatedSubdirPath, evaluatedBase, perm)
	}

	// Base is somewhere on the host's filesystem. Add /rootfs and try to make
	// the directory there.
	// This requires /rootfs to be writable.
	kubeletSubdirPath := sp.ne.KubeletPath(evaluatedSubdirPath)
	kubeletBase := sp.ne.KubeletPath(evaluatedBase)
	return doSafeMakeDir(kubeletSubdirPath, kubeletBase, perm)
}

func (sp *subpathNSE) doNsEnterBindSubPath(subpath Subpath) (hostPath string, err error) {
	// Linux, kubelet runs in a container:
	// - safely open the subpath
	// - bind-mount the subpath to target (this can be unsafe)
	// - check that we mounted the right thing by comparing device ID and inode
	//   of the subpath (via safely opened fd) and the target (that's under our
	//   control)

	// Evaluate all symlinks here once for all subsequent functions.
	evaluatedHostVolumePath, err := sp.ne.EvalSymlinks(subpath.VolumePath, true /*mustExist*/)
	if err != nil {
		return "", fmt.Errorf("error resolving symlinks in %q: %v", subpath.VolumePath, err)
	}
	evaluatedHostSubpath, err := sp.ne.EvalSymlinks(subpath.Path, true /*mustExist*/)
	if err != nil {
		return "", fmt.Errorf("error resolving symlinks in %q: %v", subpath.Path, err)
	}
	klog.V(5).Infof("doBindSubPath %q (%q) for volumepath %q", subpath.Path, evaluatedHostSubpath, subpath.VolumePath)
	subpath.VolumePath = sp.ne.KubeletPath(evaluatedHostVolumePath)
	subpath.Path = sp.ne.KubeletPath(evaluatedHostSubpath)

	// Check the subpath is correct and open it
	fd, err := safeOpenSubPath(sp.mounter, subpath)
	if err != nil {
		return "", err
	}
	defer syscall.Close(fd)

	alreadyMounted, bindPathTarget, err := prepareSubpathTarget(sp.mounter, subpath)
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
			klog.V(4).Infof("doNsEnterBindSubPath() failed for %q, cleaning up subpath", bindPathTarget)
			if cleanErr := cleanSubPath(sp.mounter, subpath); cleanErr != nil {
				klog.Errorf("Failed to clean subpath %q: %v", bindPathTarget, cleanErr)
			}
		}
	}()

	// Leap of faith: optimistically expect that nobody has modified previously
	// expanded evalSubPath with evil symlinks and bind-mount it.
	// Mount is done on the host! don't use kubelet path!
	klog.V(5).Infof("bind mounting %q at %q", evaluatedHostSubpath, bindPathTarget)
	if err = sp.mounter.Mount(evaluatedHostSubpath, bindPathTarget, "" /*fstype*/, []string{"bind"}); err != nil {
		return "", fmt.Errorf("error mounting %s: %s", evaluatedHostSubpath, err)
	}

	// Check that the bind-mount target is the same inode and device as the
	// source that we keept open, i.e. we mounted the right thing.
	err = checkDeviceInode(fd, bindPathTarget)
	if err != nil {
		return "", fmt.Errorf("error checking bind mount for subpath %s: %s", subpath.VolumePath, err)
	}

	success = true
	klog.V(3).Infof("Bound SubPath %s into %s", subpath.Path, bindPathTarget)
	return bindPathTarget, nil
}

// checkDeviceInode checks that opened file and path represent the same file.
func checkDeviceInode(fd int, path string) error {
	var srcStat, dstStat unix.Stat_t
	err := unix.Fstat(fd, &srcStat)
	if err != nil {
		return fmt.Errorf("error running fstat on subpath FD: %v", err)
	}

	err = unix.Stat(path, &dstStat)
	if err != nil {
		return fmt.Errorf("error running fstat on %s: %v", path, err)
	}

	if srcStat.Dev != dstStat.Dev {
		return fmt.Errorf("different device number")
	}
	if srcStat.Ino != dstStat.Ino {
		return fmt.Errorf("different inode")
	}
	return nil
}
