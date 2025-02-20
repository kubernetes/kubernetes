//go:build linux
// +build linux

/*
Copyright 2016 The Kubernetes Authors.

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

package volume

import (
	"context"
	"fmt"
	"path/filepath"
	"sync/atomic"
	"syscall"

	"os"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	rwMask   = os.FileMode(0660)
	roMask   = os.FileMode(0440)
	execMask = os.FileMode(0110)

	progressReportDuration = 60 * time.Second
)

type VolumeOwnership struct {
	mounter             Mounter
	dir                 string
	fsGroup             *int64
	fsGroupChangePolicy *v1.PodFSGroupChangePolicy
	completionCallback  func(types.CompleteFuncParam)

	// for monitoring progress of permission change operation
	pod         *v1.Pod
	fileCounter atomic.Int64
	recorder    record.EventRecorder
}

func NewVolumeOwnership(mounter Mounter, dir string, fsGroup *int64, fsGroupChangePolicy *v1.PodFSGroupChangePolicy, completeFunc func(types.CompleteFuncParam)) *VolumeOwnership {
	vo := &VolumeOwnership{
		mounter:             mounter,
		dir:                 dir,
		fsGroup:             fsGroup,
		fsGroupChangePolicy: fsGroupChangePolicy,
		completionCallback:  completeFunc,
	}
	vo.fileCounter.Store(0)
	return vo
}

func (vo *VolumeOwnership) AddProgressNotifier(pod *v1.Pod, recorder record.EventRecorder) *VolumeOwnership {
	vo.pod = pod
	vo.recorder = recorder
	return vo
}

func (vo *VolumeOwnership) ChangePermissions() error {
	if vo.fsGroup == nil {
		return nil
	}

	if skipPermissionChange(vo.mounter, vo.dir, vo.fsGroup, vo.fsGroupChangePolicy) {
		klog.V(3).InfoS("Skipping permission and ownership change for volume", "path", vo.dir)
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	timer := time.AfterFunc(30*time.Second, func() {
		vo.initiateProgressMonitor(ctx)
	})
	defer timer.Stop()

	return vo.changePermissionsRecursively()
}

func (vo *VolumeOwnership) initiateProgressMonitor(ctx context.Context) {
	klog.Warningf("Setting volume ownership for %s and fsGroup set. If the volume has a lot of files then setting volume ownership could be slow, see https://github.com/kubernetes/kubernetes/issues/69699", vo.dir)
	if vo.pod != nil {
		go vo.monitorProgress(ctx)
	}
}

func (vo *VolumeOwnership) changePermissionsRecursively() error {
	err := walkDeep(vo.dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		vo.fileCounter.Add(1)
		return changeFilePermission(path, vo.fsGroup, vo.mounter.GetAttributes().ReadOnly, info)
	})

	if vo.completionCallback != nil {
		vo.completionCallback(types.CompleteFuncParam{
			Err: &err,
		})
	}
	return err
}

func (vo *VolumeOwnership) monitorProgress(ctx context.Context) {
	ticker := time.NewTicker(progressReportDuration)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			vo.logWarning()
		}
	}
}

func (vo *VolumeOwnership) logWarning() {
	msg := fmt.Sprintf("Setting volume ownership for %s, processed %d files", vo.dir, vo.fileCounter.Load())
	klog.Warning(msg)
	vo.recorder.Event(vo.pod, v1.EventTypeWarning, events.VolumePermissionChangeInProgress, msg)
}

// SetVolumeOwnership modifies the given volume to be owned by
// fsGroup, and sets SetGid so that newly created files are owned by
// fsGroup. If fsGroup is nil nothing is done.
func SetVolumeOwnership(mounter Mounter, dir string, fsGroup *int64, fsGroupChangePolicy *v1.PodFSGroupChangePolicy, completeFunc func(types.CompleteFuncParam)) error {
	if fsGroup == nil {
		return nil
	}

	timer := time.AfterFunc(30*time.Second, func() {
		klog.Warningf("Setting volume ownership for %s and fsGroup set. If the volume has a lot of files then setting volume ownership could be slow, see https://github.com/kubernetes/kubernetes/issues/69699", dir)
	})
	defer timer.Stop()

	if skipPermissionChange(mounter, dir, fsGroup, fsGroupChangePolicy) {
		klog.V(3).InfoS("Skipping permission and ownership change for volume", "path", dir)
		return nil
	}

	err := walkDeep(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		return changeFilePermission(path, fsGroup, mounter.GetAttributes().ReadOnly, info)
	})
	if completeFunc != nil {
		completeFunc(types.CompleteFuncParam{
			Err: &err,
		})
	}
	return err
}

func changeFilePermission(filename string, fsGroup *int64, readonly bool, info os.FileInfo) error {
	err := os.Lchown(filename, -1, int(*fsGroup))
	if err != nil {
		klog.ErrorS(err, "Lchown failed", "path", filename)
	}

	// chmod passes through to the underlying file for symlinks.
	// Symlinks have a mode of 777 but this really doesn't mean anything.
	// The permissions of the underlying file are what matter.
	// However, if one reads the mode of a symlink then chmods the symlink
	// with that mode, it changes the mode of the underlying file, overridden
	// the defaultMode and permissions initialized by the volume plugin, which
	// is not what we want; thus, we skip chmod for symlinks.
	if info.Mode()&os.ModeSymlink != 0 {
		return nil
	}

	mask := rwMask
	if readonly {
		mask = roMask
	}

	if info.IsDir() {
		mask |= os.ModeSetgid
		mask |= execMask
	}

	err = os.Chmod(filename, info.Mode()|mask)
	if err != nil {
		klog.ErrorS(err, "chmod failed", "path", filename)
	}

	return nil
}

func skipPermissionChange(mounter Mounter, dir string, fsGroup *int64, fsGroupChangePolicy *v1.PodFSGroupChangePolicy) bool {
	if fsGroupChangePolicy == nil || *fsGroupChangePolicy != v1.FSGroupChangeOnRootMismatch {
		klog.V(4).InfoS("Perform recursive ownership change for directory", "path", dir)
		return false
	}
	return !requiresPermissionChange(dir, fsGroup, mounter.GetAttributes().ReadOnly)
}

func requiresPermissionChange(rootDir string, fsGroup *int64, readonly bool) bool {
	fsInfo, err := os.Stat(rootDir)
	if err != nil {
		klog.ErrorS(err, "Performing recursive ownership change on rootDir because reading permissions of root volume failed", "path", rootDir)
		return true
	}
	stat, ok := fsInfo.Sys().(*syscall.Stat_t)
	if !ok || stat == nil {
		klog.ErrorS(nil, "Performing recursive ownership change on rootDir because reading permissions of root volume failed", "path", rootDir)
		return true
	}

	if int(stat.Gid) != int(*fsGroup) {
		klog.V(4).InfoS("Expected group ownership of volume did not match with Gid", "path", rootDir, "GID", stat.Gid)
		return true
	}
	unixPerms := rwMask

	if readonly {
		unixPerms = roMask
	}

	// if rootDir is not a directory then we should apply permission change anyways
	if !fsInfo.IsDir() {
		return true
	}
	unixPerms |= execMask
	filePerm := fsInfo.Mode().Perm()

	// We need to check if actual permissions of root directory is a superset of permissions required by unixPerms.
	// This is done by checking if permission bits expected in unixPerms is set in actual permissions of the directory.
	// We use bitwise AND operation to check set bits. For example:
	//     unixPerms: 770, filePerms: 775 : 770&775 = 770 (perms on directory is a superset)
	//     unixPerms: 770, filePerms: 770 : 770&770 = 770 (perms on directory is a superset)
	//     unixPerms: 770, filePerms: 750 : 770&750 = 750 (perms on directory is NOT a superset)
	// We also need to check if setgid bits are set in permissions of the directory.
	if (unixPerms&filePerm != unixPerms) || (fsInfo.Mode()&os.ModeSetgid == 0) {
		klog.V(4).InfoS("Performing recursive ownership change on rootDir because of mismatching mode", "path", rootDir)
		return true
	}
	return false
}

// readDirNames reads the directory named by dirname and returns
// a list of directory entries.
// We are not using filepath.readDirNames because we do not want to sort files found in a directory before changing
// permissions for performance reasons.
func readDirNames(dirname string) ([]string, error) {
	f, err := os.Open(dirname)
	if err != nil {
		return nil, err
	}
	names, err := f.Readdirnames(-1)
	f.Close()
	if err != nil {
		return nil, err
	}
	return names, nil
}

// walkDeep can be used to traverse directories and has two minor differences
// from filepath.Walk:
//   - List of files/dirs is not sorted for performance reasons
//   - callback walkFunc is invoked on root directory after visiting children dirs and files
func walkDeep(root string, walkFunc filepath.WalkFunc) error {
	info, err := os.Lstat(root)
	if err != nil {
		return walkFunc(root, nil, err)
	}
	return walk(root, info, walkFunc)
}

func walk(path string, info os.FileInfo, walkFunc filepath.WalkFunc) error {
	if !info.IsDir() {
		return walkFunc(path, info, nil)
	}
	names, err := readDirNames(path)
	if err != nil {
		return err
	}
	for _, name := range names {
		filename := filepath.Join(path, name)
		fileInfo, err := os.Lstat(filename)
		if err != nil {
			if err := walkFunc(filename, fileInfo, err); err != nil {
				return err
			}
		} else {
			err = walk(filename, fileInfo, walkFunc)
			if err != nil {
				return err
			}
		}
	}
	return walkFunc(path, info, nil)
}
