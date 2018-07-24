// +build !linux

/*
Copyright 2017 The Kubernetes Authors.

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
	"errors"
	"os"
)

type execMounter struct{}

// ExecMounter is a mounter that uses provided Exec interface to mount and
// unmount a filesystem. For all other calls it uses a wrapped mounter.
func NewExecMounter(exec Exec, wrapped Interface) Interface {
	return &execMounter{}
}

func (mounter *execMounter) Mount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *execMounter) Unmount(target string) error {
	return nil
}

func (mounter *execMounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

func (mounter *execMounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return (mp.Path == dir)
}

func (mounter *execMounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(mounter, dir)
}

func (mounter *execMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

func (mounter *execMounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return "", nil
}

func (mounter *execMounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

func (mounter *execMounter) PathIsDevice(pathname string) (bool, error) {
	return true, nil
}

func (mounter *execMounter) MakeRShared(path string) error {
	return nil
}

func (mounter *execMounter) GetFileType(pathname string) (FileType, error) {
	return FileType("fake"), errors.New("not implemented")
}

func (mounter *execMounter) MakeDir(pathname string) error {
	return nil
}

func (mounter *execMounter) MakeFile(pathname string) error {
	return nil
}

func (mounter *execMounter) ExistsPath(pathname string) (bool, error) {
	return true, errors.New("not implemented")
}

func (m *execMounter) EvalHostSymlinks(pathname string) (string, error) {
	return "", errors.New("not implemented")
}

func (mounter *execMounter) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	return subPath.Path, nil, nil
}

func (mounter *execMounter) CleanSubPaths(podDir string, volumeName string) error {
	return nil
}

func (mounter *execMounter) SafeMakeDir(pathname string, base string, perm os.FileMode) error {
	return nil
}

func (mounter *execMounter) GetMountRefs(pathname string) ([]string, error) {
	return nil, errors.New("not implemented")
}

func (mounter *execMounter) GetFSGroup(pathname string) (int64, error) {
	return -1, errors.New("not implemented")
}

func (mounter *execMounter) GetSELinuxSupport(pathname string) (bool, error) {
	return false, errors.New("not implemented")
}

func (mounter *execMounter) GetMode(pathname string) (os.FileMode, error) {
	return 0, errors.New("not implemented")
}
