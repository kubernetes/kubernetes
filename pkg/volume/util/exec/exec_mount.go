// +build linux

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

package exec

import (
	"fmt"
	"os"

	"k8s.io/klog"

	"k8s.io/kubernetes/pkg/util/mount"
)

// ExecMounter is a mounter that uses provided Exec interface to mount and
// unmount a filesystem. For all other calls it uses a wrapped mounter.
type execMounter struct {
	wrappedMounter mount.Interface
	exec           mount.Exec
}

// NewExecMounter returns a mounter that uses provided Exec interface to mount and
// unmount a filesystem. For all other calls it uses a wrapped mounter.
func NewExecMounter(exec mount.Exec, wrapped mount.Interface) mount.Interface {
	return &execMounter{
		wrappedMounter: wrapped,
		exec:           exec,
	}
}

// execMounter implements mount.Interface
var _ mount.Interface = &execMounter{}

// Mount runs mount(8) using given exec interface.
func (m *execMounter) Mount(source string, target string, fstype string, options []string) error {
	bind, bindOpts, bindRemountOpts := mount.IsBind(options)

	if bind {
		err := m.doExecMount(source, target, fstype, bindOpts)
		if err != nil {
			return err
		}
		return m.doExecMount(source, target, fstype, bindRemountOpts)
	}

	return m.doExecMount(source, target, fstype, options)
}

// doExecMount calls exec(mount <what> <where>) using given exec interface.
func (m *execMounter) doExecMount(source, target, fstype string, options []string) error {
	klog.V(5).Infof("Exec Mounting %s %s %s %v", source, target, fstype, options)
	mountArgs := mount.MakeMountArgs(source, target, fstype, options)
	output, err := m.exec.Run("mount", mountArgs...)
	klog.V(5).Infof("Exec mounted %v: %v: %s", mountArgs, err, string(output))
	if err != nil {
		return fmt.Errorf("mount failed: %v\nMounting command: %s\nMounting arguments: %s %s %s %v\nOutput: %s",
			err, "mount", source, target, fstype, options, string(output))
	}

	return err
}

// Unmount runs umount(8) using given exec interface.
func (m *execMounter) Unmount(target string) error {
	outputBytes, err := m.exec.Run("umount", target)
	if err == nil {
		klog.V(5).Infof("Exec unmounted %s: %s", target, string(outputBytes))
	} else {
		klog.V(5).Infof("Failed to exec unmount %s: err: %q, umount output: %s", target, err, string(outputBytes))
	}

	return err
}

// List returns a list of all mounted filesystems.
func (m *execMounter) List() ([]mount.MountPoint, error) {
	return m.wrappedMounter.List()
}

// IsLikelyNotMountPoint determines whether a path is a mountpoint.
func (m *execMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return m.wrappedMounter.IsLikelyNotMountPoint(file)
}

// DeviceOpened checks if block device in use by calling Open with O_EXCL flag.
// Returns true if open returns errno EBUSY, and false if errno is nil.
// Returns an error if errno is any error other than EBUSY.
// Returns with error if pathname is not a device.
func (m *execMounter) DeviceOpened(pathname string) (bool, error) {
	return m.wrappedMounter.DeviceOpened(pathname)
}

// PathIsDevice uses FileInfo returned from os.Stat to check if path refers
// to a device.
func (m *execMounter) PathIsDevice(pathname string) (bool, error) {
	return m.wrappedMounter.PathIsDevice(pathname)
}

//GetDeviceNameFromMount given a mount point, find the volume id from checking /proc/mounts
func (m *execMounter) GetDeviceNameFromMount(mountPath, pluginMountDir string) (string, error) {
	return m.wrappedMounter.GetDeviceNameFromMount(mountPath, pluginMountDir)
}

func (m *execMounter) IsMountPointMatch(mp mount.MountPoint, dir string) bool {
	return m.wrappedMounter.IsMountPointMatch(mp, dir)
}

func (m *execMounter) MakeRShared(path string) error {
	return m.wrappedMounter.MakeRShared(path)
}

func (m *execMounter) GetFileType(pathname string) (mount.FileType, error) {
	return m.wrappedMounter.GetFileType(pathname)
}

func (m *execMounter) MakeFile(pathname string) error {
	return m.wrappedMounter.MakeFile(pathname)
}

func (m *execMounter) MakeDir(pathname string) error {
	return m.wrappedMounter.MakeDir(pathname)
}

func (m *execMounter) ExistsPath(pathname string) (bool, error) {
	return m.wrappedMounter.ExistsPath(pathname)
}

func (m *execMounter) EvalHostSymlinks(pathname string) (string, error) {
	return m.wrappedMounter.EvalHostSymlinks(pathname)
}

func (m *execMounter) GetMountRefs(pathname string) ([]string, error) {
	return m.wrappedMounter.GetMountRefs(pathname)
}

func (m *execMounter) GetFSGroup(pathname string) (int64, error) {
	return m.wrappedMounter.GetFSGroup(pathname)
}

func (m *execMounter) GetSELinuxSupport(pathname string) (bool, error) {
	return m.wrappedMounter.GetSELinuxSupport(pathname)
}

func (m *execMounter) GetMode(pathname string) (os.FileMode, error) {
	return m.wrappedMounter.GetMode(pathname)
}
