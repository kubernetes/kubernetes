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

func (m *execMounter) IsMountPointMatch(mp mount.MountPoint, dir string) bool {
	return m.wrappedMounter.IsMountPointMatch(mp, dir)
}

// IsLikelyNotMountPoint determines whether a path is a mountpoint.
func (m *execMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return m.wrappedMounter.IsLikelyNotMountPoint(file)
}

func (m *execMounter) GetMountRefs(pathname string) ([]string, error) {
	return m.wrappedMounter.GetMountRefs(pathname)
}
