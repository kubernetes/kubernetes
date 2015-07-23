// +build linux

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/golang/glog"
	"os"
	"syscall"
)

// ContainerMounter is part of experimental support for running mount tools
// in a container.
type ContainerMounter struct {
	executor ContainerExecutor
	config   *MountConfig
}

// ContainerMounter implements mount.Interface
var _ = Interface(&ContainerMounter{})

func (cm *ContainerMounter) SetRunner(executor ContainerExecutor) {
	cm.executor = executor
}

// Mount runs mount(8) in the containermount namespace.  Aside from this
// aspect, Mount has the same semantics as the mounter returned by mount.New()
func (cm *ContainerMounter) Mount(source string, target string, fstype string, options []string) error {
	bind, bindRemountOpts := isBind(options)

	if bind {
		err := cm.doContainerMount(source, target, fstype, []string{"bind"})
		if err != nil {
			return err
		}
		return cm.doContainerMount(source, target, fstype, bindRemountOpts)
	}

	return cm.doContainerMount(source, target, fstype, options)
}

// doContainerMount nsenters the host's mount namespace and performs the
// requested mount.
func (cm *ContainerMounter) doContainerMount(source, target, fstype string, options []string) error {
	spec := cm.findMountContainer(fstype)
	if spec == nil {
		return doMount(source, target, fstype, options)
	}
	cmd := []string{
		"/bin/mount",
	}
	args := makeMountArgs(source, target, fstype, options)
	cmd = append(cmd, args...)

	glog.V(5).Infof("Mount command: %v", cmd)
	out, err := cm.executor.RunInContainerBySelector(spec.Selector, spec.ContainerName, cmd)
	glog.V(5).Infof("Output of containerized mount command: %s", string(out))
	return err
}

func (cm *ContainerMounter) findMountContainer(fstype string) *MountContainerConfig {
	spec, present := cm.config.MountContainers[fstype]
	if !present {
		// no container defined for this fstype, use standard mount
		return nil
	}
	return spec
}

// Unmount runs umount(8) in the host's mount namespace.
func (*ContainerMounter) Unmount(target string) error {
	args := []string{
		target,
	}

	glog.V(5).Infof("Unmount command: umount %v", args)
	exec := exec.New()
	outputBytes, err := exec.Command("umount", args...).CombinedOutput()
	if len(outputBytes) != 0 {
		glog.V(5).Infof("Output from mount command: %v", string(outputBytes))
	}

	return err
}

// List returns a list of all mounted filesystems in the host's mount namespace.
func (*ContainerMounter) List() ([]MountPoint, error) {
	return listProcMounts(hostProcMountsPath)
}

// IsMountPoint determines whether a path is a mountpoint by calling findmnt
// in the host's root mount namespace.
func (*ContainerMounter) IsMountPoint(file string) (bool, error) {
	stat, err := os.Stat(file)
	if err != nil {
		return false, err
	}
	rootStat, err := os.Lstat(file + "/..")
	if err != nil {
		return false, err
	}
	// If the directory has the same device as parent, then it's not a mountpoint.
	return stat.Sys().(*syscall.Stat_t).Dev != rootStat.Sys().(*syscall.Stat_t).Dev, nil
}
