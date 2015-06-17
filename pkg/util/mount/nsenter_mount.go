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
	"os"
	"path/filepath"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/golang/glog"
)

// NsenterMounter is part of experimental support for running the kubelet
// in a container.  Currently, all docker containers receive their own mount
// namespaces.  NsenterMounter works by executing nsenter to run commands in
// the host's mount namespace.
//
// NsenterMounter requires:
//
// 1.  Docker >= 1.6 due to the dependency on the slave propagation mode
//     of the bind-mount of the kubelet root directory in the container.
//     Docker 1.5 used a private propagation mode for bind-mounts, so mounts
//     performed in the host's mount namespace do not propagate out to the
//     bind-mount in this docker version.
// 2.  The host's root filesystem must be available at /rootfs
// 3.  The nsenter binary must be on the Kubelet process' PATH in the container's
//     filesystem.
// 4.  The Kubelet process must have CAP_SYS_ADMIN (required by nsenter); at
//     the present, this effectively means that the kubelet is running in a
//     privileged container.
// 5.  The volume path used by the Kubelet must be the same inside and outside
//     the container and be writable by the container (to initialize volume)
//     contents. TODO: remove this requirement.
// 6.  The host image must have mount, findmnt, and umount binaries in /bin,
//     /usr/sbin, or /usr/bin
//
// For more information about mount propagation modes, see:
//   https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
type NsenterMounter struct {
	// a map of commands to their paths on the host filesystem
	paths map[string]string
}

func NewNsenterMounter() *NsenterMounter {
	m := &NsenterMounter{
		paths: map[string]string{
			"mount":   "",
			"findmnt": "",
			"umount":  "",
		},
	}
	// search for the mount command in other locations besides /usr/bin
	for binary := range m.paths {
		// default to root
		m.paths[binary] = filepath.Join("/", binary)
		for _, path := range []string{"/bin", "/usr/sbin", "/usr/bin"} {
			binPath := filepath.Join(hostRootFsPath, path, binary)
			if _, err := os.Stat(binPath); err != nil {
				continue
			}
			m.paths[binary] = binPath
			break
		}
		// TODO: error, so that the kubelet can stop if the mounts don't exist
	}
	return m
}

// NsenterMounter implements mount.Interface
var _ = Interface(&NsenterMounter{})

const (
	hostRootFsPath     = "/rootfs"
	hostProcMountsPath = "/rootfs/proc/mounts"
	nsenterPath        = "nsenter"
)

// Mount runs mount(8) in the host's root mount namespace.  Aside from this
// aspect, Mount has the same semantics as the mounter returned by mount.New()
func (n *NsenterMounter) Mount(source string, target string, fstype string, options []string) error {
	bind, bindRemountOpts := isBind(options)

	if bind {
		err := n.doNsenterMount(source, target, fstype, []string{"bind"})
		if err != nil {
			return err
		}
		return n.doNsenterMount(source, target, fstype, bindRemountOpts)
	}

	return n.doNsenterMount(source, target, fstype, options)
}

// doNsenterMount nsenters the host's mount namespace and performs the
// requested mount.
func (n *NsenterMounter) doNsenterMount(source, target, fstype string, options []string) error {
	glog.V(5).Infof("nsenter Mounting %s %s %s %v", source, target, fstype, options)
	args := n.makeNsenterArgs(source, target, fstype, options)

	glog.V(5).Infof("Mount command: %v %v", nsenterPath, args)
	exec := exec.New()
	outputBytes, err := exec.Command(nsenterPath, args...).CombinedOutput()
	if len(outputBytes) != 0 {
		glog.V(5).Infof("Output from mount command: %v", string(outputBytes))
	}

	return err
}

// makeNsenterArgs makes a list of argument to nsenter in order to do the
// requested mount.
func (n *NsenterMounter) makeNsenterArgs(source, target, fstype string, options []string) []string {
	nsenterArgs := []string{
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
		n.absHostPath("mount"),
	}

	args := makeMountArgs(source, target, fstype, options)

	return append(nsenterArgs, args...)
}

// Unmount runs umount(8) in the host's mount namespace.
func (n *NsenterMounter) Unmount(target string) error {
	args := []string{
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
		n.absHostPath("umount"),
		target,
	}

	glog.V(5).Infof("Unmount command: %v %v", nsenterPath, args)
	exec := exec.New()
	outputBytes, err := exec.Command(nsenterPath, args...).CombinedOutput()
	if len(outputBytes) != 0 {
		glog.V(5).Infof("Output from mount command: %v", string(outputBytes))
	}

	return err
}

// List returns a list of all mounted filesystems in the host's mount namespace.
func (*NsenterMounter) List() ([]MountPoint, error) {
	return listProcMounts(hostProcMountsPath)
}

// IsMountPoint determines whether a path is a mountpoint by calling findmnt
// in the host's root mount namespace.
func (n *NsenterMounter) IsMountPoint(file string) (bool, error) {
	file, err := filepath.Abs(file)
	if err != nil {
		return false, err
	}

	args := []string{"--mount=/rootfs/proc/1/ns/mnt", "--", n.absHostPath("findmnt"), "-o", "target", "--noheadings", "--target", file}
	glog.V(5).Infof("findmnt command: %v %v", nsenterPath, args)

	exec := exec.New()
	out, err := exec.Command(nsenterPath, args...).CombinedOutput()
	if err != nil {
		// If findmnt didn't run, just claim it's not a mount point.
		return false, nil
	}
	strOut := strings.TrimSuffix(string(out), "\n")

	glog.V(5).Infof("IsMountPoint findmnt output: %v", strOut)
	if strOut == file {
		return true, nil
	}

	return false, nil
}

func (n *NsenterMounter) absHostPath(command string) string {
	path, ok := n.paths[command]
	if !ok {
		return command
	}
	return path
}
