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

package mount

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"k8s.io/utils/exec"
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
// 7.  The host image should have systemd-run in /bin, /usr/sbin, or /usr/bin
// For more information about mount propagation modes, see:
//   https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt
type NsenterMounter struct {
	// a map of commands to their paths on the host filesystem
	paths map[string]string
}

func NewNsenterMounter() *NsenterMounter {
	m := &NsenterMounter{
		paths: map[string]string{
			"mount":       "",
			"findmnt":     "",
			"umount":      "",
			"systemd-run": "",
		},
	}
	// search for the mount command in other locations besides /usr/bin
	for binary := range m.paths {
		// default to root
		m.paths[binary] = filepath.Join("/", binary)
		for _, path := range []string{"/bin", "/usr/sbin", "/usr/bin"} {
			binPath := filepath.Join(path, binary)
			if _, err := os.Stat(filepath.Join(hostRootFsPath, binPath)); err != nil {
				continue
			}
			m.paths[binary] = binPath
			break
		}
		// TODO: error, so that the kubelet can stop if the mounts don't exist
		// (don't forget that systemd-run is optional)
	}
	return m
}

// NsenterMounter implements mount.Interface
var _ = Interface(&NsenterMounter{})

const (
	hostRootFsPath     = "/rootfs"
	hostProcMountsPath = "/rootfs/proc/1/mounts"
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
		glog.V(5).Infof("Output of mounting %s to %s: %v", source, target, string(outputBytes))
	}

	return err
}

// makeNsenterArgs makes a list of argument to nsenter in order to do the
// requested mount.
func (n *NsenterMounter) makeNsenterArgs(source, target, fstype string, options []string) []string {
	mountCmd := n.absHostPath("mount")
	mountArgs := makeMountArgs(source, target, fstype, options)

	if systemdRunPath, hasSystemd := n.paths["systemd-run"]; hasSystemd {
		// Complete command line:
		// nsenter --mount=/rootfs/proc/1/ns/mnt -- /bin/systemd-run --description=... --scope -- /bin/mount -t <type> <what> <where>
		// Expected flow is:
		// * nsenter breaks out of container's mount namespace and executes
		//   host's systemd-run.
		// * systemd-run creates a transient scope (=~ cgroup) and executes its
		//   argument (/bin/mount) there.
		// * mount does its job, forks a fuse daemon if necessary and finishes.
		//   (systemd-run --scope finishes at this point, returning mount's exit
		//   code and stdout/stderr - thats one of --scope benefits).
		// * systemd keeps the fuse daemon running in the scope (i.e. in its own
		//   cgroup) until the fuse daemon dies (another --scope benefit).
		//   Kubelet container can be restarted and the fuse daemon survives.
		// * When the daemon dies (e.g. during unmount) systemd removes the
		//   scope automatically.
		mountCmd, mountArgs = addSystemdScope(systemdRunPath, target, mountCmd, mountArgs)
	} else {
		// Fall back to simple mount when the host has no systemd.
		// Complete command line:
		// nsenter --mount=/rootfs/proc/1/ns/mnt -- /bin/mount -t <type> <what> <where>
		// Expected flow is:
		// * nsenter breaks out of container's mount namespace and executes host's /bin/mount.
		// * mount does its job, forks a fuse daemon if necessary and finishes.
		// * Any fuse daemon runs in cgroup of kubelet docker container,
		//   restart of kubelet container will kill it!

		// No code here, mountCmd and mountArgs use /bin/mount
	}

	nsenterArgs := []string{
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
		mountCmd,
	}
	nsenterArgs = append(nsenterArgs, mountArgs...)

	return nsenterArgs
}

// Unmount runs umount(8) in the host's mount namespace.
func (n *NsenterMounter) Unmount(target string) error {
	args := []string{
		"--mount=/rootfs/proc/1/ns/mnt",
		"--",
		n.absHostPath("umount"),
		target,
	}
	// No need to execute systemd-run here, it's enough that unmount is executed
	// in the host's mount namespace. It will finish appropriate fuse daemon(s)
	// running in any scope.
	glog.V(5).Infof("Unmount command: %v %v", nsenterPath, args)
	exec := exec.New()
	outputBytes, err := exec.Command(nsenterPath, args...).CombinedOutput()
	if len(outputBytes) != 0 {
		glog.V(5).Infof("Output of unmounting %s: %v", target, string(outputBytes))
	}

	return err
}

// List returns a list of all mounted filesystems in the host's mount namespace.
func (*NsenterMounter) List() ([]MountPoint, error) {
	return listProcMounts(hostProcMountsPath)
}

func (m *NsenterMounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(m, dir)
}

func (*NsenterMounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	deletedDir := fmt.Sprintf("%s\\040(deleted)", dir)
	return ((mp.Path == dir) || (mp.Path == deletedDir))
}

// IsLikelyNotMountPoint determines whether a path is a mountpoint by calling findmnt
// in the host's root mount namespace.
func (n *NsenterMounter) IsLikelyNotMountPoint(file string) (bool, error) {
	file, err := filepath.Abs(file)
	if err != nil {
		return true, err
	}

	// Check the directory exists
	if _, err = os.Stat(file); os.IsNotExist(err) {
		glog.V(5).Infof("findmnt: directory %s does not exist", file)
		return true, err
	}
	// Add --first-only option: since we are testing for the absence of a mountpoint, it is sufficient to get only
	// the first of multiple possible mountpoints using --first-only.
	// Also add fstype output to make sure that the output of target file will give the full path
	// TODO: Need more refactoring for this function. Track the solution with issue #26996
	args := []string{"--mount=/rootfs/proc/1/ns/mnt", "--", n.absHostPath("findmnt"), "-o", "target,fstype", "--noheadings", "--first-only", "--target", file}
	glog.V(5).Infof("findmnt command: %v %v", nsenterPath, args)

	exec := exec.New()
	out, err := exec.Command(nsenterPath, args...).CombinedOutput()
	if err != nil {
		glog.V(2).Infof("Failed findmnt command for path %s: %v", file, err)
		// Different operating systems behave differently for paths which are not mount points.
		// On older versions (e.g. 2.20.1) we'd get error, on newer ones (e.g. 2.26.2) we'd get "/".
		// It's safer to assume that it's not a mount point.
		return true, nil
	}
	mountTarget, err := parseFindMnt(string(out))
	if err != nil {
		return false, err
	}

	glog.V(5).Infof("IsLikelyNotMountPoint findmnt output for path %s: %v:", file, mountTarget)

	if mountTarget == file {
		glog.V(5).Infof("IsLikelyNotMountPoint: %s is a mount point", file)
		return false, nil
	}
	glog.V(5).Infof("IsLikelyNotMountPoint: %s is not a mount point", file)
	return true, nil
}

// parse output of "findmnt -o target,fstype" and return just the target
func parseFindMnt(out string) (string, error) {
	// cut trailing newline
	out = strings.TrimSuffix(out, "\n")
	// cut everything after the last space - it's the filesystem type
	i := strings.LastIndex(out, " ")
	if i == -1 {
		return "", fmt.Errorf("error parsing findmnt output, expected at least one space: %q", out)
	}
	return out[:i], nil
}

// DeviceOpened checks if block device in use by calling Open with O_EXCL flag.
// Returns true if open returns errno EBUSY, and false if errno is nil.
// Returns an error if errno is any error other than EBUSY.
// Returns with error if pathname is not a device.
func (n *NsenterMounter) DeviceOpened(pathname string) (bool, error) {
	return exclusiveOpenFailsOnDevice(pathname)
}

// PathIsDevice uses FileInfo returned from os.Stat to check if path refers
// to a device.
func (n *NsenterMounter) PathIsDevice(pathname string) (bool, error) {
	return pathIsDevice(pathname)
}

//GetDeviceNameFromMount given a mount point, find the volume id from checking /proc/mounts
func (n *NsenterMounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return getDeviceNameFromMount(n, mountPath, pluginDir)
}

func (n *NsenterMounter) absHostPath(command string) string {
	path, ok := n.paths[command]
	if !ok {
		return command
	}
	return path
}
