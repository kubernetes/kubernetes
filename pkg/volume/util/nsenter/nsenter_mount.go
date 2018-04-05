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

package nsenter

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/utils/nsenter"
	utilpath "k8s.io/utils/path"
)

const (
	// hostProcMountsPath is the default mount path for rootfs
	hostProcMountsPath = "/rootfs/proc/1/mounts"
	// hostProcMountinfoPath is the default mount info path for rootfs
	hostProcMountinfoPath = "/rootfs/proc/1/mountinfo"
)

// Mounter implements mount.Interface
// Currently, all docker containers receive their own mount namespaces.
// Mounter works by executing nsenter to run commands in
// the host's mount namespace.
type Mounter struct {
	ne *nsenter.Nsenter
	// rootDir is location of /var/lib/kubelet directory.
	rootDir string
}

// NewMounter creates a new mounter for kubelet that runs as a container.
func NewMounter(rootDir string, ne *nsenter.Nsenter) *Mounter {
	return &Mounter{
		rootDir: rootDir,
		ne:      ne,
	}
}

// Mounter implements mount.Interface
var _ = mount.Interface(&Mounter{})

// Mount runs mount(8) in the host's root mount namespace.  Aside from this
// aspect, Mount has the same semantics as the mounter returned by mount.New()
func (n *Mounter) Mount(source string, target string, fstype string, options []string) error {
	bind, bindOpts, bindRemountOpts := mount.IsBind(options)

	if bind {
		err := n.doNsenterMount(source, target, fstype, bindOpts)
		if err != nil {
			return err
		}
		return n.doNsenterMount(source, target, fstype, bindRemountOpts)
	}

	return n.doNsenterMount(source, target, fstype, options)
}

// doNsenterMount nsenters the host's mount namespace and performs the
// requested mount.
func (n *Mounter) doNsenterMount(source, target, fstype string, options []string) error {
	klog.V(5).Infof("nsenter mount %s %s %s %v", source, target, fstype, options)
	cmd, args := n.makeNsenterArgs(source, target, fstype, options)
	outputBytes, err := n.ne.Exec(cmd, args).CombinedOutput()
	if len(outputBytes) != 0 {
		klog.V(5).Infof("Output of mounting %s to %s: %v", source, target, string(outputBytes))
	}
	return err
}

// makeNsenterArgs makes a list of argument to nsenter in order to do the
// requested mount.
func (n *Mounter) makeNsenterArgs(source, target, fstype string, options []string) (string, []string) {
	mountCmd := n.ne.AbsHostPath("mount")
	mountArgs := mount.MakeMountArgs(source, target, fstype, options)

	if systemdRunPath, hasSystemd := n.ne.SupportsSystemd(); hasSystemd {
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
		mountCmd, mountArgs = mount.AddSystemdScope(systemdRunPath, target, mountCmd, mountArgs)
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

	return mountCmd, mountArgs
}

// Unmount runs umount(8) in the host's mount namespace.
func (n *Mounter) Unmount(target string) error {
	args := []string{target}
	// No need to execute systemd-run here, it's enough that unmount is executed
	// in the host's mount namespace. It will finish appropriate fuse daemon(s)
	// running in any scope.
	klog.V(5).Infof("nsenter unmount args: %v", args)
	outputBytes, err := n.ne.Exec("umount", args).CombinedOutput()
	if len(outputBytes) != 0 {
		klog.V(5).Infof("Output of unmounting %s: %v", target, string(outputBytes))
	}
	return err
}

// List returns a list of all mounted filesystems in the host's mount namespace.
func (*Mounter) List() ([]mount.MountPoint, error) {
	return mount.ListProcMounts(hostProcMountsPath)
}

// IsMountPointMatch tests if dir and mp are the same path
func (*Mounter) IsMountPointMatch(mp mount.MountPoint, dir string) bool {
	deletedDir := fmt.Sprintf("%s\\040(deleted)", dir)
	return (mp.Path == dir) || (mp.Path == deletedDir)
}

// IsLikelyNotMountPoint determines whether a path is a mountpoint by calling findmnt
// in the host's root mount namespace.
func (n *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	file, err := filepath.Abs(file)
	if err != nil {
		return true, err
	}

	// Check the directory exists
	if _, err = os.Stat(file); os.IsNotExist(err) {
		klog.V(5).Infof("findmnt: directory %s does not exist", file)
		return true, err
	}

	// Resolve any symlinks in file, kernel would do the same and use the resolved path in /proc/mounts
	resolvedFile, err := n.EvalHostSymlinks(file)
	if err != nil {
		return true, err
	}

	// Add --first-only option: since we are testing for the absence of a mountpoint, it is sufficient to get only
	// the first of multiple possible mountpoints using --first-only.
	// Also add fstype output to make sure that the output of target file will give the full path
	// TODO: Need more refactoring for this function. Track the solution with issue #26996
	args := []string{"-o", "target,fstype", "--noheadings", "--first-only", "--target", resolvedFile}
	klog.V(5).Infof("nsenter findmnt args: %v", args)
	out, err := n.ne.Exec("findmnt", args).CombinedOutput()
	if err != nil {
		klog.V(2).Infof("Failed findmnt command for path %s: %s %v", resolvedFile, out, err)
		// Different operating systems behave differently for paths which are not mount points.
		// On older versions (e.g. 2.20.1) we'd get error, on newer ones (e.g. 2.26.2) we'd get "/".
		// It's safer to assume that it's not a mount point.
		return true, nil
	}
	mountTarget, err := parseFindMnt(string(out))
	if err != nil {
		return false, err
	}

	klog.V(5).Infof("IsLikelyNotMountPoint findmnt output for path %s: %v:", resolvedFile, mountTarget)

	if mountTarget == resolvedFile {
		klog.V(5).Infof("IsLikelyNotMountPoint: %s is a mount point", resolvedFile)
		return false, nil
	}
	klog.V(5).Infof("IsLikelyNotMountPoint: %s is not a mount point", resolvedFile)
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
func (n *Mounter) DeviceOpened(pathname string) (bool, error) {
	return mount.ExclusiveOpenFailsOnDevice(pathname)
}

// PathIsDevice uses FileInfo returned from os.Stat to check if path refers
// to a device.
func (n *Mounter) PathIsDevice(pathname string) (bool, error) {
	pathType, err := n.GetFileType(pathname)
	isDevice := pathType == mount.FileTypeCharDev || pathType == mount.FileTypeBlockDev
	return isDevice, err
}

//GetDeviceNameFromMount given a mount point, find the volume id from checking /proc/mounts
func (n *Mounter) GetDeviceNameFromMount(mountPath, pluginMountDir string) (string, error) {
	return mount.GetDeviceNameFromMountLinux(n, mountPath, pluginMountDir)
}

// MakeRShared checks if path is shared and bind-mounts it as rshared if needed.
func (n *Mounter) MakeRShared(path string) error {
	return mount.DoMakeRShared(path, hostProcMountinfoPath)
}

// GetFileType checks for file/directory/socket/block/character devices.
func (n *Mounter) GetFileType(pathname string) (mount.FileType, error) {
	var pathType mount.FileType
	outputBytes, err := n.ne.Exec("stat", []string{"-L", "--printf=%F", pathname}).CombinedOutput()
	if err != nil {
		if strings.Contains(string(outputBytes), "No such file") {
			err = fmt.Errorf("%s does not exist", pathname)
		} else {
			err = fmt.Errorf("stat %s error: %v", pathname, string(outputBytes))
		}
		return pathType, err
	}

	switch string(outputBytes) {
	case "socket":
		return mount.FileTypeSocket, nil
	case "character special file":
		return mount.FileTypeCharDev, nil
	case "block special file":
		return mount.FileTypeBlockDev, nil
	case "directory":
		return mount.FileTypeDirectory, nil
	case "regular file", "regular empty file":
		return mount.FileTypeFile, nil
	}

	return pathType, fmt.Errorf("only recognise file, directory, socket, block device and character device")
}

// MakeDir creates a new directory.
func (n *Mounter) MakeDir(pathname string) error {
	args := []string{"-p", pathname}
	if _, err := n.ne.Exec("mkdir", args).CombinedOutput(); err != nil {
		return err
	}
	return nil
}

// MakeFile creates an empty file.
func (n *Mounter) MakeFile(pathname string) error {
	args := []string{pathname}
	if _, err := n.ne.Exec("touch", args).CombinedOutput(); err != nil {
		return err
	}
	return nil
}

// ExistsPath checks if pathname exists.
// Error is returned on any other error than "file not found".
func (n *Mounter) ExistsPath(pathname string) (bool, error) {
	// Resolve the symlinks but allow the target not to exist. EvalSymlinks
	// would return an generic error when the target does not exist.
	hostPath, err := n.ne.EvalSymlinks(pathname, false /* mustExist */)
	if err != nil {
		return false, err
	}
	kubeletpath := n.ne.KubeletPath(hostPath)
	return utilpath.Exists(utilpath.CheckFollowSymlink, kubeletpath)
}

// EvalHostSymlinks returns the path name after evaluating symlinks.
func (n *Mounter) EvalHostSymlinks(pathname string) (string, error) {
	return n.ne.EvalSymlinks(pathname, true)
}

// GetMountRefs finds all mount references to the path, returns a
// list of paths. Path could be a mountpoint path, device or a normal
// directory (for bind mount).
func (n *Mounter) GetMountRefs(pathname string) ([]string, error) {
	pathExists, pathErr := mount.PathExists(pathname)
	if !pathExists || mount.IsCorruptedMnt(pathErr) {
		return []string{}, nil
	} else if pathErr != nil {
		return nil, fmt.Errorf("Error checking path %s: %v", pathname, pathErr)
	}
	hostpath, err := n.ne.EvalSymlinks(pathname, true /* mustExist */)
	if err != nil {
		return nil, err
	}
	return mount.SearchMountPoints(hostpath, hostProcMountinfoPath)
}

// GetFSGroup returns FSGroup of pathname.
func (n *Mounter) GetFSGroup(pathname string) (int64, error) {
	hostPath, err := n.ne.EvalSymlinks(pathname, true /* mustExist */)
	if err != nil {
		return -1, err
	}
	kubeletpath := n.ne.KubeletPath(hostPath)
	return mount.GetFSGroupLinux(kubeletpath)
}

// GetSELinuxSupport tests if pathname is on a mount that supports SELinux.
func (n *Mounter) GetSELinuxSupport(pathname string) (bool, error) {
	return mount.GetSELinux(pathname, hostProcMountsPath)
}

// GetMode returns permissions of pathname.
func (n *Mounter) GetMode(pathname string) (os.FileMode, error) {
	hostPath, err := n.ne.EvalSymlinks(pathname, true /* mustExist */)
	if err != nil {
		return 0, err
	}
	kubeletpath := n.ne.KubeletPath(hostPath)
	return mount.GetModeLinux(kubeletpath)
}
