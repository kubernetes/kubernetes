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

	"k8s.io/klog"
	"k8s.io/utils/nsenter"
	utilpath "k8s.io/utils/path"
)

const (
	// hostProcMountsPath is the default mount path for rootfs
	hostProcMountsPath = "/rootfs/proc/1/mounts"
	// hostProcMountinfoPath is the default mount info path for rootfs
	hostProcMountinfoPath = "/rootfs/proc/1/mountinfo"
)

// Currently, all docker containers receive their own mount namespaces.
// NsenterMounter works by executing nsenter to run commands in
// the host's mount namespace.
type NsenterMounter struct {
	ne *nsenter.Nsenter
	// rootDir is location of /var/lib/kubelet directory.
	rootDir string
}

// NewNsenterMounter creates a new mounter for kubelet that runs as a container.
func NewNsenterMounter(rootDir string, ne *nsenter.Nsenter) *NsenterMounter {
	return &NsenterMounter{
		rootDir: rootDir,
		ne:      ne,
	}
}

// NsenterMounter implements mount.Interface
var _ = Interface(&NsenterMounter{})

// Mount runs mount(8) in the host's root mount namespace.  Aside from this
// aspect, Mount has the same semantics as the mounter returned by mount.New()
func (n *NsenterMounter) Mount(source string, target string, fstype string, options []string) error {
	bind, bindOpts, bindRemountOpts := isBind(options)

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
func (n *NsenterMounter) doNsenterMount(source, target, fstype string, options []string) error {
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
func (n *NsenterMounter) makeNsenterArgs(source, target, fstype string, options []string) (string, []string) {
	mountCmd := n.ne.AbsHostPath("mount")
	mountArgs := makeMountArgs(source, target, fstype, options)

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

	return mountCmd, mountArgs
}

// Unmount runs umount(8) in the host's mount namespace.
func (n *NsenterMounter) Unmount(target string) error {
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
func (*NsenterMounter) List() ([]MountPoint, error) {
	return listProcMounts(hostProcMountsPath)
}

func (m *NsenterMounter) IsNotMountPoint(dir string) (bool, error) {
	return isNotMountPoint(m, dir)
}

func (*NsenterMounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	deletedDir := fmt.Sprintf("%s\\040(deleted)", dir)
	return (mp.Path == dir) || (mp.Path == deletedDir)
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
func (n *NsenterMounter) DeviceOpened(pathname string) (bool, error) {
	return exclusiveOpenFailsOnDevice(pathname)
}

// PathIsDevice uses FileInfo returned from os.Stat to check if path refers
// to a device.
func (n *NsenterMounter) PathIsDevice(pathname string) (bool, error) {
	pathType, err := n.GetFileType(pathname)
	isDevice := pathType == FileTypeCharDev || pathType == FileTypeBlockDev
	return isDevice, err
}

//GetDeviceNameFromMount given a mount point, find the volume id from checking /proc/mounts
func (n *NsenterMounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return getDeviceNameFromMount(n, mountPath, pluginDir)
}

func (n *NsenterMounter) MakeRShared(path string) error {
	return doMakeRShared(path, hostProcMountinfoPath)
}

func (mounter *NsenterMounter) GetFileType(pathname string) (FileType, error) {
	var pathType FileType
	outputBytes, err := mounter.ne.Exec("stat", []string{"-L", "--printf=%F", pathname}).CombinedOutput()
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
		return FileTypeSocket, nil
	case "character special file":
		return FileTypeCharDev, nil
	case "block special file":
		return FileTypeBlockDev, nil
	case "directory":
		return FileTypeDirectory, nil
	case "regular file":
		return FileTypeFile, nil
	}

	return pathType, fmt.Errorf("only recognise file, directory, socket, block device and character device")
}

func (mounter *NsenterMounter) MakeDir(pathname string) error {
	args := []string{"-p", pathname}
	if _, err := mounter.ne.Exec("mkdir", args).CombinedOutput(); err != nil {
		return err
	}
	return nil
}

func (mounter *NsenterMounter) MakeFile(pathname string) error {
	args := []string{pathname}
	if _, err := mounter.ne.Exec("touch", args).CombinedOutput(); err != nil {
		return err
	}
	return nil
}

func (mounter *NsenterMounter) ExistsPath(pathname string) (bool, error) {
	// Resolve the symlinks but allow the target not to exist. EvalSymlinks
	// would return an generic error when the target does not exist.
	hostPath, err := mounter.ne.EvalSymlinks(pathname, false /* mustExist */)
	if err != nil {
		return false, err
	}
	kubeletpath := mounter.ne.KubeletPath(hostPath)
	return utilpath.Exists(utilpath.CheckFollowSymlink, kubeletpath)
}

func (mounter *NsenterMounter) EvalHostSymlinks(pathname string) (string, error) {
	return mounter.ne.EvalSymlinks(pathname, true)
}

func (mounter *NsenterMounter) GetMountRefs(pathname string) ([]string, error) {
	pathExists, pathErr := PathExists(pathname)
	if !pathExists || IsCorruptedMnt(pathErr) {
		return []string{}, nil
	} else if pathErr != nil {
		return nil, fmt.Errorf("Error checking path %s: %v", pathname, pathErr)
	}
	hostpath, err := mounter.ne.EvalSymlinks(pathname, true /* mustExist */)
	if err != nil {
		return nil, err
	}
	return searchMountPoints(hostpath, hostProcMountinfoPath)
}

func (mounter *NsenterMounter) GetFSGroup(pathname string) (int64, error) {
	hostPath, err := mounter.ne.EvalSymlinks(pathname, true /* mustExist */)
	if err != nil {
		return -1, err
	}
	kubeletpath := mounter.ne.KubeletPath(hostPath)
	return getFSGroup(kubeletpath)
}

func (mounter *NsenterMounter) GetSELinuxSupport(pathname string) (bool, error) {
	return getSELinuxSupport(pathname, hostProcMountsPath)
}

func (mounter *NsenterMounter) GetMode(pathname string) (os.FileMode, error) {
	hostPath, err := mounter.ne.EvalSymlinks(pathname, true /* mustExist */)
	if err != nil {
		return 0, err
	}
	kubeletpath := mounter.ne.KubeletPath(hostPath)
	return getMode(kubeletpath)
}
