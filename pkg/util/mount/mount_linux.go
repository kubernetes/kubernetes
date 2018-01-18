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
	"bufio"
	"fmt"
	"hash/fnv"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/sets"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	utilio "k8s.io/kubernetes/pkg/util/io"
)

const (
	// How many times to retry for a consistent read of /proc/mounts.
	maxListTries = 3
	// Number of fields per line in /proc/mounts as per the fstab man page.
	expectedNumFieldsPerLine = 6
	// Location of the mount file to use
	procMountsPath = "/proc/mounts"
)

const (
	// 'fsck' found errors and corrected them
	fsckErrorsCorrected = 1
	// 'fsck' found errors but exited without correcting them
	fsckErrorsUncorrected = 4

	// place for subpath mounts
	containerSubPathDirectoryName = "volume-subpaths"
	// syscall.Openat flags used to traverse directories not following symlinks
	nofollowFlags = syscall.O_RDONLY | syscall.O_NOFOLLOW
)

// Mounter provides the default implementation of mount.Interface
// for the linux platform.  This implementation assumes that the
// kubelet is running in the host's root mount namespace.
type Mounter struct {
	mounterPath string
}

// Mount mounts source to target as fstype with given options. 'source' and 'fstype' must
// be an emtpy string in case it's not required, e.g. for remount, or for auto filesystem
// type, where kernel handles fs type for you. The mount 'options' is a list of options,
// currently come from mount(8), e.g. "ro", "remount", "bind", etc. If no more option is
// required, call Mount with an empty string list or nil.
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	// Path to mounter binary if containerized mounter is needed. Otherwise, it is set to empty.
	// All Linux distros are expected to be shipped with a mount utility that an support bind mounts.
	mounterPath := ""
	bind, bindRemountOpts := isBind(options)
	if bind {
		err := doMount(mounterPath, defaultMountCommand, source, target, fstype, []string{"bind"})
		if err != nil {
			return err
		}
		return doMount(mounterPath, defaultMountCommand, source, target, fstype, bindRemountOpts)
	}
	// The list of filesystems that require containerized mounter on GCI image cluster
	fsTypesNeedMounter := sets.NewString("nfs", "glusterfs", "ceph", "cifs")
	if fsTypesNeedMounter.Has(fstype) {
		mounterPath = mounter.mounterPath
	}
	return doMount(mounterPath, defaultMountCommand, source, target, fstype, options)
}

// isBind detects whether a bind mount is being requested and makes the remount options to
// use in case of bind mount, due to the fact that bind mount doesn't respect mount options.
// The list equals:
//   options - 'bind' + 'remount' (no duplicate)
func isBind(options []string) (bool, []string) {
	bindRemountOpts := []string{"remount"}
	bind := false

	if len(options) != 0 {
		for _, option := range options {
			switch option {
			case "bind":
				bind = true
				break
			case "remount":
				break
			default:
				bindRemountOpts = append(bindRemountOpts, option)
			}
		}
	}

	return bind, bindRemountOpts
}

// doMount runs the mount command. mounterPath is the path to mounter binary if containerized mounter is used.
func doMount(mounterPath string, mountCmd string, source string, target string, fstype string, options []string) error {
	mountArgs := makeMountArgs(source, target, fstype, options)
	if len(mounterPath) > 0 {
		mountArgs = append([]string{mountCmd}, mountArgs...)
		mountCmd = mounterPath
	}

	glog.V(4).Infof("Mounting cmd (%s) with arguments (%s)", mountCmd, mountArgs)
	command := exec.Command(mountCmd, mountArgs...)
	output, err := command.CombinedOutput()
	if err != nil {
		glog.Errorf("Mount failed: %v\nMounting command: %s\nMounting arguments: %s %s %s %v\nOutput: %s\n", err, mountCmd, source, target, fstype, options, string(output))
		return fmt.Errorf("mount failed: %v\nMounting command: %s\nMounting arguments: %s %s %s %v\nOutput: %s\n",
			err, mountCmd, source, target, fstype, options, string(output))
	}
	return err
}

// makeMountArgs makes the arguments to the mount(8) command.
func makeMountArgs(source, target, fstype string, options []string) []string {
	// Build mount command as follows:
	//   mount [-t $fstype] [-o $options] [$source] $target
	mountArgs := []string{}
	if len(fstype) > 0 {
		mountArgs = append(mountArgs, "-t", fstype)
	}
	if len(options) > 0 {
		mountArgs = append(mountArgs, "-o", strings.Join(options, ","))
	}
	if len(source) > 0 {
		mountArgs = append(mountArgs, source)
	}
	mountArgs = append(mountArgs, target)

	return mountArgs
}

// Unmount unmounts the target.
func (mounter *Mounter) Unmount(target string) error {
	glog.V(4).Infof("Unmounting %s", target)
	command := exec.Command("umount", target)
	output, err := command.CombinedOutput()
	if err != nil {
		return fmt.Errorf("Unmount failed: %v\nUnmounting arguments: %s\nOutput: %s\n", err, target, string(output))
	}
	return nil
}

// List returns a list of all mounted filesystems.
func (*Mounter) List() ([]MountPoint, error) {
	return listProcMounts(procMountsPath)
}

func (mounter *Mounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	deletedDir := fmt.Sprintf("%s\\040(deleted)", dir)
	return ((mp.Path == dir) || (mp.Path == deletedDir))
}

func (mounter *Mounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(mounter, dir)
}

// IsLikelyNotMountPoint determines if a directory is not a mountpoint.
// It is fast but not necessarily ALWAYS correct. If the path is in fact
// a bind mount from one part of a mount to another it will not be detected.
// mkdir /tmp/a /tmp/b; mount --bin /tmp/a /tmp/b; IsLikelyNotMountPoint("/tmp/b")
// will return true. When in fact /tmp/b is a mount point. If this situation
// if of interest to you, don't use this function...
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	stat, err := os.Stat(file)
	if err != nil {
		return true, err
	}
	rootStat, err := os.Lstat(filepath.Dir(strings.TrimSuffix(file, "/")))
	if err != nil {
		return true, err
	}
	// If the directory has a different device as parent, then it is a mountpoint.
	if stat.Sys().(*syscall.Stat_t).Dev != rootStat.Sys().(*syscall.Stat_t).Dev {
		return false, nil
	}

	return true, nil
}

// DeviceOpened checks if block device in use by calling Open with O_EXCL flag.
// If pathname is not a device, log and return false with nil error.
// If open returns errno EBUSY, return true with nil error.
// If open returns nil, return false with nil error.
// Otherwise, return false with error
func (mounter *Mounter) DeviceOpened(pathname string) (bool, error) {
	return exclusiveOpenFailsOnDevice(pathname)
}

// PathIsDevice uses FileInfo returned from os.Stat to check if path refers
// to a device.
func (mounter *Mounter) PathIsDevice(pathname string) (bool, error) {
	return pathIsDevice(pathname)
}

func exclusiveOpenFailsOnDevice(pathname string) (bool, error) {
	isDevice, err := pathIsDevice(pathname)
	if err != nil {
		return false, fmt.Errorf(
			"PathIsDevice failed for path %q: %v",
			pathname,
			err)
	}
	if !isDevice {
		glog.Errorf("Path %q is not refering to a device.", pathname)
		return false, nil
	}
	fd, errno := syscall.Open(pathname, syscall.O_RDONLY|syscall.O_EXCL, 0)
	// If the device is in use, open will return an invalid fd.
	// When this happens, it is expected that Close will fail and throw an error.
	defer syscall.Close(fd)
	if errno == nil {
		// device not in use
		return false, nil
	} else if errno == syscall.EBUSY {
		// device is in use
		return true, nil
	}
	// error during call to Open
	return false, errno
}

func pathIsDevice(pathname string) (bool, error) {
	finfo, err := os.Stat(pathname)
	if os.IsNotExist(err) {
		return false, nil
	}
	// err in call to os.Stat
	if err != nil {
		return false, err
	}
	// path refers to a device
	if finfo.Mode()&os.ModeDevice != 0 {
		return true, nil
	}
	// path does not refer to device
	return false, nil
}

//GetDeviceNameFromMount: given a mount point, find the device name from its global mount point
func (mounter *Mounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginDir)
}

func listProcMounts(mountFilePath string) ([]MountPoint, error) {
	hash1, err := readProcMounts(mountFilePath, nil)
	if err != nil {
		return nil, err
	}

	for i := 0; i < maxListTries; i++ {
		mps := []MountPoint{}
		hash2, err := readProcMounts(mountFilePath, &mps)
		if err != nil {
			return nil, err
		}
		if hash1 == hash2 {
			// Success
			return mps, nil
		}
		hash1 = hash2
	}
	return nil, fmt.Errorf("failed to get a consistent snapshot of %v after %d tries", mountFilePath, maxListTries)
}

// readProcMounts reads the given mountFilePath (normally /proc/mounts) and produces a hash
// of the contents.  If the out argument is not nil, this fills it with MountPoint structs.
func readProcMounts(mountFilePath string, out *[]MountPoint) (uint32, error) {
	file, err := os.Open(mountFilePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	return readProcMountsFrom(file, out)
}

func readProcMountsFrom(file io.Reader, out *[]MountPoint) (uint32, error) {
	hash := fnv.New32a()
	scanner := bufio.NewReader(file)
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		// See `man proc` for authoritative description of format of the file.
		fields := strings.Fields(line)
		if len(fields) != expectedNumFieldsPerLine {
			return 0, fmt.Errorf("wrong number of fields (expected %d, got %d): %s", expectedNumFieldsPerLine, len(fields), line)
		}

		fmt.Fprintf(hash, "%s", line)

		if out != nil {
			mp := MountPoint{
				Device: fields[0],
				Path:   fields[1],
				Type:   fields[2],
				Opts:   strings.Split(fields[3], ","),
			}

			freq, err := strconv.Atoi(fields[4])
			if err != nil {
				return 0, err
			}
			mp.Freq = freq

			pass, err := strconv.Atoi(fields[5])
			if err != nil {
				return 0, err
			}
			mp.Pass = pass

			*out = append(*out, mp)
		}
	}
	return hash.Sum32(), nil
}

// formatAndMount uses unix utils to format and mount the given disk
func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	options = append(options, "defaults")

	// Run fsck on the disk to fix repairable issues
	glog.V(4).Infof("Checking for issues with fsck on disk: %s", source)
	args := []string{"-a", source}
	cmd := mounter.Runner.Command("fsck", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		ee, isExitError := err.(utilexec.ExitError)
		switch {
		case err == utilexec.ErrExecutableNotFound:
			glog.Warningf("'fsck' not found on system; continuing mount without running 'fsck'.")
		case isExitError && ee.ExitStatus() == fsckErrorsCorrected:
			glog.Infof("Device %s has errors which were corrected by fsck.", source)
		case isExitError && ee.ExitStatus() == fsckErrorsUncorrected:
			return fmt.Errorf("'fsck' found errors on device %s but could not correct them: %s.", source, string(out))
		case isExitError && ee.ExitStatus() > fsckErrorsUncorrected:
			glog.Infof("`fsck` error %s", string(out))
		}
	}

	// Try to mount the disk
	glog.V(4).Infof("Attempting to mount disk: %s %s %s", fstype, source, target)
	mountErr := mounter.Interface.Mount(source, target, fstype, options)
	if mountErr != nil {
		// Mount failed. This indicates either that the disk is unformatted or
		// it contains an unexpected filesystem.
		existingFormat, err := mounter.getDiskFormat(source)
		if err != nil {
			return err
		}
		if existingFormat == "" {
			// Disk is unformatted so format it.
			args = []string{source}
			// Use 'ext4' as the default
			if len(fstype) == 0 {
				fstype = "ext4"
			}

			if fstype == "ext4" || fstype == "ext3" {
				args = []string{"-F", source}
			}
			glog.Infof("Disk %q appears to be unformatted, attempting to format as type: %q with options: %v", source, fstype, args)
			cmd := mounter.Runner.Command("mkfs."+fstype, args...)
			_, err := cmd.CombinedOutput()
			if err == nil {
				// the disk has been formatted successfully try to mount it again.
				glog.Infof("Disk successfully formatted (mkfs): %s - %s %s", fstype, source, target)
				return mounter.Interface.Mount(source, target, fstype, options)
			}
			glog.Errorf("format of disk %q failed: type:(%q) target:(%q) options:(%q)error:(%v)", source, fstype, target, options, err)
			return err
		} else {
			// Disk is already formatted and failed to mount
			if len(fstype) == 0 || fstype == existingFormat {
				// This is mount error
				return mountErr
			} else {
				// Block device is formatted with unexpected filesystem, let the user know
				return fmt.Errorf("failed to mount the volume as %q, it already contains %s. Mount error: %v", fstype, existingFormat, mountErr)
			}
		}
	}
	return mountErr
}

// diskLooksUnformatted uses 'lsblk' to see if the given disk is unformated
func (mounter *SafeFormatAndMount) getDiskFormat(disk string) (string, error) {
	args := []string{"-n", "-o", "FSTYPE", disk}
	cmd := mounter.Runner.Command("lsblk", args...)
	glog.V(4).Infof("Attempting to determine if disk %q is formatted using lsblk with args: (%v)", disk, args)
	dataOut, err := cmd.CombinedOutput()
	output := string(dataOut)
	glog.V(4).Infof("Output: %q", output)

	if err != nil {
		glog.Errorf("Could not determine if disk %q is formatted (%v)", disk, err)
		return "", err
	}

	// Split lsblk output into lines. Unformatted devices should contain only
	// "\n". Beware of "\n\n", that's a device with one empty partition.
	output = strings.TrimSuffix(output, "\n") // Avoid last empty line
	lines := strings.Split(output, "\n")
	if lines[0] != "" {
		// The device is formatted
		return lines[0], nil
	}

	if len(lines) == 1 {
		// The device is unformatted and has no dependent devices
		return "", nil
	}

	// The device has dependent devices, most probably partitions (LVM, LUKS
	// and MD RAID are reported as FSTYPE and caught above).
	return "unknown data, probably partitions", nil
}

func (mounter *Mounter) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	newHostPath, err = doBindSubPath(mounter, subPath, os.Getpid())
	// There is no action when the container starts. Bind-mount will be cleaned
	// when container stops by CleanSubPaths.
	cleanupAction = nil
	return newHostPath, cleanupAction, err
}

// This implementation is shared between Linux and NsEnterMounter
// kubeletPid is PID of kubelet in the PID namespace where bind-mount is done,
// i.e. pid on the *host* if kubelet runs in a container.
func doBindSubPath(mounter Interface, subpath Subpath, kubeletPid int) (hostPath string, err error) {
	// Check early for symlink. This is just a pre-check to avoid bind-mount
	// before the final check.
	evalSubPath, err := filepath.EvalSymlinks(subpath.Path)
	if err != nil {
		return "", fmt.Errorf("evalSymlinks %q failed: %v", subpath.Path, err)
	}
	glog.V(5).Infof("doBindSubPath %q, full subpath %q for volumepath %q", subpath.Path, evalSubPath, subpath.VolumePath)

	evalSubPath = filepath.Clean(evalSubPath)
	if !pathWithinBase(evalSubPath, subpath.VolumePath) {
		return "", fmt.Errorf("subpath %q not within volume path %q", evalSubPath, subpath.VolumePath)
	}

	// Prepare directory for bind mounts
	// containerName is DNS label, i.e. safe as a directory name.
	bindDir := filepath.Join(subpath.PodDir, containerSubPathDirectoryName, subpath.VolumeName, subpath.ContainerName)
	err = os.MkdirAll(bindDir, 0750)
	if err != nil && !os.IsExist(err) {
		return "", fmt.Errorf("error creating directory %s: %s", bindDir, err)
	}
	bindPathTarget := filepath.Join(bindDir, strconv.Itoa(subpath.VolumeMountIndex))

	success := false
	defer func() {
		// Cleanup subpath on error
		if !success {
			glog.V(4).Infof("doBindSubPath() failed for %q, cleaning up subpath", bindPathTarget)
			if cleanErr := cleanSubPath(mounter, subpath); cleanErr != nil {
				glog.Errorf("Failed to clean subpath %q: %v", bindPathTarget, cleanErr)
			}
		}
	}()

	// Check it's not already bind-mounted
	notMount, err := IsNotMountPoint(mounter, bindPathTarget)
	if err != nil {
		if !os.IsNotExist(err) {
			return "", fmt.Errorf("error checking path %s for mount: %s", bindPathTarget, err)
		}
		// Ignore ErrorNotExist: the file/directory will be created below if it does not exist yet.
		notMount = true
	}
	if !notMount {
		// It's already mounted
		glog.V(5).Infof("Skipping bind-mounting subpath %s: already mounted", bindPathTarget)
		success = true
		return bindPathTarget, nil
	}

	// Create target of the bind mount. A directory for directories, empty file
	// for everything else.
	t, err := os.Lstat(subpath.Path)
	if err != nil {
		return "", fmt.Errorf("lstat %s failed: %s", subpath.Path, err)
	}
	if t.Mode()&os.ModeDir > 0 {
		if err = os.Mkdir(bindPathTarget, 0750); err != nil && !os.IsExist(err) {
			return "", fmt.Errorf("error creating directory %s: %s", bindPathTarget, err)
		}
	} else {
		// "/bin/touch <bindDir>".
		// A file is enough for all possible targets (symlink, device, pipe,
		// socket, ...), bind-mounting them into a file correctly changes type
		// of the target file.
		if err = ioutil.WriteFile(bindPathTarget, []byte{}, 0640); err != nil {
			return "", fmt.Errorf("error creating file %s: %s", bindPathTarget, err)
		}
	}

	// Safe open subpath and get the fd
	fd, err := doSafeOpen(evalSubPath, subpath.VolumePath)
	if err != nil {
		return "", fmt.Errorf("error opening subpath %v: %v", evalSubPath, err)
	}
	defer syscall.Close(fd)

	mountSource := fmt.Sprintf("/proc/%d/fd/%v", kubeletPid, fd)

	// Do the bind mount
	glog.V(5).Infof("bind mounting %q at %q", mountSource, bindPathTarget)
	if err = mounter.Mount(mountSource, bindPathTarget, "" /*fstype*/, []string{"bind"}); err != nil {
		return "", fmt.Errorf("error mounting %s: %s", subpath.Path, err)
	}

	success = true
	glog.V(3).Infof("Bound SubPath %s into %s", subpath.Path, bindPathTarget)
	return bindPathTarget, nil
}

func (mounter *Mounter) CleanSubPaths(podDir string, volumeName string) error {
	return doCleanSubPaths(mounter, podDir, volumeName)
}

// This implementation is shared between Linux and NsEnterMounter
func doCleanSubPaths(mounter Interface, podDir string, volumeName string) error {
	glog.V(4).Infof("Cleaning up subpath mounts for %s", podDir)
	// scan /var/lib/kubelet/pods/<uid>/volume-subpaths/<volume>/*
	subPathDir := filepath.Join(podDir, containerSubPathDirectoryName, volumeName)
	containerDirs, err := ioutil.ReadDir(subPathDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("error reading %s: %s", subPathDir, err)
	}

	for _, containerDir := range containerDirs {
		if !containerDir.IsDir() {
			glog.V(4).Infof("Container file is not a directory: %s", containerDir.Name())
			continue
		}
		glog.V(4).Infof("Cleaning up subpath mounts for container %s", containerDir.Name())

		// scan /var/lib/kubelet/pods/<uid>/volume-subpaths/<volume>/<container name>/*
		fullContainerDirPath := filepath.Join(subPathDir, containerDir.Name())
		subPaths, err := ioutil.ReadDir(fullContainerDirPath)
		if err != nil {
			return fmt.Errorf("error reading %s: %s", fullContainerDirPath, err)
		}
		for _, subPath := range subPaths {
			if err = doCleanSubPath(mounter, fullContainerDirPath, subPath.Name()); err != nil {
				return err
			}
		}
		// Whole container has been processed, remove its directory.
		if err := os.Remove(fullContainerDirPath); err != nil {
			return fmt.Errorf("error deleting %s: %s", fullContainerDirPath, err)
		}
		glog.V(5).Infof("Removed %s", fullContainerDirPath)
	}
	// Whole pod volume subpaths have been cleaned up, remove its subpath directory.
	if err := os.Remove(subPathDir); err != nil {
		return fmt.Errorf("error deleting %s: %s", subPathDir, err)
	}
	glog.V(5).Infof("Removed %s", subPathDir)

	// Remove entire subpath directory if it's the last one
	podSubPathDir := filepath.Join(podDir, containerSubPathDirectoryName)
	if err := os.Remove(podSubPathDir); err != nil && !os.IsExist(err) {
		return fmt.Errorf("error deleting %s: %s", podSubPathDir, err)
	}
	glog.V(5).Infof("Removed %s", podSubPathDir)
	return nil
}

// doCleanSubPath tears down the single subpath bind mount
func doCleanSubPath(mounter Interface, fullContainerDirPath, subPathIndex string) error {
	// process /var/lib/kubelet/pods/<uid>/volume-subpaths/<volume>/<container name>/<subPathName>
	glog.V(4).Infof("Cleaning up subpath mounts for subpath %v", subPathIndex)
	fullSubPath := filepath.Join(fullContainerDirPath, subPathIndex)
	notMnt, err := IsNotMountPoint(mounter, fullSubPath)
	if err != nil {
		return fmt.Errorf("error checking %s for mount: %s", fullSubPath, err)
	}
	// Unmount it
	if !notMnt {
		if err = mounter.Unmount(fullSubPath); err != nil {
			return fmt.Errorf("error unmounting %s: %s", fullSubPath, err)
		}
		glog.V(5).Infof("Unmounted %s", fullSubPath)
	}
	// Remove it *non*-recursively, just in case there were some hiccups.
	if err = os.Remove(fullSubPath); err != nil {
		return fmt.Errorf("error deleting %s: %s", fullSubPath, err)
	}
	glog.V(5).Infof("Removed %s", fullSubPath)
	return nil
}

// cleanSubPath will teardown the subpath bind mount and any remove any directories if empty
func cleanSubPath(mounter Interface, subpath Subpath) error {
	containerDir := filepath.Join(subpath.PodDir, containerSubPathDirectoryName, subpath.VolumeName, subpath.ContainerName)

	// Clean subdir bindmount
	if err := doCleanSubPath(mounter, containerDir, strconv.Itoa(subpath.VolumeMountIndex)); err != nil && !os.IsNotExist(err) {
		return err
	}

	// Recusively remove directories if empty
	if err := removeEmptyDirs(subpath.PodDir, containerDir); err != nil {
		return err
	}

	return nil
}

// removeEmptyDirs works backwards from endDir to baseDir and removes each directory
// if it is empty.  It stops once it encounters a directory that has content
func removeEmptyDirs(baseDir, endDir string) error {
	if !pathWithinBase(endDir, baseDir) {
		return fmt.Errorf("endDir %q is not within baseDir %q", endDir, baseDir)
	}

	for curDir := endDir; curDir != baseDir; curDir = filepath.Dir(curDir) {
		s, err := os.Stat(curDir)
		if err != nil {
			if os.IsNotExist(err) {
				glog.V(5).Infof("curDir %q doesn't exist, skipping", curDir)
				continue
			}
			return fmt.Errorf("error stat %q: %v", curDir, err)
		}
		if !s.IsDir() {
			return fmt.Errorf("path %q not a directory", curDir)
		}

		err = os.Remove(curDir)
		if os.IsExist(err) {
			glog.V(5).Infof("Directory %q not empty, not removing", curDir)
			break
		} else if err != nil {
			return fmt.Errorf("error removing directory %q: %v", curDir, err)
		}
		glog.V(5).Infof("Removed directory %q", curDir)
	}
	return nil
}

func (mounter *Mounter) SafeMakeDir(pathname string, base string, perm os.FileMode) error {
	return doSafeMakeDir(pathname, base, perm)
}

// This implementation is shared between Linux and NsEnterMounter
func doSafeMakeDir(pathname string, base string, perm os.FileMode) error {
	glog.V(4).Infof("Creating directory %q within base %q", pathname, base)

	if !pathWithinBase(pathname, base) {
		return fmt.Errorf("path %s is outside of allowed base %s", pathname, base)
	}

	// Quick check if the directory already exists
	s, err := os.Stat(pathname)
	if err == nil {
		// Path exists
		if s.IsDir() {
			// The directory already exists. It can be outside of the parent,
			// but there is no race-proof check.
			glog.V(4).Infof("Directory %s already exists", pathname)
			return nil
		}
		return &os.PathError{Op: "mkdir", Path: pathname, Err: syscall.ENOTDIR}
	}

	// Find all existing directories
	existingPath, toCreate, err := findExistingPrefix(base, pathname)
	if err != nil {
		return fmt.Errorf("error opening directory %s: %s", pathname, err)
	}
	// Ensure the existing directory is inside allowed base
	fullExistingPath, err := filepath.EvalSymlinks(existingPath)
	if err != nil {
		return fmt.Errorf("error opening directory %s: %s", existingPath, err)
	}
	if !pathWithinBase(fullExistingPath, base) {
		return fmt.Errorf("path %s is outside of allowed base %s", fullExistingPath, err)
	}

	glog.V(4).Infof("%q already exists, %q to create", fullExistingPath, filepath.Join(toCreate...))
	parentFD, err := doSafeOpen(fullExistingPath, base)
	if err != nil {
		return fmt.Errorf("cannot open directory %s: %s", existingPath, err)
	}
	childFD := -1
	defer func() {
		if parentFD != -1 {
			if err = syscall.Close(parentFD); err != nil {
				glog.V(4).Infof("Closing FD %v failed for safemkdir(%v): %v", parentFD, pathname, err)
			}
		}
		if childFD != -1 {
			if err = syscall.Close(childFD); err != nil {
				glog.V(4).Infof("Closing FD %v failed for safemkdir(%v): %v", childFD, pathname, err)
			}
		}
	}()

	currentPath := fullExistingPath
	// create the directories one by one, making sure nobody can change
	// created directory into symlink.
	for _, dir := range toCreate {
		currentPath = filepath.Join(currentPath, dir)
		glog.V(4).Infof("Creating %s", dir)
		err = syscall.Mkdirat(parentFD, currentPath, uint32(perm))
		if err != nil {
			return fmt.Errorf("cannot create directory %s: %s", currentPath, err)
		}
		// Dive into the created directory
		childFD, err := syscall.Openat(parentFD, dir, nofollowFlags, 0)
		if err != nil {
			return fmt.Errorf("cannot open %s: %s", currentPath, err)
		}
		// We can be sure that childFD is safe to use. It could be changed
		// by user after Mkdirat() and before Openat(), however:
		// - it could not be changed to symlink - we use nofollowFlags
		// - it could be changed to a file (or device, pipe, socket, ...)
		//   but either subsequent Mkdirat() fails or we mount this file
		//   to user's container. Security is no violated in both cases
		//   and user either gets error or the file that it can already access.

		if err = syscall.Close(parentFD); err != nil {
			glog.V(4).Infof("Closing FD %v failed for safemkdir(%v): %v", parentFD, pathname, err)
		}
		parentFD = childFD
		childFD = -1
	}

	// Everything was created. mkdirat(..., perm) above was affected by current
	// umask and we must apply the right permissions to the last directory
	// (that's the one that will be available to the container as subpath)
	// so user can read/write it. This is the behavior of previous code.
	// TODO: chmod all created directories, not just the last one.
	// parentFD is the last created directory.
	if err = syscall.Fchmod(parentFD, uint32(perm)&uint32(os.ModePerm)); err != nil {
		return fmt.Errorf("chmod %q failed: %s", currentPath, err)
	}
	return nil
}

// findExistingPrefix finds prefix of pathname that exists. In addition, it
// returns list of remaining directories that don't exist yet.
func findExistingPrefix(base, pathname string) (string, []string, error) {
	rel, err := filepath.Rel(base, pathname)
	if err != nil {
		return base, nil, err
	}
	dirs := strings.Split(rel, string(filepath.Separator))

	// Do OpenAt in a loop to find the first non-existing dir. Resolve symlinks.
	// This should be faster than looping through all dirs and calling os.Stat()
	// on each of them, as the symlinks are resolved only once with OpenAt().
	currentPath := base
	fd, err := syscall.Open(currentPath, syscall.O_RDONLY, 0)
	if err != nil {
		return pathname, nil, fmt.Errorf("error opening %s: %s", currentPath, err)
	}
	defer func() {
		if err = syscall.Close(fd); err != nil {
			glog.V(4).Infof("Closing FD %v failed for findExistingPrefix(%v): %v", fd, pathname, err)
		}
	}()
	for i, dir := range dirs {
		childFD, err := syscall.Openat(fd, dir, syscall.O_RDONLY, 0)
		if err != nil {
			if os.IsNotExist(err) {
				return currentPath, dirs[i:], nil
			}
			return base, nil, err
		}
		if err = syscall.Close(fd); err != nil {
			glog.V(4).Infof("Closing FD %v failed for findExistingPrefix(%v): %v", fd, pathname, err)
		}
		fd = childFD
		currentPath = filepath.Join(currentPath, dir)
	}
	return pathname, []string{}, nil
}

// This implementation is shared between Linux and NsEnterMounter
// Open path and return its fd.
// Symlinks are disallowed (pathname must already resolve symlinks),
// and the path must be within the base directory.
func doSafeOpen(pathname string, base string) (int, error) {
	// Calculate segments to follow
	subpath, err := filepath.Rel(base, pathname)
	if err != nil {
		return -1, err
	}
	segments := strings.Split(subpath, string(filepath.Separator))

	// Assumption: base is the only directory that we have under control.
	// Base dir is not allowed to be a symlink.
	parentFD, err := syscall.Open(base, nofollowFlags, 0)
	if err != nil {
		return -1, fmt.Errorf("cannot open directory %s: %s", base, err)
	}
	defer func() {
		if parentFD != -1 {
			if err = syscall.Close(parentFD); err != nil {
				glog.V(4).Infof("Closing FD %v failed for safeopen(%v): %v", parentFD, pathname, err)
			}
		}
	}()

	childFD := -1
	defer func() {
		if childFD != -1 {
			if err = syscall.Close(childFD); err != nil {
				glog.V(4).Infof("Closing FD %v failed for safeopen(%v): %v", childFD, pathname, err)
			}
		}
	}()

	currentPath := base

	// Follow the segments one by one using openat() to make
	// sure the user cannot change already existing directories into symlinks.
	for _, seg := range segments {
		currentPath = filepath.Join(currentPath, seg)
		if !pathWithinBase(currentPath, base) {
			return -1, fmt.Errorf("path %s is outside of allowed base %s", currentPath, base)
		}

		glog.V(5).Infof("Opening path %s", currentPath)
		childFD, err = syscall.Openat(parentFD, seg, nofollowFlags, 0)
		if err != nil {
			return -1, fmt.Errorf("cannot open %s: %s", currentPath, err)
		}

		// Close parentFD
		if err = syscall.Close(parentFD); err != nil {
			return -1, fmt.Errorf("closing fd for %q failed: %v", filepath.Dir(currentPath), err)
		}
		// Set child to new parent
		parentFD = childFD
		childFD = -1
	}

	// We made it to the end, return this fd, don't close it
	finalFD := parentFD
	parentFD = -1

	return finalFD, nil
}

type mountInfo struct {
	mountPoint string
	// list of "optional parameters", mount propagation is one of them
	optional []string
}

// parseMountInfo parses /proc/xxx/mountinfo.
func parseMountInfo(filename string) ([]mountInfo, error) {
	content, err := utilio.ConsistentRead(filename, maxListTries)
	if err != nil {
		return []mountInfo{}, err
	}
	contentStr := string(content)
	infos := []mountInfo{}

	for _, line := range strings.Split(contentStr, "\n") {
		if line == "" {
			// the last split() item is empty string following the last \n
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 7 {
			return nil, fmt.Errorf("wrong number of fields in (expected %d, got %d): %s", 8, len(fields), line)
		}
		info := mountInfo{
			mountPoint: fields[4],
			optional:   []string{},
		}
		for i := 6; i < len(fields) && fields[i] != "-"; i++ {
			info.optional = append(info.optional, fields[i])
		}
		infos = append(infos, info)
	}
	return infos, nil
}
