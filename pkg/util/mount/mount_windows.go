// +build windows

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
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/golang/glog"

	utilfile "k8s.io/kubernetes/pkg/util/file"
)

// Mounter provides the default implementation of mount.Interface
// for the windows platform.  This implementation assumes that the
// kubelet is running in the host's root mount namespace.
type Mounter struct {
	mounterPath string
}

// New returns a mount.Interface for the current system.
// It provides options to override the default mounter behavior.
// mounterPath allows using an alternative to `/bin/mount` for mounting.
func New(mounterPath string) Interface {
	return &Mounter{
		mounterPath: mounterPath,
	}
}

// Mount : mounts source to target as NTFS with given options.
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	target = normalizeWindowsPath(target)

	if source == "tmpfs" {
		glog.V(3).Infof("azureMount: mounting source (%q), target (%q), with options (%q)", source, target, options)
		return os.MkdirAll(target, 0755)
	}

	parentDir := filepath.Dir(target)
	if err := os.MkdirAll(parentDir, 0755); err != nil {
		return err
	}

	glog.V(4).Infof("azureMount: mount options(%q) source:%q, target:%q, fstype:%q, begin to mount",
		options, source, target, fstype)
	bindSource := ""

	// tell it's going to mount azure disk or azure file according to options
	if bind, _, _ := isBind(options); bind {
		// mount azure disk
		bindSource = normalizeWindowsPath(source)
	} else {
		if len(options) < 2 {
			glog.Warningf("azureMount: mount options(%q) command number(%d) less than 2, source:%q, target:%q, skip mounting",
				options, len(options), source, target)
			return nil
		}

		// currently only cifs mount is supported
		if strings.ToLower(fstype) != "cifs" {
			return fmt.Errorf("azureMount: only cifs mount is supported now, fstype: %q, mounting source (%q), target (%q), with options (%q)", fstype, source, target, options)
		}

		bindSource = source

		// use PowerShell Environment Variables to store user input string to prevent command line injection
		// https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-5.1
		cmdLine := fmt.Sprintf(`$PWord = ConvertTo-SecureString -String $Env:smbpassword -AsPlainText -Force` +
			`;$Credential = New-Object -TypeName System.Management.Automation.PSCredential -ArgumentList $Env:smbuser, $PWord` +
			`;New-SmbGlobalMapping -RemotePath $Env:smbremotepath -Credential $Credential`)

		cmd := exec.Command("powershell", "/c", cmdLine)
		cmd.Env = append(os.Environ(),
			fmt.Sprintf("smbuser=%s", options[0]),
			fmt.Sprintf("smbpassword=%s", options[1]),
			fmt.Sprintf("smbremotepath=%s", source))
		if output, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("azureMount: SmbGlobalMapping failed: %v, only SMB mount is supported now, output: %q", err, string(output))
		}
	}

	if output, err := exec.Command("cmd", "/c", "mklink", "/D", target, bindSource).CombinedOutput(); err != nil {
		glog.Errorf("mklink failed: %v, source(%q) target(%q) output: %q", err, bindSource, target, string(output))
		return err
	}

	return nil
}

// Unmount unmounts the target.
func (mounter *Mounter) Unmount(target string) error {
	glog.V(4).Infof("azureMount: Unmount target (%q)", target)
	target = normalizeWindowsPath(target)
	if output, err := exec.Command("cmd", "/c", "rmdir", target).CombinedOutput(); err != nil {
		glog.Errorf("rmdir failed: %v, output: %q", err, string(output))
		return err
	}
	return nil
}

// List returns a list of all mounted filesystems. todo
func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

// IsMountPointMatch determines if the mountpoint matches the dir
func (mounter *Mounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return mp.Path == dir
}

// IsNotMountPoint determines if a directory is a mountpoint.
func (mounter *Mounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(mounter, dir)
}

// IsLikelyNotMountPoint determines if a directory is not a mountpoint.
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	stat, err := os.Lstat(file)
	if err != nil {
		return true, err
	}
	// If current file is a symlink, then it is a mountpoint.
	if stat.Mode()&os.ModeSymlink != 0 {
		target, err := os.Readlink(file)
		if err != nil {
			return true, fmt.Errorf("readlink error: %v", err)
		}
		exists, err := mounter.ExistsPath(target)
		if err != nil {
			return true, err
		}
		return !exists, nil
	}

	return true, nil
}

// GetDeviceNameFromMount given a mnt point, find the device
func (mounter *Mounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginDir)
}

// getDeviceNameFromMount find the device(drive) name in which
// the mount path reference should match the given plugin directory. In case no mount path reference
// matches, returns the volume name taken from its given mountPath
func getDeviceNameFromMount(mounter Interface, mountPath, pluginDir string) (string, error) {
	refs, err := mounter.GetMountRefs(mountPath)
	if err != nil {
		glog.V(4).Infof("GetMountRefs failed for mount path %q: %v", mountPath, err)
		return "", err
	}
	if len(refs) == 0 {
		return "", fmt.Errorf("directory %s is not mounted", mountPath)
	}
	basemountPath := normalizeWindowsPath(path.Join(pluginDir, MountsInGlobalPDPath))
	for _, ref := range refs {
		if strings.Contains(ref, basemountPath) {
			volumeID, err := filepath.Rel(normalizeWindowsPath(basemountPath), ref)
			if err != nil {
				glog.Errorf("Failed to get volume id from mount %s - %v", mountPath, err)
				return "", err
			}
			return volumeID, nil
		}
	}

	return path.Base(mountPath), nil
}

// DeviceOpened determines if the device is in use elsewhere
func (mounter *Mounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

// PathIsDevice determines if a path is a device.
func (mounter *Mounter) PathIsDevice(pathname string) (bool, error) {
	return false, nil
}

// MakeRShared checks that given path is on a mount with 'rshared' mount
// propagation. Empty implementation here.
func (mounter *Mounter) MakeRShared(path string) error {
	return nil
}

// GetFileType checks for sockets/block/character devices
func (mounter *Mounter) GetFileType(pathname string) (FileType, error) {
	return getFileType(pathname)
}

// MakeFile creates a new directory
func (mounter *Mounter) MakeDir(pathname string) error {
	err := os.MkdirAll(pathname, os.FileMode(0755))
	if err != nil {
		if !os.IsExist(err) {
			return err
		}
	}
	return nil
}

// MakeFile creates an empty file
func (mounter *Mounter) MakeFile(pathname string) error {
	f, err := os.OpenFile(pathname, os.O_CREATE, os.FileMode(0644))
	defer f.Close()
	if err != nil {
		if !os.IsExist(err) {
			return err
		}
	}
	return nil
}

// ExistsPath checks whether the path exists
func (mounter *Mounter) ExistsPath(pathname string) (bool, error) {
	return utilfile.FileExists(pathname)
}

// EvalHostSymlinks returns the path name after evaluating symlinks
func (mounter *Mounter) EvalHostSymlinks(pathname string) (string, error) {
	return filepath.EvalSymlinks(pathname)
}

// check whether hostPath is within volume path
// this func will lock all intermediate subpath directories, need to close handle outside of this func after container started
func lockAndCheckSubPath(volumePath, hostPath string) ([]uintptr, error) {
	if len(volumePath) == 0 || len(hostPath) == 0 {
		return []uintptr{}, nil
	}

	finalSubPath, err := filepath.EvalSymlinks(hostPath)
	if err != nil {
		return []uintptr{}, fmt.Errorf("cannot read link %s: %s", hostPath, err)
	}
	finalVolumePath, err := filepath.EvalSymlinks(volumePath)
	if err != nil {
		return []uintptr{}, fmt.Errorf("cannot read link %s: %s", volumePath, err)
	}

	return lockAndCheckSubPathWithoutSymlink(finalVolumePath, finalSubPath)
}

// lock all intermediate subPath directories and check they are all within volumePath
// volumePath & subPath should not contain any symlink, otherwise it will return error
func lockAndCheckSubPathWithoutSymlink(volumePath, subPath string) ([]uintptr, error) {
	if len(volumePath) == 0 || len(subPath) == 0 {
		return []uintptr{}, nil
	}

	// get relative path to volumePath
	relSubPath, err := filepath.Rel(volumePath, subPath)
	if err != nil {
		return []uintptr{}, fmt.Errorf("Rel(%s, %s) error: %v", volumePath, subPath, err)
	}
	if startsWithBackstep(relSubPath) {
		return []uintptr{}, fmt.Errorf("SubPath %q not within volume path %q", subPath, volumePath)
	}

	if relSubPath == "." {
		// volumePath and subPath are equal
		return []uintptr{}, nil
	}

	fileHandles := []uintptr{}
	var errorResult error

	currentFullPath := volumePath
	dirs := strings.Split(relSubPath, string(os.PathSeparator))
	for _, dir := range dirs {
		// lock intermediate subPath directory first
		currentFullPath = filepath.Join(currentFullPath, dir)
		handle, err := lockPath(currentFullPath)
		if err != nil {
			errorResult = fmt.Errorf("cannot lock path %s: %s", currentFullPath, err)
			break
		}
		fileHandles = append(fileHandles, handle)

		// make sure intermediate subPath directory does not contain symlink any more
		stat, err := os.Lstat(currentFullPath)
		if err != nil {
			errorResult = fmt.Errorf("Lstat(%q) error: %v", currentFullPath, err)
			break
		}
		if stat.Mode()&os.ModeSymlink != 0 {
			errorResult = fmt.Errorf("subpath %q is an unexpected symlink after EvalSymlinks", currentFullPath)
			break
		}

		if !PathWithinBase(currentFullPath, volumePath) {
			errorResult = fmt.Errorf("SubPath %q not within volume path %q", currentFullPath, volumePath)
			break
		}
	}

	return fileHandles, errorResult
}

// unlockPath unlock directories
func unlockPath(fileHandles []uintptr) {
	if fileHandles != nil {
		for _, handle := range fileHandles {
			syscall.CloseHandle(syscall.Handle(handle))
		}
	}
}

// lockPath locks a directory or symlink, return handle, exec "syscall.CloseHandle(handle)" to unlock the path
func lockPath(path string) (uintptr, error) {
	if len(path) == 0 {
		return uintptr(syscall.InvalidHandle), syscall.ERROR_FILE_NOT_FOUND
	}
	pathp, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return uintptr(syscall.InvalidHandle), err
	}
	access := uint32(syscall.GENERIC_READ)
	sharemode := uint32(syscall.FILE_SHARE_READ)
	createmode := uint32(syscall.OPEN_EXISTING)
	flags := uint32(syscall.FILE_FLAG_BACKUP_SEMANTICS | syscall.FILE_FLAG_OPEN_REPARSE_POINT)
	fd, err := syscall.CreateFile(pathp, access, sharemode, nil, createmode, flags, 0)
	return uintptr(fd), err
}

// Lock all directories in subPath and check they're not symlinks.
func (mounter *Mounter) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	handles, err := lockAndCheckSubPath(subPath.VolumePath, subPath.Path)

	// Unlock the directories when the container starts
	cleanupAction = func() {
		unlockPath(handles)
	}
	return subPath.Path, cleanupAction, err
}

// No bind-mounts for subpaths are necessary on Windows
func (mounter *Mounter) CleanSubPaths(podDir string, volumeName string) error {
	return nil
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	// Try to mount the disk
	glog.V(4).Infof("Attempting to formatAndMount disk: %s %s %s", fstype, source, target)

	if err := ValidateDiskNumber(source); err != nil {
		glog.Errorf("diskMount: formatAndMount failed, err: %v", err)
		return err
	}

	if len(fstype) == 0 {
		// Use 'NTFS' as the default
		fstype = "NTFS"
	}

	// format disk if it is unformatted(raw)
	cmd := fmt.Sprintf("Get-Disk -Number %s | Where partitionstyle -eq 'raw' | Initialize-Disk -PartitionStyle MBR -PassThru"+
		" | New-Partition -AssignDriveLetter -UseMaximumSize | Format-Volume -FileSystem %s -Confirm:$false", source, fstype)
	if output, err := mounter.Exec.Run("powershell", "/c", cmd); err != nil {
		return fmt.Errorf("diskMount: format disk failed, error: %v, output: %q", err, string(output))
	}
	glog.V(4).Infof("diskMount: Disk successfully formatted, disk: %q, fstype: %q", source, fstype)

	driveLetter, err := getDriveLetterByDiskNumber(source, mounter.Exec)
	if err != nil {
		return err
	}
	driverPath := driveLetter + ":"
	target = normalizeWindowsPath(target)
	glog.V(4).Infof("Attempting to formatAndMount disk: %s %s %s", fstype, driverPath, target)
	if output, err := mounter.Exec.Run("cmd", "/c", "mklink", "/D", target, driverPath); err != nil {
		glog.Errorf("mklink failed: %v, output: %q", err, string(output))
		return err
	}
	return nil
}

func normalizeWindowsPath(path string) string {
	normalizedPath := strings.Replace(path, "/", "\\", -1)
	if strings.HasPrefix(normalizedPath, "\\") {
		normalizedPath = "c:" + normalizedPath
	}
	return normalizedPath
}

// ValidateDiskNumber : disk number should be a number in [0, 99]
func ValidateDiskNumber(disk string) error {
	diskNum, err := strconv.Atoi(disk)
	if err != nil {
		return fmt.Errorf("wrong disk number format: %q, err:%v", disk, err)
	}

	if diskNum < 0 || diskNum > 99 {
		return fmt.Errorf("disk number out of range: %q", disk)
	}

	return nil
}

// Get drive letter according to windows disk number
func getDriveLetterByDiskNumber(diskNum string, exec Exec) (string, error) {
	cmd := fmt.Sprintf("(Get-Partition -DiskNumber %s).DriveLetter", diskNum)
	output, err := exec.Run("powershell", "/c", cmd)
	if err != nil {
		return "", fmt.Errorf("azureMount: Get Drive Letter failed: %v, output: %q", err, string(output))
	}
	if len(string(output)) < 1 {
		return "", fmt.Errorf("azureMount: Get Drive Letter failed, output is empty")
	}
	return string(output)[:1], nil
}

// getAllParentLinks walks all symbolic links and return all the parent targets recursively
func getAllParentLinks(path string) ([]string, error) {
	const maxIter = 255
	links := []string{}
	for {
		links = append(links, path)
		if len(links) > maxIter {
			return links, fmt.Errorf("unexpected length of parent links: %v", links)
		}

		fi, err := os.Lstat(path)
		if err != nil {
			return links, fmt.Errorf("Lstat: %v", err)
		}
		if fi.Mode()&os.ModeSymlink == 0 {
			break
		}

		path, err = os.Readlink(path)
		if err != nil {
			return links, fmt.Errorf("Readlink error: %v", err)
		}
	}

	return links, nil
}

// GetMountRefs : empty implementation here since there is no place to query all mount points on Windows
func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	if _, err := os.Stat(normalizeWindowsPath(pathname)); os.IsNotExist(err) {
		return []string{}, nil
	} else if err != nil {
		return nil, err
	}
	return []string{pathname}, nil
}

// Note that on windows, it always returns 0. We actually don't set FSGroup on
// windows platform, see SetVolumeOwnership implementation.
func (mounter *Mounter) GetFSGroup(pathname string) (int64, error) {
	return 0, nil
}

func (mounter *Mounter) GetSELinuxSupport(pathname string) (bool, error) {
	// Windows does not support SELinux.
	return false, nil
}

func (mounter *Mounter) GetMode(pathname string) (os.FileMode, error) {
	info, err := os.Stat(pathname)
	if err != nil {
		return 0, err
	}
	return info.Mode(), nil
}

// SafeMakeDir makes sure that the created directory does not escape given base directory mis-using symlinks.
func (mounter *Mounter) SafeMakeDir(subdir string, base string, perm os.FileMode) error {
	realBase, err := filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("error resolving symlinks in %s: %s", base, err)
	}

	realFullPath := filepath.Join(realBase, subdir)
	return doSafeMakeDir(realFullPath, realBase, perm)
}

func doSafeMakeDir(pathname string, base string, perm os.FileMode) error {
	glog.V(4).Infof("Creating directory %q within base %q", pathname, base)

	if !PathWithinBase(pathname, base) {
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
	if len(toCreate) == 0 {
		return nil
	}

	// Ensure the existing directory is inside allowed base
	fullExistingPath, err := filepath.EvalSymlinks(existingPath)
	if err != nil {
		return fmt.Errorf("error opening existing directory %s: %s", existingPath, err)
	}
	fullBasePath, err := filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("cannot read link %s: %s", base, err)
	}
	if !PathWithinBase(fullExistingPath, fullBasePath) {
		return fmt.Errorf("path %s is outside of allowed base %s", fullExistingPath, err)
	}

	// lock all intermediate directories from fullBasePath to fullExistingPath (top to bottom)
	fileHandles, err := lockAndCheckSubPathWithoutSymlink(fullBasePath, fullExistingPath)
	defer unlockPath(fileHandles)
	if err != nil {
		return err
	}

	glog.V(4).Infof("%q already exists, %q to create", fullExistingPath, filepath.Join(toCreate...))
	currentPath := fullExistingPath
	// create the directories one by one, making sure nobody can change
	// created directory into symlink by lock that directory immediately
	for _, dir := range toCreate {
		currentPath = filepath.Join(currentPath, dir)
		glog.V(4).Infof("Creating %s", dir)
		if err := os.Mkdir(currentPath, perm); err != nil {
			return fmt.Errorf("cannot create directory %s: %s", currentPath, err)
		}
		handle, err := lockPath(currentPath)
		if err != nil {
			return fmt.Errorf("cannot lock path %s: %s", currentPath, err)
		}
		defer syscall.CloseHandle(syscall.Handle(handle))
		// make sure newly created directory does not contain symlink after lock
		stat, err := os.Lstat(currentPath)
		if err != nil {
			return fmt.Errorf("Lstat(%q) error: %v", currentPath, err)
		}
		if stat.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("subpath %q is an unexpected symlink after Mkdir", currentPath)
		}
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

	if startsWithBackstep(rel) {
		return base, nil, fmt.Errorf("pathname(%s) is not within base(%s)", pathname, base)
	}

	if rel == "." {
		// base and pathname are equal
		return pathname, []string{}, nil
	}

	dirs := strings.Split(rel, string(filepath.Separator))

	parent := base
	currentPath := base
	for i, dir := range dirs {
		parent = currentPath
		currentPath = filepath.Join(parent, dir)
		if _, err := os.Lstat(currentPath); err != nil {
			if os.IsNotExist(err) {
				return parent, dirs[i:], nil
			}
			return base, nil, err
		}
	}

	return pathname, []string{}, nil
}
