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
	if bind, _ := isBind(options); bind {
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

		cmdLine := fmt.Sprintf(`$User = "%s";$PWord = ConvertTo-SecureString -String "%s" -AsPlainText -Force;`+
			`$Credential = New-Object -TypeName System.Management.Automation.PSCredential -ArgumentList $User, $PWord`,
			options[0], options[1])

		driverLetter, err := getAvailableDriveLetter()
		if err != nil {
			return err
		}
		bindSource = driverLetter + ":"
		cmdLine += fmt.Sprintf(";New-SmbGlobalMapping -LocalPath %s -RemotePath %s -Credential $Credential", bindSource, source)

		if output, err := exec.Command("powershell", "/c", cmdLine).CombinedOutput(); err != nil {
			// we don't return error here, even though New-SmbGlobalMapping failed, we still make it successful,
			// will return error when Windows 2016 RS3 is ready on azure
			glog.Errorf("azureMount: SmbGlobalMapping failed: %v, only SMB mount is supported now, output: %q", err, string(output))
			return os.MkdirAll(target, 0755)
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

// GetMountRefs finds all other references to the device(drive) referenced
// by mountPath; returns a list of paths.
func GetMountRefs(mounter Interface, mountPath string) ([]string, error) {
	refs, err := getAllParentLinks(normalizeWindowsPath(mountPath))
	if err != nil {
		return nil, err
	}
	return refs, nil
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
		return false, nil
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
	refs, err := GetMountRefs(mounter, mountPath)
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
	var pathType FileType
	info, err := os.Stat(pathname)
	if os.IsNotExist(err) {
		return pathType, fmt.Errorf("path %q does not exist", pathname)
	}
	// err in call to os.Stat
	if err != nil {
		return pathType, err
	}

	mode := info.Sys().(*syscall.Win32FileAttributeData).FileAttributes
	switch mode & syscall.S_IFMT {
	case syscall.S_IFSOCK:
		return FileTypeSocket, nil
	case syscall.S_IFBLK:
		return FileTypeBlockDev, nil
	case syscall.S_IFCHR:
		return FileTypeCharDev, nil
	case syscall.S_IFDIR:
		return FileTypeDirectory, nil
	case syscall.S_IFREG:
		return FileTypeFile, nil
	}

	return pathType, fmt.Errorf("only recognise file, directory, socket, block device and character device")
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
func (mounter *Mounter) ExistsPath(pathname string) bool {
	_, err := os.Stat(pathname)
	if err != nil {
		return false
	}
	return true
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	// Try to mount the disk
	glog.V(4).Infof("Attempting to formatAndMount disk: %s %s %s", fstype, source, target)

	if err := ValidateDiskNumber(source); err != nil {
		glog.Errorf("azureMount: formatAndMount failed, err: %v\n", err)
		return err
	}

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

func getAvailableDriveLetter() (string, error) {
	cmd := "$used = Get-PSDrive | Select-Object -Expand Name | Where-Object { $_.Length -eq 1 }"
	cmd += ";$drive = 67..90 | ForEach-Object { [string][char]$_ } | Where-Object { $used -notcontains $_ } | Select-Object -First 1;$drive"
	output, err := exec.Command("powershell", "/c", cmd).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("getAvailableDriveLetter failed: %v, output: %q", err, string(output))
	}

	if len(output) == 0 {
		return "", fmt.Errorf("azureMount: there is no available drive letter now")
	}
	return string(output)[:1], nil
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
