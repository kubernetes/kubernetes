//go:build windows
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
	"path/filepath"
	"strings"

	"k8s.io/klog/v2"
	"k8s.io/utils/keymutex"
)

const (
	accessDenied string = "access is denied"
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

// acquire lock for smb mount
var getSMBMountMutex = keymutex.NewHashed(0)

// Mount : mounts source to target with given options.
// currently only supports cifs(smb), bind mount(for disk)
func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	return mounter.MountSensitive(source, target, fstype, options, nil /* sensitiveOptions */)
}

// MountSensitiveWithoutSystemd is the same as MountSensitive() but disable using ssytemd mount.
// Windows not supported systemd mount, this function degrades to MountSensitive().
func (mounter *Mounter) MountSensitiveWithoutSystemd(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	return mounter.MountSensitive(source, target, fstype, options, sensitiveOptions /* sensitiveOptions */)
}

// MountSensitiveWithoutSystemdWithMountFlags is the same as MountSensitiveWithoutSystemd with additional mount flags
// Windows not supported systemd mount, this function degrades to MountSensitive().
func (mounter *Mounter) MountSensitiveWithoutSystemdWithMountFlags(source string, target string, fstype string, options []string, sensitiveOptions []string, mountFlags []string) error {
	return mounter.MountSensitive(source, target, fstype, options, sensitiveOptions /* sensitiveOptions */)
}

// MountSensitive is the same as Mount() but this method allows
// sensitiveOptions to be passed in a separate parameter from the normal
// mount options and ensures the sensitiveOptions are never logged. This
// method should be used by callers that pass sensitive material (like
// passwords) as mount options.
func (mounter *Mounter) MountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	target = NormalizeWindowsPath(target)
	sanitizedOptionsForLogging := sanitizedOptionsForLogging(options, sensitiveOptions)

	if source == "tmpfs" {
		klog.V(3).Infof("mounting source (%q), target (%q), with options (%q)", source, target, sanitizedOptionsForLogging)
		return os.MkdirAll(target, 0755)
	}

	parentDir := filepath.Dir(target)
	if err := os.MkdirAll(parentDir, 0755); err != nil {
		return err
	}

	klog.V(4).Infof("mount options(%q) source:%q, target:%q, fstype:%q, begin to mount",
		sanitizedOptionsForLogging, source, target, fstype)
	bindSource := source

	if bind, _, _, _ := MakeBindOptsSensitive(options, sensitiveOptions); bind {
		bindSource = NormalizeWindowsPath(source)
	} else {
		allOptions := []string{}
		allOptions = append(allOptions, options...)
		allOptions = append(allOptions, sensitiveOptions...)
		if len(allOptions) < 2 {
			return fmt.Errorf("mount options(%q) should have at least 2 options, current number:%d, source:%q, target:%q",
				sanitizedOptionsForLogging, len(allOptions), source, target)
		}

		// currently only cifs mount is supported
		if strings.ToLower(fstype) != "cifs" {
			return fmt.Errorf("only cifs mount is supported now, fstype: %q, mounting source (%q), target (%q), with options (%q)", fstype, source, target, sanitizedOptionsForLogging)
		}

		// lock smb mount for the same source
		getSMBMountMutex.LockKey(source)
		defer getSMBMountMutex.UnlockKey(source)

		username := allOptions[0]
		password := allOptions[1]
		if output, err := newSMBMapping(username, password, source); err != nil {
			klog.Warningf("SMB Mapping(%s) returned with error(%v), output(%s)", source, err, string(output))
			if isSMBMappingExist(source) {
				valid, err := isValidPath(source)
				if !valid {
					if err == nil || isAccessDeniedError(err) {
						klog.V(2).Infof("SMB Mapping(%s) already exists while it's not valid, return error: %v, now begin to remove and remount", source, err)
						if output, err = removeSMBMapping(source); err != nil {
							return fmt.Errorf("Remove-SmbGlobalMapping failed: %v, output: %q", err, output)
						}
						if output, err := newSMBMapping(username, password, source); err != nil {
							return fmt.Errorf("New-SmbGlobalMapping(%s) failed: %v, output: %q", source, err, output)
						}
					}
				} else {
					klog.V(2).Infof("SMB Mapping(%s) already exists and is still valid, skip error(%v)", source, err)
				}
			} else {
				return fmt.Errorf("New-SmbGlobalMapping(%s) failed: %v, output: %q", source, err, output)
			}
		}
	}

	// There is an issue in golang where EvalSymlinks fails on Windows when passed a
	// UNC share root path without a trailing backslash.
	// Ex: \\SERVER\share will fail to resolve but \\SERVER\share\ will resolve
	// containerD on Windows calls EvalSymlinks so we'll add the backslash when making the symlink if it is missing.
	// https://github.com/golang/go/pull/42096 fixes this issue in golang but a fix will not be available until
	// golang v1.16
	mklinkSource := bindSource
	if !strings.HasSuffix(mklinkSource, "\\") {
		mklinkSource = mklinkSource + "\\"
	}

	output, err := exec.Command("cmd", "/c", "mklink", "/D", target, mklinkSource).CombinedOutput()
	if err != nil {
		klog.Errorf("mklink failed: %v, source(%q) target(%q) output: %q", err, mklinkSource, target, string(output))
		return err
	}
	klog.V(2).Infof("mklink source(%q) on target(%q) successfully, output: %q", mklinkSource, target, string(output))

	return nil
}

// do the SMB mount with username, password, remotepath
// return (output, error)
func newSMBMapping(username, password, remotepath string) (string, error) {
	if username == "" || password == "" || remotepath == "" {
		return "", fmt.Errorf("invalid parameter(username: %s, password: %s, remoteapth: %s)", username, sensitiveOptionsRemoved, remotepath)
	}

	// use PowerShell Environment Variables to store user input string to prevent command line injection
	// https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables?view=powershell-5.1
	cmdLine := `$PWord = ConvertTo-SecureString -String $Env:smbpassword -AsPlainText -Force` +
		`;$Credential = New-Object -TypeName System.Management.Automation.PSCredential -ArgumentList $Env:smbuser, $PWord` +
		`;New-SmbGlobalMapping -RemotePath $Env:smbremotepath -Credential $Credential -RequirePrivacy $true`
	cmd := exec.Command("powershell", "/c", cmdLine)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("smbuser=%s", username),
		fmt.Sprintf("smbpassword=%s", password),
		fmt.Sprintf("smbremotepath=%s", remotepath))

	output, err := cmd.CombinedOutput()
	return string(output), err
}

// check whether remotepath is already mounted
func isSMBMappingExist(remotepath string) bool {
	cmd := exec.Command("powershell", "/c", `Get-SmbGlobalMapping -RemotePath $Env:smbremotepath`)
	cmd.Env = append(os.Environ(), fmt.Sprintf("smbremotepath=%s", remotepath))
	_, err := cmd.CombinedOutput()
	return err == nil
}

// check whether remotepath is valid
// return (true, nil) if remotepath is valid
func isValidPath(remotepath string) (bool, error) {
	cmd := exec.Command("powershell", "/c", `Test-Path $Env:remoteapth`)
	cmd.Env = append(os.Environ(), fmt.Sprintf("remoteapth=%s", remotepath))
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false, fmt.Errorf("returned output: %s, error: %v", string(output), err)
	}

	return strings.HasPrefix(strings.ToLower(string(output)), "true"), nil
}

func isAccessDeniedError(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), accessDenied)
}

// remove SMB mapping
func removeSMBMapping(remotepath string) (string, error) {
	cmd := exec.Command("powershell", "/c", `Remove-SmbGlobalMapping -RemotePath $Env:smbremotepath -Force`)
	cmd.Env = append(os.Environ(), fmt.Sprintf("smbremotepath=%s", remotepath))
	output, err := cmd.CombinedOutput()
	return string(output), err
}

// Unmount unmounts the target.
func (mounter *Mounter) Unmount(target string) error {
	klog.V(4).Infof("Unmount target (%q)", target)
	target = NormalizeWindowsPath(target)
	if output, err := exec.Command("cmd", "/c", "rmdir", target).CombinedOutput(); err != nil {
		klog.Errorf("rmdir failed: %v, output: %q", err, string(output))
		return err
	}
	return nil
}

// List returns a list of all mounted filesystems. todo
func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

// IsLikelyNotMountPoint determines if a directory is not a mountpoint.
func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	stat, err := os.Lstat(file)
	if err != nil {
		return true, err
	}

	if stat.Mode()&os.ModeSymlink != 0 {
		return false, err
	}
	return true, nil
}

// canSafelySkipMountPointCheck always returns false on Windows
func (mounter *Mounter) canSafelySkipMountPointCheck() bool {
	return false
}

// IsMountPoint: determines if a directory is a mountpoint.
func (mounter *Mounter) IsMountPoint(file string) (bool, error) {
	isNotMnt, err := mounter.IsLikelyNotMountPoint(file)
	if err != nil {
		return false, err
	}
	return !isNotMnt, nil
}

// GetMountRefs : empty implementation here since there is no place to query all mount points on Windows
func (mounter *Mounter) GetMountRefs(pathname string) ([]string, error) {
	windowsPath := NormalizeWindowsPath(pathname)
	pathExists, pathErr := PathExists(windowsPath)
	if !pathExists {
		return []string{}, nil
	} else if IsCorruptedMnt(pathErr) {
		klog.Warningf("GetMountRefs found corrupted mount at %s, treating as unmounted path", windowsPath)
		return []string{}, nil
	} else if pathErr != nil {
		return nil, fmt.Errorf("error checking path %s: %v", windowsPath, pathErr)
	}
	return []string{pathname}, nil
}

func (mounter *SafeFormatAndMount) formatAndMountSensitive(source string, target string, fstype string, options []string, sensitiveOptions []string) error {
	// Try to mount the disk
	klog.V(4).Infof("Attempting to formatAndMount disk: %s %s %s", fstype, source, target)

	if err := ValidateDiskNumber(source); err != nil {
		klog.Errorf("diskMount: formatAndMount failed, err: %v", err)
		return err
	}

	if len(fstype) == 0 {
		// Use 'NTFS' as the default
		fstype = "NTFS"
	}

	// format disk if it is unformatted(raw)
	cmd := fmt.Sprintf("Get-Disk -Number %s | Where partitionstyle -eq 'raw' | Initialize-Disk -PartitionStyle GPT -PassThru"+
		" | New-Partition -UseMaximumSize | Format-Volume -FileSystem %s -Confirm:$false", source, fstype)
	if output, err := mounter.Exec.Command("powershell", "/c", cmd).CombinedOutput(); err != nil {
		return fmt.Errorf("diskMount: format disk failed, error: %v, output: %q", err, string(output))
	}
	klog.V(4).Infof("diskMount: Disk successfully formatted, disk: %q, fstype: %q", source, fstype)

	volumeIds, err := listVolumesOnDisk(source)
	if err != nil {
		return err
	}
	driverPath := volumeIds[0]
	target = NormalizeWindowsPath(target)
	output, err := mounter.Exec.Command("cmd", "/c", "mklink", "/D", target, driverPath).CombinedOutput()
	if err != nil {
		klog.Errorf("mklink(%s, %s) failed: %v, output: %q", target, driverPath, err, string(output))
		return err
	}
	klog.V(2).Infof("formatAndMount disk(%s) fstype(%s) on(%s) with output(%s) successfully", driverPath, fstype, target, string(output))
	return nil
}

// ListVolumesOnDisk - returns back list of volumes(volumeIDs) in the disk (requested in diskID).
func listVolumesOnDisk(diskID string) (volumeIDs []string, err error) {
	cmd := fmt.Sprintf("(Get-Disk -DeviceId %s | Get-Partition | Get-Volume).UniqueId", diskID)
	output, err := exec.Command("powershell", "/c", cmd).CombinedOutput()
	klog.V(4).Infof("listVolumesOnDisk id from %s: %s", diskID, string(output))
	if err != nil {
		return []string{}, fmt.Errorf("error list volumes on disk. cmd: %s, output: %s, error: %v", cmd, string(output), err)
	}

	volumeIds := strings.Split(strings.TrimSpace(string(output)), "\r\n")
	return volumeIds, nil
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
