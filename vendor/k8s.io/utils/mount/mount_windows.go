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
	utilexec "k8s.io/utils/exec"
	"k8s.io/utils/keymutex"
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
			klog.Warningf("mount options(%q) command number(%d) less than 2, source:%q, target:%q, skip mounting",
				sanitizedOptionsForLogging, len(allOptions), source, target)
			return nil
		}

		// currently only cifs mount is supported
		if strings.ToLower(fstype) != "cifs" {
			return fmt.Errorf("only cifs mount is supported now, fstype: %q, mounting source (%q), target (%q), with options (%q)", fstype, source, target, sanitizedOptionsForLogging)
		}

		// lock smb mount for the same source
		getSMBMountMutex.LockKey(source)
		defer getSMBMountMutex.UnlockKey(source)

		if output, err := newSMBMapping(allOptions[0], allOptions[1], source); err != nil {
			if isSMBMappingExist(source) {
				klog.V(2).Infof("SMB Mapping(%s) already exists, now begin to remove and remount", source)
				if output, err := removeSMBMapping(source); err != nil {
					return fmt.Errorf("Remove-SmbGlobalMapping failed: %v, output: %q", err, output)
				}
				if output, err := newSMBMapping(allOptions[0], allOptions[1], source); err != nil {
					return fmt.Errorf("New-SmbGlobalMapping remount failed: %v, output: %q", err, output)
				}
			} else {
				return fmt.Errorf("New-SmbGlobalMapping failed: %v, output: %q", err, output)
			}
		}
	}

	output, err := exec.Command("cmd", "/c", "mklink", "/D", target, bindSource).CombinedOutput()
	if err != nil {
		klog.Errorf("mklink failed: %v, source(%q) target(%q) output: %q", err, bindSource, target, string(output))
		return err
	}
	klog.V(2).Infof("mklink source(%q) on target(%q) successfully, output: %q", bindSource, target, string(output))

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
		`;New-SmbGlobalMapping -RemotePath $Env:smbremotepath -Credential $Credential`
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

// remove SMB mapping
func removeSMBMapping(remotepath string) (string, error) {
	cmd := exec.Command("powershell", "/c", `Remove-SmbGlobalMapping -RemotePath $Env:smbremotepath -Force`)
	cmd.Env = append(os.Environ(), fmt.Sprintf("smbremotepath=%s", remotepath))
	output, err := cmd.CombinedOutput()
	return string(output), err
}

// Unmount unmounts the target.
func (mounter *Mounter) Unmount(target string) error {
	klog.V(4).Infof("azureMount: Unmount target (%q)", target)
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
	cmd := fmt.Sprintf("Get-Disk -Number %s | Where partitionstyle -eq 'raw' | Initialize-Disk -PartitionStyle MBR -PassThru"+
		" | New-Partition -AssignDriveLetter -UseMaximumSize | Format-Volume -FileSystem %s -Confirm:$false", source, fstype)
	if output, err := mounter.Exec.Command("powershell", "/c", cmd).CombinedOutput(); err != nil {
		return fmt.Errorf("diskMount: format disk failed, error: %v, output: %q", err, string(output))
	}
	klog.V(4).Infof("diskMount: Disk successfully formatted, disk: %q, fstype: %q", source, fstype)

	driveLetter, err := getDriveLetterByDiskNumber(source, mounter.Exec)
	if err != nil {
		return err
	}
	driverPath := driveLetter + ":"
	target = NormalizeWindowsPath(target)
	output, err := mounter.Exec.Command("cmd", "/c", "mklink", "/D", target, driverPath).CombinedOutput()
	if err != nil {
		klog.Errorf("mklink(%s, %s) failed: %v, output: %q", target, driverPath, err, string(output))
		return err
	}
	klog.V(2).Infof("formatAndMount disk(%s) fstype(%s) on(%s) with output(%s) successfully", driverPath, fstype, target, string(output))
	return nil
}

// Get drive letter according to windows disk number
func getDriveLetterByDiskNumber(diskNum string, exec utilexec.Interface) (string, error) {
	cmd := fmt.Sprintf("(Get-Partition -DiskNumber %s).DriveLetter", diskNum)
	output, err := exec.Command("powershell", "/c", cmd).CombinedOutput()
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
