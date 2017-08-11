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

	"github.com/golang/glog"
)

type Mounter struct {
	mounterPath string
}

func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	if !strings.HasPrefix(target, "c:") && !strings.HasPrefix(target, "C:") {
		target = "c:" + target
	}

	if source == "tmpfs" {
		glog.Infof("azureMount: mounting source (%q), target (%q)\n, with options (%q)", source, target, options)
		os.MkdirAll(target, 0755)
		return nil
	}

	parentDir := filepath.Dir(target)
	err := os.MkdirAll(parentDir, 0755)
	if err != nil {
		return fmt.Errorf("mkdir(%q) failed: %v", parentDir, err)
	}

	if len(options) != 1 {
		glog.Warningf("azureMount: mount options(%q) command(%n) not equal to 1, skip mounting.", options, len(options))
		return nil
	}
	cmd := options[0]

	driverLetter, err := getAvailableDriveLetter()
	if err != nil {
		return err
	}
	driverPath := driverLetter + ":"
	cmd += fmt.Sprintf(";New-SmbGlobalMapping -LocalPath %s -RemotePath %s -Credential $Credential", driverPath, source)

	_, err = exec.Command("powershell", "/c", cmd).CombinedOutput()
	if err != nil {
		// we don't return error here, even though New-SmbGlobalMapping failed, we still make it successful,
		// will return error when Windows 2016 RS3 is ready on azure
		glog.Errorf("azureMount: SmbGlobalMapping failed: %v", err)
		os.MkdirAll(target, 0755)
		return nil
	}

	_, err = exec.Command("cmd", "/c", "mklink", "/D", target, driverPath).CombinedOutput()
	if err != nil {
		return fmt.Errorf("mklink failed: %v", err)
	}

	return nil
}

func (mounter *Mounter) Unmount(target string) error {
	glog.Infof("azureMount: Unmount target (%q)", target)
	output, err := exec.Command("cmd", "/c", "rmdir", target).CombinedOutput()
	if err != nil {
		return fmt.Errorf("Unmount failed: %v", err)
	}
	glog.Infof("azureMount: Unmount succeeded, output: %q", output)
	return nil
}

func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

func (mounter *Mounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return mp.Path == dir
}

func (mounter *Mounter) IsNotMountPoint(dir string) (bool, error) {
	return IsNotMountPoint(mounter, dir)
}

func (mounter *Mounter) IsLikelyNotMountPoint(file string) (bool, error) {
	return true, nil
}

func (mounter *Mounter) GetDeviceNameFromMount(mountPath, pluginDir string) (string, error) {
	return getDeviceNameFromMount(mounter, mountPath, pluginDir)
}

func (mounter *Mounter) DeviceOpened(pathname string) (bool, error) {
	return false, nil
}

func (mounter *Mounter) PathIsDevice(pathname string) (bool, error) {
	return false, nil
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	return nil
}

func getAvailableDriveLetter() (string, error) {
	cmd := "$used = Get-PSDrive | Select-Object -Expand Name | Where-Object { $_.Length -eq 1 }"
	cmd += ";$drive = 67..90 | ForEach-Object { [string][char]$_ } | Where-Object { $used -notcontains $_ } | Select-Object -First 1;$drive"
	output, err := exec.Command("powershell", "/c", cmd).CombinedOutput()
	if err != nil || len(output) == 0 {
		return "", fmt.Errorf("getAvailableDriveLetter failed: %v", err)
	}
	return string(output)[:1], nil
}
