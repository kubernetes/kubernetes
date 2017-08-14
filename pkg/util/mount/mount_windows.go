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
	"path/filepath"
	"strings"

	"k8s.io/utils/exec"

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
		glog.Infof("windowsMount: mounting source (%q), target (%q)\n, with options (%q)", source, target, options)
		return os.MkdirAll(target, 0755)
	}

	parentDir := filepath.Dir(target)
	if err := os.MkdirAll(parentDir, 0755); err != nil {
		return err
	}

	if len(options) < 2 {
		return fmt.Errorf("windowsMount: mount options(%q) command number(%d) less than 2, skip mounting", options, len(options))
	}
	cmd := fmt.Sprintf(`$User = "AZURE\%s";$PWord = ConvertTo-SecureString -String "%s" -AsPlainText -Force;`+
		`$Credential = New-Object -TypeName System.Management.Automation.PSCredential -ArgumentList $User, $PWord`,
		options[0], options[1])

	driverLetter, err := getAvailableDriveLetter()
	if err != nil {
		return err
	}
	driverPath := driverLetter + ":"
	cmd += fmt.Sprintf(";New-SmbGlobalMapping -LocalPath %s -RemotePath %s -Credential $Credential", driverPath, source)

	ex := exec.New()
	if output, err := ex.Command("powershell", "/c", cmd).CombinedOutput(); err != nil {
		// we don't return error here, even though New-SmbGlobalMapping failed, we still make it successful,
		// will return error when Windows 2016 RS3 is ready on azure
		glog.Errorf("windowsMount: SmbGlobalMapping failed: %v, output: %q", err, string(output))
		return os.MkdirAll(target, 0755)
	}

	if output, err := ex.Command("cmd", "/c", "mklink", "/D", target, driverPath).CombinedOutput(); err != nil {
		return fmt.Errorf("mklink failed: %v, output: %q", err, string(output))
	}

	return nil
}

func (mounter *Mounter) Unmount(target string) error {
	glog.Infof("windowsMount: Unmount target (%q)", target)
	ex := exec.New()
	if output, err := ex.Command("cmd", "/c", "rmdir", target).CombinedOutput(); err != nil {
		return fmt.Errorf("rmdir failed: %v, output: %q", err, string(output))
	}
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
	ex := exec.New()
	output, err := ex.Command("powershell", "/c", cmd).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("getAvailableDriveLetter failed: %v, output: %q", err, string(output))
	}

	if len(output) == 0 {
		return "", fmt.Errorf("windowsMount: there is no available drive letter now")
	}
	return string(output)[:1], nil
}
