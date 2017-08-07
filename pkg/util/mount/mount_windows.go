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
	"os"
	"os/exec"
	"strings"

	"github.com/golang/glog"
)

type Mounter struct {
	mounterPath string
}

func (mounter *Mounter) Mount(source string, target string, fstype string, options []string) error {
	if source == "tmpfs" || len(target) < 3 {
		glog.Infof("Skip mounting source (%s), target (%s)\n, with arguments (%s) in windows", source, target, options)
		return nil
	}

	if !strings.HasPrefix(target, "c:") && !strings.HasPrefix(target, "C:") {
		target = "c:" + target
	}
	parentDir := getWindowsParentDir(target)
	err := os.MkdirAll(parentDir, 0700)
	if err != nil {
		glog.Infof("mkdir(%s) failed: %v\n", parentDir, err)
		return err
	}

	/*
		Comment below code since in Windows Server 2016 RS3, there will be new cmdlet(New-SmbGlobalMapping) to replace "net use"
		mountCmd := "net"
		glog.Infof("Mounting cmd (%s) with arguments (%s), source (%s), target (%s)\n", mountCmd, options, source, target)
		output, err := exec.Command(mountCmd, options...).CombinedOutput()
		if err != nil {
			glog.Infof("Mount failed: %v\nMounting command: %s\n", err, mountCmd)
			return err
		}
		glog.Infof("Mount succeeded, output: %s", output)
	*/

	_, err = exec.Command("cmd", "/c", "mklink", "/D", target, source).CombinedOutput()
	if err != nil {
		glog.Infof("mklink failed: %v\n", err)
	}

	return nil
}

func (mounter *Mounter) Unmount(target string) error {
	glog.Infof("Unmount target (%s)\n", target)
	output, err := exec.Command("cmd", "/c", "rmdir", target).CombinedOutput()
	if err != nil {
		glog.Infof("Unmount failed: %v\n", err)
	} else {
		glog.Infof("Unmount succeeded, output: %s", output)
	}
	return nil
}

func (mounter *Mounter) List() ([]MountPoint, error) {
	return []MountPoint{}, nil
}

func (mounter *Mounter) IsMountPointMatch(mp MountPoint, dir string) bool {
	return (mp.Path == dir)
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
	return true, nil
}

func (mounter *SafeFormatAndMount) formatAndMount(source string, target string, fstype string, options []string) error {
	return nil
}

func (mounter *SafeFormatAndMount) diskLooksUnformatted(disk string) (bool, error) {
	return true, nil
}

func getWindowsParentDir(path string) string {
	index := strings.LastIndex(path, "\\")
	if index != -1 {
		return path[:index]
	}
	return ""
}
