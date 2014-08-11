// +build linux

/*
Copyright 2014 Google Inc. All rights reserved.

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

package volume

import (
	"io/ioutil"
	"regexp"
	"strings"
	"syscall"
)

const MOUNT_MS_BIND = syscall.MS_BIND
const MOUNT_MS_RDONLY = syscall.MS_RDONLY

type DiskMounter struct{}

// Wraps syscall.Mount()
func (mounter *DiskMounter) Mount(source string, target string, fstype string, flags uintptr, data string) error {
	return syscall.Mount(source, target, fstype, flags, data)
}

// Wraps syscall.Unmount()
func (mounter *DiskMounter) Unmount(target string, flags int) error {
	return syscall.Unmount(target, flags)
}

// Examines /proc/mounts to find the source device of the PD resource and the
// number of references to that device. Returns both the full device path under
// the /dev tree and the number of references.
func (mounter *DiskMounter) RefCount(PD *PersistentDisk) (string, int, error) {
	contents, err := ioutil.ReadFile("/proc/mounts")
	if err != nil {
		return "", -1, err
	}
	refCount := 0
	var deviceName string
	lines := strings.Split(string(contents), "\n")
	// Find the actual device path.
	for _, line := range lines {
		success, err := regexp.MatchString(PD.GetPath(), line)
		if err != nil {
			return "", -1, err
		}
		if success {
			deviceName = strings.Split(line, " ")[0]
		}
	}
	// Find the number of references to the device.
	for _, line := range lines {
		success, err := regexp.MatchString(deviceName, line)
		if err != nil {
			return "", -1, err
		}
		if success {
			refCount++
		}
	}
	return deviceName, refCount, nil
}
