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
	"bufio"
	"io"
	"os"
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
func (mounter *DiskMounter) RefCount(mount Interface) (string, int, error) {
	// TODO(jonesdl) This can be split up into two procedures, finding the device path
	// and finding the number of references. The parsing could also be separated and another
	// utility could determine if a volume's path is an active mount point.
	file, err := os.Open("/proc/mounts")
	if err != nil {
		return "", -1, err
	}
	defer file.Close()
	scanner := bufio.NewReader(file)
	refCount := 0
	var deviceName string
	// Find the actual device path.
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		success, err := regexp.MatchString(mount.GetPath(), line)
		if err != nil {
			return "", -1, err
		}
		if success {
			deviceName = strings.Split(line, " ")[0]
		}
	}
	file.Close()
	file, err = os.Open("/proc/mounts")
	scanner.Reset(bufio.NewReader(file))
	// Find the number of references to the device.
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		if strings.Split(line, " ")[0] == deviceName {
			refCount++
		}
	}
	return deviceName, refCount, nil
}
