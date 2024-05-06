//go:build windows
// +build windows

/*
Copyright 2022 The Kubernetes Authors.

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

package rbd

import (
	"strconv"

	"k8s.io/mount-utils"
)

func (fake *fakeDiskManager) AttachDisk(b rbdMounter) (string, error) {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()

	// Windows expects Disk Numbers. We start with rbdMapIndex 0, referring to the first Disk.
	volIds, err := mount.ListVolumesOnDisk(strconv.Itoa(fake.rbdMapIndex))
	if err != nil {
		return "", err
	}
	fake.rbdDevices[volIds[0]] = true
	devicePath := strconv.Itoa(fake.rbdMapIndex)
	fake.rbdMapIndex++
	return devicePath, nil
}

func getLoggedSource(devicePath string) (string, error) {
	// Windows mounter is mounting based on the Disk's Unique ID.
	volIds, err := mount.ListVolumesOnDisk(devicePath)
	if err != nil {
		return "", err
	}
	return volIds[0], nil
}
