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
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

func (fake *fakeDiskManager) AttachDisk(b rbdMounter) (string, error) {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	fake.rbdMapIndex++

	// Windows expects Disk Numbers.
	volIds, err := listVolumesOnDisk(strconv.Itoa(fake.rbdMapIndex))
	if err != nil {
		return "", err
	}
	fake.rbdDevices[volIds[0]] = true
	devicePath := strconv.Itoa(fake.rbdMapIndex)
	return devicePath, nil
}

// listVolumesOnDisk - returns back list of volumes(volumeIDs) in the disk (requested in diskID) on Windows.
func listVolumesOnDisk(diskID string) (volumeIDs []string, err error) {
	cmd := fmt.Sprintf("(Get-Disk -DeviceId %s | Get-Partition | Get-Volume).UniqueId", diskID)
	output, err := exec.Command("powershell", "/c", cmd).CombinedOutput()
	if err != nil {
		return []string{}, fmt.Errorf("error list volumes on disk. cmd: %s, output: %s, error: %v", cmd, string(output), err)
	}

	volumeIds := strings.Split(strings.TrimSpace(string(output)), "\r\n")
	return volumeIds, nil
}

func getLoggedSource(devicePath string) (string, error) {
	// Windows mounter is mounting based on the Disk's Unique ID.
	volIds, err := listVolumesOnDisk(devicePath)
	if err != nil {
		return "", err
	}
	return volIds[0], nil
}
