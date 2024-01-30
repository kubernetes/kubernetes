//go:build !windows
// +build !windows

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
)

func (fake *fakeDiskManager) AttachDisk(b rbdMounter) (string, error) {
	fake.mutex.Lock()
	defer fake.mutex.Unlock()
	devicePath := fmt.Sprintf("/dev/rbd%d", fake.rbdMapIndex)
	fake.rbdDevices[devicePath] = true
	// Increment rbdMapIndex afterwards, so we can start from rbd0.
	fake.rbdMapIndex++
	return devicePath, nil
}

func getLoggedSource(devicePath string) (string, error) {
	return devicePath, nil
}
