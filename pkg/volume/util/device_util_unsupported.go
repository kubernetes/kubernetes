//go:build !linux
// +build !linux

/*
Copyright 2016 The Kubernetes Authors.

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

package util

// FindMultipathDeviceForDevice unsupported returns ""
func (handler *deviceHandler) FindMultipathDeviceForDevice(device string) string {
	return ""
}

// FindSlaveDevicesOnMultipath unsupported returns ""
func (handler *deviceHandler) FindSlaveDevicesOnMultipath(disk string) []string {
	out := []string{}
	return out
}

// GetISCSIPortalHostMapForTarget unsupported returns nil
func (handler *deviceHandler) GetISCSIPortalHostMapForTarget(targetIqn string) (map[string]int, error) {
	portalHostMap := make(map[string]int)
	return portalHostMap, nil
}

// FindDevicesForISCSILun unsupported returns nil
func (handler *deviceHandler) FindDevicesForISCSILun(targetIqn string, lun int) ([]string, error) {
	devices := []string{}
	return devices, nil
}
