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

//DeviceUtil is a util for common device methods
type DeviceUtil interface {
	FindMultipathDeviceForDevice(disk string) string
	FindSlaveDevicesOnMultipath(disk string) []string
	GetISCSIPortalHostMapForTarget(targetIqn string) (map[string]int, error)
	FindDevicesForISCSILun(targetIqn string, lun int) ([]string, error)
}

type deviceHandler struct {
	getIo IoUtil
}

//NewDeviceHandler Create a new IoHandler implementation
func NewDeviceHandler(io IoUtil) DeviceUtil {
	return &deviceHandler{getIo: io}
}
