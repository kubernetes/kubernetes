/*
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

package gpu_detection

import (
	"fmt"
	"github.com/StackExchange/wmi"
	"github.com/golang/glog"
	"strings"
)

type GPUInfo struct {
	Name                 string
	PNPDeviceID          string
	Status               string
	DriverVersion        string
	AdapterCompatibility string
}

func GetGPUList() []GPUInfo {
	var list []GPUInfo
	err := wmi.Query("Select * from Win32_VideoController", &list)
	if err != nil {
		fmt.Errorf("failed to request wmi to get GPU info : %v", err)
	}
	return list
}

func (gpu GPUInfo) MatchName(vendor string) bool {
	if strings.Contains(strings.ToLower(gpu.Name), strings.ToLower(vendor)) {
		return true
	}
	return false
}

func (gpu GPUInfo) IsStatusOK() bool {
	if strings.ToLower(gpu.Status) == "ok" {
		return true
	}
	return false
}

func GetGPUInfo(id string) *GPUInfo {
	for _, gpu := range GetGPUList() {
		if gpu.PNPDeviceID == id {
			return &gpu
		}
	}
	glog.Errorf("GPU not found for : %s", id)
	return nil
}
