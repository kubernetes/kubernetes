// +build !providerless
// +build windows

/*
Copyright 2019 The Kubernetes Authors.

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

package vsphere_volume

import (
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"

	"k8s.io/klog/v2"
)

type diskInfoResult struct {
	Number       json.Number
	SerialNumber string
}

func verifyDevicePath(path string) (string, error) {
	if !strings.Contains(path, diskByIDPath) {
		// If this volume has already been mounted then
		// its devicePath will have already been converted to a disk number
		klog.V(4).Infof("Found vSphere disk attached with disk number %v", path)
		return path, nil
	}
	cmd := exec.Command("powershell", "/c", "Get-Disk | Select Number, SerialNumber | ConvertTo-JSON")
	output, err := cmd.Output()
	if err != nil {
		klog.Errorf("Get-Disk failed, error: %v, output: %q", err, string(output))
		return "", err
	}

	var results []diskInfoResult
	if err = json.Unmarshal(output, &results); err != nil {
		klog.Errorf("Failed to unmarshal Get-Disk json, output: %q", string(output))
		return "", err
	}
	serialNumber := strings.TrimPrefix(path, diskByIDPath+diskSCSIPrefix)
	for _, v := range results {
		if v.SerialNumber == serialNumber {
			klog.V(4).Infof("Found vSphere disk attached with serial %v", serialNumber)
			return v.Number.String(), nil
		}
	}

	return "", fmt.Errorf("unable to find vSphere disk with serial %v", serialNumber)
}
