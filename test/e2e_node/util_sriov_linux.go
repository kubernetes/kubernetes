//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package e2enode

import (
	"os/exec"
	"strconv"
	"strings"
)

// countSRIOVDevices provides a rough estimate of SRIOV Virtual Functions available on the system.
// This is a rough check we use to rule out unsuitable systems, not to detect suitable systems.
func countSRIOVDevices() (int, error) {
	outData, err := exec.Command("/bin/sh", "-c", "ls /sys/bus/pci/devices/*/physfn | wc -w").Output()
	if err != nil {
		return -1, err
	}
	return strconv.Atoi(strings.TrimSpace(string(outData)))
}
