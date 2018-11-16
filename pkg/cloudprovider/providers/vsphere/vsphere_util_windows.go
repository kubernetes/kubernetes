// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"fmt"
	"os/exec"
	"strings"
)

func GetVMUUID() (string, error) {
	result, err := exec.Command("wmic", "csproduct", "get", "UUID").Output()
	if err != nil {
		return "", fmt.Errorf("error retrieving vm uuid: %s", err)
	}
	fields := strings.Fields(string(result))
	if len(fields) != 2 {
		return "", fmt.Errorf("received unexpected value retrieving vm uuid: %q", string(result))
	}
	return fields[1], nil
}
