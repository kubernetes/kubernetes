//go:build !providerless && windows
// +build !providerless,windows

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

func getRawUUID() (string, error) {
	result, err := exec.Command("wmic", "bios", "get", "serialnumber").Output()
	if err != nil {
		return "", err
	}
	lines := strings.FieldsFunc(string(result), func(r rune) bool {
		switch r {
		case '\n', '\r':
			return true
		default:
			return false
		}
	})
	if len(lines) != 2 {
		return "", fmt.Errorf("received unexpected value retrieving vm uuid: %q", string(result))
	}
	return lines[1], nil
}
