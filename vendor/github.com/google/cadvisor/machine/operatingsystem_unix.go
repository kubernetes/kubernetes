// Copyright 2020 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build freebsd || darwin || linux
// +build freebsd darwin linux

package machine

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
)

var rex = regexp.MustCompile("(PRETTY_NAME)=(.*)")

// getOperatingSystem gets the name of the current operating system.
func getOperatingSystem() (string, error) {
	if runtime.GOOS == "darwin" || runtime.GOOS == "freebsd" {
		cmd := exec.Command("uname", "-s")
		osName, err := cmd.Output()
		if err != nil {
			return "", err
		}
		return string(osName), nil
	}
	bytes, err := os.ReadFile("/etc/os-release")
	if err != nil && os.IsNotExist(err) {
		// /usr/lib/os-release in stateless systems like Clear Linux
		bytes, err = os.ReadFile("/usr/lib/os-release")
	}
	if err != nil {
		return "", fmt.Errorf("error opening file : %v", err)
	}
	line := rex.FindAllStringSubmatch(string(bytes), -1)
	if len(line) > 0 {
		return strings.Trim(line[0][2], "\""), nil
	}
	return "Linux", nil
}
