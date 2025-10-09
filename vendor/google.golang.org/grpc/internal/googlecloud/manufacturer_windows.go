/*
 *
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package googlecloud

import (
	"errors"
	"os/exec"
	"regexp"
	"strings"
)

const (
	windowsCheckCommand      = "powershell.exe"
	windowsCheckCommandArgs  = "Get-WmiObject -Class Win32_BIOS"
	powershellOutputFilter   = "Manufacturer"
	windowsManufacturerRegex = ":(.*)"
)

func manufacturer() ([]byte, error) {
	cmd := exec.Command(windowsCheckCommand, windowsCheckCommandArgs)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(strings.TrimSuffix(string(out), "\n"), "\n") {
		if strings.HasPrefix(line, powershellOutputFilter) {
			re := regexp.MustCompile(windowsManufacturerRegex)
			name := re.FindString(line)
			name = strings.TrimLeft(name, ":")
			return []byte(name), nil
		}
	}
	return nil, errors.New("cannot determine the machine's manufacturer")
}
