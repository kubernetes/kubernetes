// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package preflight

import (
	"os/exec"
	"strings"

	"github.com/pkg/errors"
)

// Check validates if an user has elevated (administrator) privileges.
func (ipuc IsPrivilegedUserCheck) Check() (warnings, errorList []error) {
	errorList = []error{}

	// The "Well-known SID" of Administrator group is S-1-5-32-544
	// The following powershell will return "True" if run as an administrator, "False" otherwise
	// See https://msdn.microsoft.com/en-us/library/cc980032.aspx
	args := []string{"[bool](([System.Security.Principal.WindowsIdentity]::GetCurrent()).groups -match \"S-1-5-32-544\")"}
	isAdmin, err := exec.Command("powershell", args...).Output()

	if err != nil {
		errorList = append(errorList, errors.Wrap(err, "unable to determine if user is running as administrator"))
	} else if strings.EqualFold(strings.TrimSpace(string(isAdmin)), "false") {
		errorList = append(errorList, errors.New("user is not running as administrator"))
	}

	return nil, errorList
}

// Check validates if Docker is setup to use systemd as the cgroup driver.
// No-op for Windows.
func (idsc IsDockerSystemdCheck) Check() (warnings, errorList []error) {
	return nil, nil
}
