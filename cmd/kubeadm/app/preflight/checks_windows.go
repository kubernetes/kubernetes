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
	"fmt"
	"os/exec"
	"strings"
)

// Check validates if an user has elevated (administrator) privileges.
func (ipuc IsPrivilegedUserCheck) Check() (warnings, errors []error) {
	errors = []error{}

	// The "Well-known SID" of Administrator group is S-1-5-32-544
	// The following powershell will return "True" if run as an administrator, "False" otherwise
	// See https://msdn.microsoft.com/en-us/library/cc980032.aspx
	args := []string{"[bool](([System.Security.Principal.WindowsIdentity]::GetCurrent()).groups -match \"S-1-5-32-544\")"}
	isAdmin, err := exec.Command("powershell", args...).Output()

	if err != nil {
		errors = append(errors, fmt.Errorf("unable to determine if user is running as administrator: %s", err))
	} else if strings.EqualFold(strings.TrimSpace(string(isAdmin)), "false") {
		errors = append(errors, fmt.Errorf("user is not running as administrator"))
	}

	return nil, errors
}
