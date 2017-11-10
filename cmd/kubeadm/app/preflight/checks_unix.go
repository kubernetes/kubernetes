// +build !windows

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
	"os"
)

// Check validates if an user has elevated (root) privileges.
func (ipuc IsPrivilegedUserCheck) Check() (warnings, errors []error) {
	errors = []error{}
	if os.Getuid() != 0 {
		errors = append(errors, fmt.Errorf("user is not running as root"))
	}

	return nil, errors
}
