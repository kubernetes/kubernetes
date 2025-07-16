//go:build windows
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
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
	utilsexec "k8s.io/utils/exec"
)

// Check validates if a user has elevated (administrator) privileges.
func (ipuc IsPrivilegedUserCheck) Check() (warnings, errorList []error) {
	hProcessToken := windows.GetCurrentProcessToken()
	if hProcessToken.IsElevated() {
		return nil, nil
	}
	return nil, []error{errors.New("the kubeadm process must be run by a user with elevated privileges")}
}

// Check number of memory required by kubeadm
// No-op for Windows.
func (mc MemCheck) Check() (warnings, errorList []error) {
	return nil, nil
}

// addExecChecks adds checks that verify if certain binaries are in PATH.
func addExecChecks(checks []Checker, execer utilsexec.Interface, _ string) []Checker {
	// kubeadm requires xcopy to be present in PATH for copying etcd directories.
	checks = append(checks, InPathCheck{executable: "xcopy", mandatory: true, exec: execer})
	return checks
}
