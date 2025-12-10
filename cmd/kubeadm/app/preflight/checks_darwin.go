//go:build darwin
// +build darwin

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

package preflight

import utilsexec "k8s.io/utils/exec"

// This is a MacOS stub

// Check number of memory required by kubeadm
// No-op for Darwin (MacOS).
func (mc MemCheck) Check() (warnings, errorList []error) {
	return nil, nil
}

// addExecChecks adds checks that verify if certain binaries are in PATH
// No-op for Darwin (MacOS).
func addExecChecks(checks []Checker, _ utilsexec.Interface, _ string) []Checker {
	return checks
}
