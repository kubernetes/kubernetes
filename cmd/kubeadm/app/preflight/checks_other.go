//go:build !linux
// +build !linux

/*
Copyright 2022 The Kubernetes Authors.

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
	system "k8s.io/system-validators/validators"
	utilsexec "k8s.io/utils/exec"
)

// addOSValidator adds a new OSValidator
// No-op for Darwin (MacOS), Windows.
func addOSValidator(validators []system.Validator, _ *system.StreamReporter) []system.Validator {
	return validators
}

// addIPv6Checks adds IPv6 related checks
// No-op for Darwin (MacOS), Windows.
func addIPv6Checks(checks []Checker) []Checker {
	return checks
}

// addIPv4Checks adds IPv4 related checks
// No-op for Darwin (MacOS), Windows.
func addIPv4Checks(checks []Checker) []Checker {
	return checks
}

// addSwapCheck adds a swap check
// No-op for Darwin (MacOS), Windows.
func addSwapCheck(checks []Checker) []Checker {
	return checks
}

// addExecChecks adds checks that verify if certain binaries are in PATH
// No-op for Darwin (MacOS), Windows.
func addExecChecks(checks []Checker, _ utilsexec.Interface, _ string) []Checker {
	return checks
}
