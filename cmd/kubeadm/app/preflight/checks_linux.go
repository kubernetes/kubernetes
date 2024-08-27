//go:build linux
// +build linux

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

import (
	"syscall"

	"github.com/pkg/errors"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	system "k8s.io/system-validators/validators"
	utilsexec "k8s.io/utils/exec"
)

// Check number of memory required by kubeadm
func (mc MemCheck) Check() (warnings, errorList []error) {
	info := syscall.Sysinfo_t{}
	err := syscall.Sysinfo(&info)
	if err != nil {
		errorList = append(errorList, errors.Wrapf(err, "failed to get system info"))
	}

	// Totalram holds the total usable memory. Unit holds the size of a memory unit in bytes. Multiply them and convert to MB
	actual := uint64(info.Totalram) * uint64(info.Unit) / 1024 / 1024
	if actual < mc.Mem {
		errorList = append(errorList, errors.Errorf("the system RAM (%d MB) is less than the minimum %d MB", actual, mc.Mem))
	}
	return warnings, errorList
}

// addOSValidator adds a new OSValidator
func addOSValidator(validators []system.Validator, reporter *system.StreamReporter) []system.Validator {
	validators = append(validators, &system.OSValidator{Reporter: reporter}, &system.CgroupsValidator{Reporter: reporter})
	return validators
}

// addIPv6Checks adds IPv6 related checks
func addIPv6Checks(checks []Checker) []Checker {
	checks = append(checks,
		FileContentCheck{Path: ipv6DefaultForwarding, Content: []byte{'1'}},
	)
	return checks
}

// addIPv4Checks adds IPv4 related checks
func addIPv4Checks(checks []Checker) []Checker {
	checks = append(checks,
		FileContentCheck{Path: ipv4Forward, Content: []byte{'1'}})
	return checks
}

// addSwapCheck adds a swap check
func addSwapCheck(checks []Checker) []Checker {
	checks = append(checks, SwapCheck{})
	return checks
}

// addExecChecks adds checks that verify if certain binaries are in PATH
func addExecChecks(checks []Checker, execer utilsexec.Interface, k8sVersion string) []Checker {
	// For k8s >= 1.32.0, kube-proxy no longer depends on conntrack to be present in PATH
	// (ref: https://github.com/kubernetes/kubernetes/pull/126952)
	if v, err := utilversion.ParseSemantic(k8sVersion); err == nil {
		if v.LessThan(utilversion.MustParseSemantic("1.32.0")) {
			checks = append(checks, InPathCheck{executable: "conntrack", mandatory: true, exec: execer})
		}
	}

	checks = append(checks,
		InPathCheck{executable: "ip", mandatory: true, exec: execer},
		InPathCheck{executable: "iptables", mandatory: true, exec: execer},
		InPathCheck{executable: "mount", mandatory: true, exec: execer},
		InPathCheck{executable: "nsenter", mandatory: true, exec: execer},
		InPathCheck{executable: "ebtables", mandatory: false, exec: execer},
		InPathCheck{executable: "ethtool", mandatory: false, exec: execer},
		InPathCheck{executable: "socat", mandatory: false, exec: execer},
		InPathCheck{executable: "tc", mandatory: false, exec: execer},
		InPathCheck{executable: "touch", mandatory: false, exec: execer})
	return checks
}
