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

package system

import (
	"os/exec"
	"strings"
)

// DefaultSysSpec is the default SysSpec for Windows.
var DefaultSysSpec = SysSpec{
	OS: "Windows Server",
	KernelSpec: KernelSpec{
		Versions:     []string{`10\.0\.1439[3-9]`, `10\.0\.14[4-9][0-9]{2}`, `10\.0\.1[5-9][0-9]{3}`, `10\.0\.[2-9][0-9]{4}`, `10\.[1-9]+\.[0-9]+`}, //requires >= '10.0.14393'
		VersionsNote: "The kernel version should be >= '10.0.14393'",
		Required:     []KernelConfig{},
		Optional:     []KernelConfig{},
		Forbidden:    []KernelConfig{},
	},
	RuntimeSpec: RuntimeSpec{
		DockerSpec: &DockerSpec{
			Version:     []string{`18\.0[6,9]\..*`, `19\.03\..*`},
			GraphDriver: []string{"windowsfilter"},
		},
	},
}

// KernelValidatorHelperImpl is the 'windows' implementation of KernelValidatorHelper
type KernelValidatorHelperImpl struct{}

var _ KernelValidatorHelper = &KernelValidatorHelperImpl{}

// GetKernelReleaseVersion returns the Windows release version (e.g. 10.0.14393) as a string.
// It does not include the UBR (revision)
func (o *KernelValidatorHelperImpl) GetKernelReleaseVersion() (string, error) {
	args := []string{`$props = Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'; ` +
		`"$($props.CurrentMajorVersionNumber).$($props.CurrentMinorVersionNumber).$($props.CurrentBuildNumber)"`}
	releaseVersion, err := exec.Command("powershell", args...).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(releaseVersion)), nil
}
