//go:build windows
// +build windows

/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"os/exec"
	"strings"
)

var _ Validator = &OSValidator{}

// OSValidator validates OS.
type OSValidator struct {
	Reporter Reporter
}

// Name is part of the system.Validator interface.
func (o *OSValidator) Name() string {
	return "os"
}

// Validate is part of the system.Validator interface.
func (o *OSValidator) Validate(spec SysSpec) ([]error, []error) {
	args := []string{`(Get-ItemProperty -Path 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').ProductName`}
	os, err := exec.Command("powershell", args...).Output()
	if err != nil {
		return nil, []error{fmt.Errorf("failed to get OS name: %w", err)}
	}
	if err = o.validateOS(strings.TrimSpace(string(os)), spec.OS); err != nil {
		return nil, []error{err}
	}
	return nil, nil
}

// validateOS would check if the reported string such as 'Windows Server 2019' contains
// the required OS prefix from the spec 'Windows Server'.
func (o *OSValidator) validateOS(os, specOS string) error {
	if !strings.HasPrefix(os, specOS) {
		o.Reporter.Report("OS", os, bad)
		return fmt.Errorf("unsupported operating system: %s", os)
	}
	o.Reporter.Report("OS", os, good)
	return nil
}
