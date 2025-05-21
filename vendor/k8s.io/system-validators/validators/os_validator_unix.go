//go:build !windows
// +build !windows

/*
Copyright 2016 The Kubernetes Authors.

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
	"strings"

	"golang.org/x/sys/unix"
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
	var utsname unix.Utsname
	err := unix.Uname(&utsname)
	if err != nil {
		return nil, []error{fmt.Errorf("failed to get OS name: %w", err)}
	}
	os := strings.TrimSpace(unix.ByteSliceToString(utsname.Sysname[:]))
	if err = o.validateOS(os, spec.OS); err != nil {
		return nil, []error{err}
	}
	return nil, nil
}

func (o *OSValidator) validateOS(os, specOS string) error {
	if os != specOS {
		o.Reporter.Report("OS", os, bad)
		return fmt.Errorf("unsupported operating system: %s", os)
	}
	o.Reporter.Report("OS", os, good)
	return nil
}
