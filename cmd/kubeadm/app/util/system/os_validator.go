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
	"os/exec"
	"strings"

	"github.com/pkg/errors"
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
func (o *OSValidator) Validate(spec SysSpec) (error, error) {
	os, err := exec.Command("uname").CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get os name")
	}
	return nil, o.validateOS(strings.TrimSpace(string(os)), spec.OS)
}

func (o *OSValidator) validateOS(os, specOS string) error {
	if os != specOS {
		o.Reporter.Report("OS", os, bad)
		return errors.Errorf("unsupported operating system: %s", os)
	}
	o.Reporter.Report("OS", os, good)
	return nil
}
