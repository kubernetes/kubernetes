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
	"os/exec"
	"strings"
)

var _ Validator = &OSValidator{}

type OSValidator struct{}

func (c *OSValidator) Name() string {
	return "os"
}

func (c *OSValidator) Validate(spec SysSpec) error {
	out, err := exec.Command("uname").CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to get os name: %v", err)
	}
	return c.validateOS(strings.TrimSpace(string(out)), spec.OS)
}

func (c *OSValidator) validateOS(os, specOS string) error {
	if os != specOS {
		report("OS", os, bad)
		return fmt.Errorf("unsupported operating system: %s", os)
	}
	report("OS", os, good)
	return nil
}
