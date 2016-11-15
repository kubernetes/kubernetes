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
	"bufio"
	"fmt"
	"os"
	"strings"
)

var _ Validator = &CgroupsValidator{}

type CgroupsValidator struct{}

func (c *CgroupsValidator) Name() string {
	return "cgroups"
}

const (
	cgroupsConfigPrefix = "CGROUPS_"
)

func (c *CgroupsValidator) Validate(spec SysSpec) error {
	subsystems, err := c.getCgroupSubsystems()
	if err != nil {
		return fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}
	return c.validateCgroupSubsystems(spec.Cgroups, subsystems)
}

func (c *CgroupsValidator) validateCgroupSubsystems(cgroupSpec, subsystems []string) error {
	missing := []string{}
	for _, cgroup := range cgroupSpec {
		found := false
		for _, subsystem := range subsystems {
			if cgroup == subsystem {
				found = true
				break
			}
		}
		item := cgroupsConfigPrefix + strings.ToUpper(cgroup)
		if found {
			report(item, "enabled", good)
		} else {
			report(item, "missing", bad)
			missing = append(missing, cgroup)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("missing cgroups: %s", strings.Join(missing, " "))
	}
	return nil

}

func (c *CgroupsValidator) getCgroupSubsystems() ([]string, error) {
	f, err := os.Open("/proc/cgroups")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	subsystems := []string{}
	s := bufio.NewScanner(f)
	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}
		text := s.Text()
		if text[0] != '#' {
			parts := strings.Fields(text)
			if len(parts) >= 4 && parts[3] != "0" {
				subsystems = append(subsystems, parts[0])
			}
		}
	}
	return subsystems, nil
}
