//go:build linux
// +build linux

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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

var _ Validator = &CgroupsValidator{}

// CgroupsValidator validates cgroup configuration.
type CgroupsValidator struct {
	Reporter Reporter
}

// Name is part of the system.Validator interface.
func (c *CgroupsValidator) Name() string {
	return "cgroups"
}

const (
	cgroupsConfigPrefix = "CGROUPS_"
	unifiedMountpoint   = "/sys/fs/cgroup"
)

// Validate is part of the system.Validator interface.
func (c *CgroupsValidator) Validate(spec SysSpec) (warns, errs []error) {
	// Get the subsystems from /sys/fs/cgroup/cgroup.controllers when cgroup v2 is used.
	// /proc/cgroups is meaningless for v2
	// https://github.com/torvalds/linux/blob/v5.3/Documentation/admin-guide/cgroup-v2.rst#deprecated-v1-core-features
	var st unix.Statfs_t
	var err error
	if err := unix.Statfs(unifiedMountpoint, &st); err != nil {
		return nil, []error{errors.Wrap(err, "cannot statfs the cgroupv2 root")}
	}
	var requiredCgroupSpec []string
	var optionalCgroupSpec []string
	var subsystems []string
	if st.Type == unix.CGROUP2_SUPER_MAGIC {
		subsystems, err = c.getCgroupV2Subsystems()
		if err != nil {
			return nil, []error{errors.Wrap(err, "failed to get cgroup v2 subsystems")}
		}
		requiredCgroupSpec = spec.CgroupsV2
		optionalCgroupSpec = spec.CgroupsV2Optional
	} else {
		subsystems, err = c.getCgroupV1Subsystems()
		if err != nil {
			return nil, []error{errors.Wrap(err, "failed to get cgroup v1 subsystems")}
		}
		requiredCgroupSpec = spec.Cgroups
		optionalCgroupSpec = spec.CgroupsOptional
	}

	if missingRequired := c.validateCgroupSubsystems(requiredCgroupSpec, subsystems, true); len(missingRequired) != 0 {
		errs = []error{errors.Errorf("missing required cgroups: %s", strings.Join(missingRequired, " "))}
	}
	if missingOptional := c.validateCgroupSubsystems(optionalCgroupSpec, subsystems, false); len(missingOptional) != 0 {
		warns = []error{errors.Errorf("missing optional cgroups: %s", strings.Join(missingOptional, " "))}
	}
	return
}

// validateCgroupSubsystems returns a list with the missing cgroups in the cgroup
func (c *CgroupsValidator) validateCgroupSubsystems(cgroups, subsystems []string, required bool) []string {
	var missing []string
	for _, cgroup := range cgroups {
		found := false
		for _, subsystem := range subsystems {
			if cgroup == subsystem {
				found = true
				break
			}
		}
		item := cgroupsConfigPrefix + strings.ToUpper(cgroup)
		if found {
			c.Reporter.Report(item, "enabled", good)
			continue
		} else if required {
			c.Reporter.Report(item, "missing", bad)
		} else {
			c.Reporter.Report(item, "missing", warn)
		}
		missing = append(missing, cgroup)
	}
	return missing

}

func (c *CgroupsValidator) getCgroupV1Subsystems() ([]string, error) {
	// Get the subsystems from /proc/cgroups when cgroup v1 is used.
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

func (c *CgroupsValidator) getCgroupV2Subsystems() ([]string, error) {
	// Some controllers are implicitly enabled by the kernel.
	// Those controllers do not appear in /sys/fs/cgroup/cgroup.controllers.
	// https://github.com/torvalds/linux/blob/v5.3/kernel/cgroup/cgroup.c#L433-L434
	// We assume these are always available, as it is hard to detect availability.
	// So, we hardcode the following as "pseudo" controllers.
	// - devices: implemented in kernel 4.15
	// - freezer: implemented in kernel 5.2
	pseudo := []string{"devices", "freezer"}
	data, err := ioutil.ReadFile(filepath.Join(unifiedMountpoint, "cgroup.controllers"))
	if err != nil {
		return nil, err
	}
	subsystems := append(pseudo, strings.Fields(string(data))...)
	return subsystems, nil
}
