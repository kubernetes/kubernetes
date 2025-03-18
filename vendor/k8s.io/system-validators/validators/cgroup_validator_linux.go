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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
	cgroupsConfigPrefix      = "CGROUPS_"
	mountsFilePath           = "/proc/mounts"
	defaultUnifiedMountPoint = "/sys/fs/cgroup"
)

// getUnifiedMountpoint checks if the default mount point is available.
// If not, it parses the mounts file to find a valid cgroup mount point.
func getUnifiedMountpoint(path string) (string, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", false, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	var cgroupV1MountPoint string
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.Contains(line, "cgroup") {
			continue
		}
		// Example fields: `cgroup2 /sys/fs/cgroup cgroup2 rw,seclabel,nosuid,nodev,noexec,relatime 0 0`.
		fields := strings.Fields(line)
		if len(fields) >= 3 {
			// If default unified mount point is available, return it directly.
			if fields[1] == defaultUnifiedMountPoint {
				if fields[2] == "tmpfs" {
					// if `/sys/fs/cgroup/memory` is a dir, this means it uses cgroups v1
					info, err := os.Stat(filepath.Join(defaultUnifiedMountPoint, "memory"))
					return defaultUnifiedMountPoint, os.IsNotExist(err) || !info.IsDir(), nil
				}
				return defaultUnifiedMountPoint, fields[2] == "cgroup2", nil
			}
			switch fields[2] {
			case "cgroup2":
				// Return the first cgroups v2 mount point directly.
				return fields[1], true, nil
			case "cgroup":
				// Set the first cgroups v1 mount point only,
				// and continue the loop to find if there is a cgroups v2 mount point.
				if len(cgroupV1MountPoint) == 0 {
					cgroupV1MountPoint = fields[1]
				}
			}
		}
	}
	// Return cgroups v1 mount point if no cgroups v2 mount point is found.
	if len(cgroupV1MountPoint) != 0 {
		return cgroupV1MountPoint, false, nil
	}
	return "", false, fmt.Errorf("cannot get a cgroupfs mount point from %q", path)
}

// Validate is part of the system.Validator interface.
func (c *CgroupsValidator) Validate(spec SysSpec) (warns, errs []error) {
	unifiedMountpoint, isCgroupsV2, err := getUnifiedMountpoint(mountsFilePath)
	if err != nil {
		return nil, []error{fmt.Errorf("cannot get a cgroup mount point: %w", err)}
	}
	var requiredCgroupSpec []string
	var optionalCgroupSpec []string
	var subsystems []string
	var warn error
	if isCgroupsV2 {
		subsystems, err, warn = c.getCgroupV2Subsystems(unifiedMountpoint)
		if err != nil {
			return nil, []error{fmt.Errorf("failed to get cgroups v2 subsystems: %w", err)}
		}
		if warn != nil {
			warns = append(warns, warn)
		}
		requiredCgroupSpec = spec.CgroupsV2
		optionalCgroupSpec = spec.CgroupsV2Optional
	} else {
		warns = append(warns, errors.New("cgroups v1 support is in maintenance mode, please migrate to cgroups v2"))
		subsystems, err = c.getCgroupV1Subsystems()
		if err != nil {
			return nil, []error{fmt.Errorf("failed to get cgroups v1 subsystems: %w", err)}
		}
		requiredCgroupSpec = spec.Cgroups
		optionalCgroupSpec = spec.CgroupsOptional
	}

	if missingRequired := c.validateCgroupSubsystems(requiredCgroupSpec, subsystems, true); len(missingRequired) != 0 {
		errs = []error{fmt.Errorf("missing required cgroups: %s", strings.Join(missingRequired, " "))}
	}
	if missingOptional := c.validateCgroupSubsystems(optionalCgroupSpec, subsystems, false); len(missingOptional) != 0 {
		warns = append(warns, fmt.Errorf("missing optional cgroups: %s", strings.Join(missingOptional, " ")))
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
	// Get the subsystems from /proc/cgroups when cgroups v1 is used.
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

func (c *CgroupsValidator) getCgroupV2Subsystems(unifiedMountpoint string) ([]string, error, error) {
	// Some controllers are implicitly enabled by the kernel.
	// Those controllers do not appear in /sys/fs/cgroup/cgroup.controllers.
	// https://github.com/torvalds/linux/blob/v5.3/kernel/cgroup/cgroup.c#L433-L434
	// For freezer, we use checkCgroupV2Freeze() to check.
	// For others, we assume these are always available, as it is hard to detect availability.
	// We hardcode the following as initial controllers.
	// - devices: implemented in kernel 4.15.
	subsystems := []string{"devices"}
	freezeSupported, warn := checkCgroupV2Freeze(unifiedMountpoint)
	if freezeSupported {
		subsystems = append(subsystems, "freezer")
	}
	data, err := os.ReadFile(filepath.Join(unifiedMountpoint, "cgroup.controllers"))
	if err != nil {
		return nil, err, warn
	}
	subsystems = append(subsystems, strings.Fields(string(data))...)
	return subsystems, err, warn
}

// checkCgroupV2Freeze checks if the freezer controller is enabled in Linux kernels 5.2.
// It determines that by creating a cgroup.freeze file under the unified mountpoint location.
func checkCgroupV2Freeze(unifiedMountpoint string) (isCgroupfs bool, warn error) {
	const freezeFile = "cgroup.freeze"
	tmpDir, warn := os.MkdirTemp(unifiedMountpoint, "freezer-test")
	if warn != nil {
		return
	}
	defer func() {
		err := os.RemoveAll(tmpDir)
		if err != nil {
			warn = fmt.Errorf("error removing directory %q: %v", tmpDir, err)
		}
	}()
	_, warn = os.Stat(filepath.Join(tmpDir, freezeFile))
	if os.IsNotExist(warn) {
		return
	} else if warn != nil {
		// If the err is not NotExist error, it means that `cgroup.freeze` exists.
		isCgroupfs = true
		warn = fmt.Errorf("could not stat %q file in %q: %v", freezeFile, tmpDir, warn)
		return
	}
	isCgroupfs = true
	return
}
