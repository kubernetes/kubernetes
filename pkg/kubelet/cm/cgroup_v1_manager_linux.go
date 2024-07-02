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

package cm

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/cgroups/fscommon"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

const cgroupv1MemLimitFile string = "memory.limit_in_bytes"

// cgroupV1impl implements the CgroupManager interface
// for cgroup v1.
// It's a stateless object which can be used to
// update, create or delete any number of cgroups
// It relies on runc/libcontainer cgroup managers.
type cgroupV1impl struct {
	cgroupCommon
}

func NewCgroupV1Manager(cs *CgroupSubsystems, cgroupDriver string) CgroupManager {
	return &cgroupV1impl{
		cgroupCommon: newCgroupCommon(cs, cgroupDriver),
	}
}

// Version of the cgroup implementation on the host
func (c *cgroupV1impl) Version() int {
	return 1
}

// Validate checks if all subsystem cgroups are valid
func (c *cgroupV1impl) Validate(name CgroupName) error {
	// Get map of all cgroup paths on the system for the particular cgroup
	cgroupPaths := c.buildCgroupPaths(name)

	// the presence of alternative control groups not known to runc confuses
	// the kubelet existence checks.
	// ideally, we would have a mechanism in runc to support Exists() logic
	// scoped to the set control groups it understands.  this is being discussed
	// in https://github.com/opencontainers/runc/issues/1440
	// once resolved, we can remove this code.
	allowlistControllers := sets.New[string]("cpu", "cpuacct", "cpuset", "memory", "systemd", "pids")

	if _, ok := c.subsystems.MountPoints["hugetlb"]; ok {
		allowlistControllers.Insert("hugetlb")
	}
	var missingPaths []string
	// If even one cgroup path doesn't exist, then the cgroup doesn't exist.
	for controller, path := range cgroupPaths {
		// ignore mounts we don't care about
		if !allowlistControllers.Has(controller) {
			continue
		}
		if !libcontainercgroups.PathExists(path) {
			missingPaths = append(missingPaths, path)
		}
	}

	if len(missingPaths) > 0 {
		return fmt.Errorf("cgroup %q has some missing paths: %v", name, strings.Join(missingPaths, ", "))
	}

	return nil
}

// Exists checks if all subsystem cgroups already exist
func (c *cgroupV1impl) Exists(name CgroupName) bool {
	return c.Validate(name) == nil
}

// MemoryUsage returns the current memory usage of the specified cgroup,
// as read from cgroupfs.
func (c *cgroupV1impl) MemoryUsage(name CgroupName) (int64, error) {
	var path, file string
	mp, ok := c.subsystems.MountPoints["memory"]
	if !ok { // should not happen
		return -1, errors.New("no cgroup v1 mountpoint for memory controller found")
	}
	path = mp + "/" + c.Name(name)
	file = "memory.usage_in_bytes"
	val, err := fscommon.GetCgroupParamUint(path, file)
	return int64(val), err
}

// Get the resource config values applied to the cgroup for specified resource type
func (c *cgroupV1impl) GetCgroupConfig(name CgroupName, resource v1.ResourceName) (*ResourceConfig, error) {
	cgroupPaths := c.buildCgroupPaths(name)
	cgroupResourcePath, found := cgroupPaths[string(resource)]
	if !found {
		return nil, fmt.Errorf("failed to build %v cgroup fs path for cgroup %v", resource, name)
	}
	switch resource {
	case v1.ResourceCPU:
		return c.getCgroupCPUConfig(cgroupResourcePath)
	case v1.ResourceMemory:
		return c.getCgroupMemoryConfig(cgroupResourcePath)
	}
	return nil, fmt.Errorf("unsupported resource %v for cgroup %v", resource, name)
}

// Set resource config for the specified resource type on the cgroup
func (c *cgroupV1impl) SetCgroupConfig(name CgroupName, resource v1.ResourceName, resourceConfig *ResourceConfig) error {
	cgroupPaths := c.buildCgroupPaths(name)
	cgroupResourcePath, found := cgroupPaths[string(resource)]
	if !found {
		return fmt.Errorf("failed to build %v cgroup fs path for cgroup %v", resource, name)
	}
	switch resource {
	case v1.ResourceCPU:
		return c.setCgroupCPUConfig(cgroupResourcePath, resourceConfig)
	case v1.ResourceMemory:
		return c.setCgroupMemoryConfig(cgroupResourcePath, resourceConfig)
	}
	return nil
}

func (c *cgroupV1impl) getCgroupCPUConfig(cgroupPath string) (*ResourceConfig, error) {
	cpuQuotaStr, errQ := fscommon.GetCgroupParamString(cgroupPath, "cpu.cfs_quota_us")
	if errQ != nil {
		return nil, fmt.Errorf("failed to read CPU quota for cgroup %v: %w", cgroupPath, errQ)
	}
	cpuQuota, errInt := strconv.ParseInt(cpuQuotaStr, 10, 64)
	if errInt != nil {
		return nil, fmt.Errorf("failed to convert CPU quota as integer for cgroup %v: %w", cgroupPath, errInt)
	}
	cpuPeriod, errP := fscommon.GetCgroupParamUint(cgroupPath, "cpu.cfs_period_us")
	if errP != nil {
		return nil, fmt.Errorf("failed to read CPU period for cgroup %v: %w", cgroupPath, errP)
	}
	cpuShares, errS := fscommon.GetCgroupParamUint(cgroupPath, "cpu.shares")
	if errS != nil {
		return nil, fmt.Errorf("failed to read CPU shares for cgroup %v: %w", cgroupPath, errS)
	}
	return &ResourceConfig{CPUShares: &cpuShares, CPUQuota: &cpuQuota, CPUPeriod: &cpuPeriod}, nil
}

func (c *cgroupV1impl) setCgroupCPUConfig(cgroupPath string, resourceConfig *ResourceConfig) error {
	var cpuQuotaStr, cpuPeriodStr, cpuSharesStr string
	if resourceConfig.CPUQuota != nil {
		cpuQuotaStr = strconv.FormatInt(*resourceConfig.CPUQuota, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.cfs_quota_us"), []byte(cpuQuotaStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %w", cpuQuotaStr, cgroupPath, err)
		}
	}
	if resourceConfig.CPUPeriod != nil {
		cpuPeriodStr = strconv.FormatUint(*resourceConfig.CPUPeriod, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.cfs_period_us"), []byte(cpuPeriodStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %w", cpuPeriodStr, cgroupPath, err)
		}
	}
	if resourceConfig.CPUShares != nil {
		cpuSharesStr = strconv.FormatUint(*resourceConfig.CPUShares, 10)
		if err := os.WriteFile(filepath.Join(cgroupPath, "cpu.shares"), []byte(cpuSharesStr), 0700); err != nil {
			return fmt.Errorf("failed to write %v to %v: %w", cpuSharesStr, cgroupPath, err)
		}
	}
	return nil
}

func (c *cgroupV1impl) setCgroupMemoryConfig(cgroupPath string, resourceConfig *ResourceConfig) error {
	return writeCgroupMemoryLimit(filepath.Join(cgroupPath, cgroupv1MemLimitFile), resourceConfig)
}

func (c *cgroupV1impl) getCgroupMemoryConfig(cgroupPath string) (*ResourceConfig, error) {
	return readCgroupMemoryConfig(cgroupPath, cgroupv1MemLimitFile)
}
