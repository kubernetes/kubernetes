// +build !windows

package main

import (
	"io/ioutil"
	"path"

	"github.com/opencontainers/runc/libcontainer/cgroups"
)

var (
	CpuCfsPeriod = TestRequirement{
		func() bool {
			cgroupCpuMountpoint, err := cgroups.FindCgroupMountpoint("cpu")
			if err != nil {
				return false
			}
			if _, err := ioutil.ReadFile(path.Join(cgroupCpuMountpoint, "cpu.cfs_period_us")); err != nil {
				return false
			}
			return true
		},
		"Test requires an environment that supports cgroup cfs period.",
	}
	CpuCfsQuota = TestRequirement{
		func() bool {
			cgroupCpuMountpoint, err := cgroups.FindCgroupMountpoint("cpu")
			if err != nil {
				return false
			}
			if _, err := ioutil.ReadFile(path.Join(cgroupCpuMountpoint, "cpu.cfs_quota_us")); err != nil {
				return false
			}
			return true
		},
		"Test requires an environment that supports cgroup cfs quota.",
	}
	OomControl = TestRequirement{
		func() bool {
			cgroupMemoryMountpoint, err := cgroups.FindCgroupMountpoint("memory")
			if err != nil {
				return false
			}
			if _, err := ioutil.ReadFile(path.Join(cgroupMemoryMountpoint, "memory.memsw.limit_in_bytes")); err != nil {
				return false
			}

			if _, err = ioutil.ReadFile(path.Join(cgroupMemoryMountpoint, "memory.oom_control")); err != nil {
				return false
			}
			return true

		},
		"Test requires Oom control enabled.",
	}
)
