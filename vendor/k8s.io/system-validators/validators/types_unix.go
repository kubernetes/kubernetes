//go:build !windows
// +build !windows

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
	"fmt"
	"strings"

	"golang.org/x/sys/unix"
)

// DefaultSysSpec is the default SysSpec for Linux
var DefaultSysSpec = SysSpec{
	OS: "Linux",
	KernelSpec: KernelSpec{
		// 5.4, 5.10, 5.15 is an active kernel Long Term Support (LTS) release, tracked in https://www.kernel.org/category/releases.html.
		Versions:     []string{`^5\.4.*$`, `^5\.10.*$`, `^5\.15.*$`, `^([6-9]|[1-9][0-9]+)\.([0-9]+)\.([0-9]+).*$`},
		VersionsNote: "Supported LTS versions from the 5.x series are 5.4, 5.10 and 5.15. Any 6.x version is also supported. For cgroups v2 support, the recommended version is 5.10 or newer",
		// TODO(random-liu): Add more config
		// TODO(random-liu): Add description for each kernel configuration:
		Required: []KernelConfig{
			{Name: "NAMESPACES"},
			{Name: "NET_NS"},
			{Name: "PID_NS"},
			{Name: "IPC_NS"},
			{Name: "UTS_NS"},
			{Name: "CGROUPS"},
			{Name: "CGROUP_BPF"},     // cgroups v2
			{Name: "CGROUP_CPUACCT"}, // cgroups v1 cpuacct
			{Name: "CGROUP_DEVICE"},
			{Name: "CGROUP_FREEZER"}, // cgroups v1 freezer
			{Name: "CGROUP_PIDS"},
			{Name: "CGROUP_SCHED"}, // cgroups v1 & v2 cpu
			{Name: "CPUSETS"},
			{Name: "MEMCG"},
			{Name: "INET"},
			{Name: "EXT4_FS"},
			{Name: "PROC_FS"},
			{Name: "NETFILTER_XT_TARGET_REDIRECT", Aliases: []string{"IP_NF_TARGET_REDIRECT"}},
			{Name: "NETFILTER_XT_MATCH_COMMENT"},
			{Name: "FAIR_GROUP_SCHED"},
		},
		Optional: []KernelConfig{
			{Name: "OVERLAY_FS", Aliases: []string{"OVERLAYFS_FS"}, Description: "Required for overlayfs."},
			{Name: "AUFS_FS", Description: "Required for aufs."},
			{Name: "BLK_DEV_DM", Description: "Required for devicemapper."},
			{Name: "CFS_BANDWIDTH", Description: "Required for CPU quota."},
			{Name: "CGROUP_HUGETLB", Description: "Required for hugetlb cgroup."},
			{Name: "SECCOMP", Description: "Required for seccomp."},
			{Name: "SECCOMP_FILTER", Description: "Required for seccomp mode 2."},
		},
		Forbidden: []KernelConfig{},
	},
	Cgroups: []string{"cpu", "cpuacct", "cpuset", "devices", "freezer", "memory", "pids"},
	CgroupsOptional: []string{
		// The hugetlb cgroup is optional since some kernels are compiled without support for huge pages
		// and therefore lacks corresponding hugetlb cgroup
		"hugetlb",
		// The blkio cgroup is optional since some kernels are compiled without support for block I/O throttling.
		// Containerd and cri-o will use blkio to track disk I/O and throttling in both cgroups v1 and v2.
		"blkio",
	},
	CgroupsV2: []string{"cpu", "cpuset", "devices", "freezer", "memory", "pids"},
	CgroupsV2Optional: []string{
		"hugetlb",
		// The cgroups v2 io controller is the successor of the v1 blkio controller.
		"io",
	},
	RuntimeSpec: RuntimeSpec{
		DockerSpec: &DockerSpec{
			Version:     []string{`1\.1[1-3]\..*`, `17\.0[3,6,9]\..*`, `18\.0[6,9]\..*`, `19\.03\..*`, `20\.10\..*`},
			GraphDriver: []string{"aufs", "btrfs", "overlay", "overlay2", "devicemapper", "zfs"},
		},
	},
}

// KernelValidatorHelperImpl is the 'linux' implementation of KernelValidatorHelper
type KernelValidatorHelperImpl struct{}

var _ KernelValidatorHelper = &KernelValidatorHelperImpl{}

// GetKernelReleaseVersion returns the kernel release version (ex. 4.4.0-96-generic) as a string
func (o *KernelValidatorHelperImpl) GetKernelReleaseVersion() (string, error) {
	return getKernelRelease()
}

// getKernelRelease returns the kernel release of the local machine.
func getKernelRelease() (string, error) {
	var utsname unix.Utsname
	err := unix.Uname(&utsname)
	if err != nil {
		return "", fmt.Errorf("failed to get kernel release: %w", err)
	}
	return strings.TrimSpace(unix.ByteSliceToString(utsname.Release[:])), nil
}
