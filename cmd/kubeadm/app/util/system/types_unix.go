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
	"os/exec"
	"strings"
)

// dockerEndpoint is the os specific endpoint for docker communication
const dockerEndpoint = "unix:///var/run/docker.sock"

// DefaultSysSpec is the default SysSpec for Linux
var DefaultSysSpec = SysSpec{
	OS: "Linux",
	KernelSpec: KernelSpec{
		Versions: []string{`3\.[1-9][0-9].*`, `4\..*`}, // Requires 3.10+ or 4+
		// TODO(random-liu): Add more config
		// TODO(random-liu): Add description for each kernel configuration:
		Required: []KernelConfig{
			{Name: "NAMESPACES"},
			{Name: "NET_NS"},
			{Name: "PID_NS"},
			{Name: "IPC_NS"},
			{Name: "UTS_NS"},
			{Name: "CGROUPS"},
			{Name: "CGROUP_CPUACCT"},
			{Name: "CGROUP_DEVICE"},
			{Name: "CGROUP_FREEZER"},
			{Name: "CGROUP_SCHED"},
			{Name: "CPUSETS"},
			{Name: "MEMCG"},
			{Name: "INET"},
			{Name: "EXT4_FS"},
			{Name: "PROC_FS"},
			{Name: "NETFILTER_XT_TARGET_REDIRECT", Aliases: []string{"IP_NF_TARGET_REDIRECT"}},
			{Name: "NETFILTER_XT_MATCH_COMMENT"},
		},
		Optional: []KernelConfig{
			{Name: "OVERLAY_FS", Aliases: []string{"OVERLAYFS_FS"}, Description: "Required for overlayfs."},
			{Name: "AUFS_FS", Description: "Required for aufs."},
			{Name: "BLK_DEV_DM", Description: "Required for devicemapper."},
		},
		Forbidden: []KernelConfig{},
	},
	Cgroups: []string{"cpu", "cpuacct", "cpuset", "devices", "freezer", "memory"},
	RuntimeSpec: RuntimeSpec{
		DockerSpec: &DockerSpec{
			Version:     []string{`1\.1[1-3]\..*`, `17\.0[3,6,9]\..*`, `18\.06\..*`},
			GraphDriver: []string{"aufs", "overlay", "overlay2", "devicemapper", "zfs"},
		},
	},
}

// KernelValidatorHelperImpl is the 'linux' implementation of KernelValidatorHelper
type KernelValidatorHelperImpl struct{}

var _ KernelValidatorHelper = &KernelValidatorHelperImpl{}

// GetKernelReleaseVersion returns the kernel release version (ex. 4.4.0-96-generic) as a string
func (o *KernelValidatorHelperImpl) GetKernelReleaseVersion() (string, error) {
	releaseVersion, err := exec.Command("uname", "-r").CombinedOutput()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(releaseVersion)), nil
}
