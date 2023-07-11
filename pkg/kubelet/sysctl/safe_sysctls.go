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

package sysctl

import (
	"fmt"
	goruntime "runtime"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
)

type sysctl struct {
	// the name of sysctl
	name string
	// the minimum kernel version where the sysctl is available
	kernel string
}

var safeSysctls = []sysctl{
	{
		name: "kernel.shm_rmid_forced",
	}, {
		name: "net.ipv4.ip_local_port_range",
	}, {
		name: "net.ipv4.tcp_syncookies",
	}, {
		name: "net.ipv4.ping_group_range",
	}, {
		name: "net.ipv4.ip_unprivileged_port_start",
	}, {
		name: "net.ipv4.ip_local_reserved_ports",
		// refer to https://github.com/torvalds/linux/commit/122ff243f5f104194750ecbc76d5946dd1eec934.
		kernel: "3.16",
	}, {
		name: "net.ipv4.tcp_keepalive_time",
		// refer to https://github.com/torvalds/linux/commit/13b287e8d1cad951634389f85b8c9b816bd3bb1e.
		kernel: "4.5",
	},
}

// SafeSysctlAllowlist returns the allowlist of safe sysctls and safe sysctl patterns (ending in *).
//
// A sysctl is called safe iff
// - it is namespaced in the container or the pod
// - it is isolated, i.e. has no influence on any other pod on the same node.
func SafeSysctlAllowlist() []string {
	if goruntime.GOOS != "linux" {
		return nil
	}
	return getSafeSysctlAllowlist(getKernelVersion)
}

func getSafeSysctlAllowlist(getVersion func() (*version.Version, error)) []string {
	kernelVersion, err := getVersion()
	if err != nil {
		klog.ErrorS(err, "failed to get kernel version, unable to determine which sysctls are available")
	}

	var safeSysctlAllowlist []string
	for _, sc := range safeSysctls {
		if sc.kernel == "" {
			safeSysctlAllowlist = append(safeSysctlAllowlist, sc.name)
			continue
		}

		if kernelVersion != nil && kernelVersion.AtLeast(version.MustParseGeneric(sc.kernel)) {
			safeSysctlAllowlist = append(safeSysctlAllowlist, sc.name)
		} else {
			klog.ErrorS(nil, "kernel version is too small, dropping the sysctl from safe sysctl list", "kernelVersion", kernelVersion, "sysctl", sc.name)
		}
	}
	return safeSysctlAllowlist
}

func getKernelVersion() (*version.Version, error) {
	kernelVersionStr, err := ipvs.NewLinuxKernelHandler().GetKernelVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get kernel version: %w", err)
	}

	kernelVersion, err := version.ParseGeneric(kernelVersionStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse kernel version: %w", err)
	}
	return kernelVersion, nil
}
