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
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
)

const ipLocalReservedPortsMinNamespacedKernelVersion = "3.16"

var safeSysctls = []string{
	"kernel.shm_rmid_forced",
	"net.ipv4.ip_local_port_range",
	"net.ipv4.tcp_syncookies",
	"net.ipv4.ping_group_range",
	"net.ipv4.ip_unprivileged_port_start",
}

// SafeSysctlAllowlist returns the allowlist of safe sysctls and safe sysctl patterns (ending in *).
//
// A sysctl is called safe iff
// - it is namespaced in the container or the pod
// - it is isolated, i.e. has no influence on any other pod on the same node.
func SafeSysctlAllowlist() []string {
	kernelVersionStr, err := ipvs.NewLinuxKernelHandler().GetKernelVersion()
	if err != nil {
		klog.ErrorS(err, "Failed to get kernel version.")
		return safeSysctls
	}
	kernelVersion, err := version.ParseGeneric(kernelVersionStr)
	if err != nil {
		klog.ErrorS(err, "Failed to parse kernel version.")
		return safeSysctls
	}
	// ip_local_reserved_ports has been changed to namesapced since kernel v3.16.
	// refer to https://github.com/torvalds/linux/commit/122ff243f5f104194750ecbc76d5946dd1eec934.
	if kernelVersion.LessThan(version.MustParseGeneric(ipLocalReservedPortsMinNamespacedKernelVersion)) {
		return safeSysctls
	}
	return []string{
		"kernel.shm_rmid_forced",
		"net.ipv4.ip_local_port_range",
		"net.ipv4.tcp_syncookies",
		"net.ipv4.ping_group_range",
		"net.ipv4.ip_unprivileged_port_start",
		"net.ipv4.ip_local_reserved_ports",
	}
}
