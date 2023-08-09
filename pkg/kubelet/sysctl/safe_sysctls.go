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

// SafeSysctlAllowlist returns the allowlist of safe sysctls and safe sysctl patterns (ending in *).
//
// A sysctl is called safe iff
// - it is namespaced in the container or the pod
// - it is isolated, i.e. has no influence on any other pod on the same node.
func SafeSysctlAllowlist() []string {
	return []string{
		"kernel.shm_rmid_forced",
		"net.ipv4.ip_local_port_range",
		"net.ipv4.tcp_syncookies",
		"net.ipv4.ping_group_range",
		"net.ipv4.ip_unprivileged_port_start",
	}
}
