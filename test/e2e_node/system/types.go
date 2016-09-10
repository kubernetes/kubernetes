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

// KernelConfig is the configuration required or forbidden
// for kubernetes.
// TODO(random-liu): Add Optional, and report warning for optional config.
type KernelConfig struct {
	// Required is all kernel configurations required to be enabled (built in
	// or as module).
	Required []string
	// Forbidden is all kernel configurations forbidden (disabled or not set)
	Forbidden []string
}

// DockerSpec defines the requirement configuration for docker. Currently, it only
// contains spec for graph driver.
type DockerSpec struct {
	// Version is a group of regex matching supported docker versions.
	Version []string
	// GraphDriver is the graph drivers supported by kubelet.
	GraphDriver []string
}

// RuntimeSpec is the abstract layer for different runtimes. Different runtimes
// should put their spec inside the RuntimeSpec.
type RuntimeSpec struct {
	*DockerSpec
}

// SysSpec defines the requirement of supported system. Currently, it only contains
// spec for OS, Kernel and Cgroups.
type SysSpec struct {
	// OS is the operating system of the SysSpec.
	OS string
	// KernelVersion is a group of regex matching supported kernel versions.
	KernelVersion []string
	// KernelConfig defines the kernel configurations which are required or
	// forbidden.
	KernelConfig KernelConfig
	// Cgroups is the required cgroups.
	Cgroups []string
	// RuntimeSpec defines the spec for runtime.
	RuntimeSpec RuntimeSpec
}

// DefaultSysSpec is the default SysSpec.
var DefaultSysSpec = SysSpec{
	OS:            "Linux",
	KernelVersion: []string{`3\.[1-9][0-9].*`, `4\..*`}, // Requires 3.10+ or 4+
	// TODO(random-liu): Add more config
	KernelConfig: KernelConfig{
		Required: []string{
			"NAMESPACES", "NET_NS", "PID_NS", "IPC_NS", "UTS_NS",
			"CGROUPS", "CGROUP_CPUACCT", "CGROUP_DEVICE", "CGROUP_FREEZER",
			"CGROUP_SCHED", "CPUSETS", "MEMCG",
		},
		Forbidden: []string{},
	},
	Cgroups: []string{"cpu", "cpuacct", "cpuset", "devices", "freezer", "memory"},
	RuntimeSpec: RuntimeSpec{
		DockerSpec: &DockerSpec{
			Version: []string{`1\.(9|\d{2,})\..*`}, // Requires 1.9+
			// TODO(random-liu): Validate overlay2.
			GraphDriver: []string{"aufs", "overlay", "devicemapper"},
		},
	},
}
