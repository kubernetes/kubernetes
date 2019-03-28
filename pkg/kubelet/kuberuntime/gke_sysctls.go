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

package kuberuntime

import (
	"strings"

	v1 "k8s.io/api/core/v1"
)

// Collection of Linux kernel parameters (sysctls) that will be applied to
// the pods running on this node.
var PodSysctls = map[string]string{}

var validSysctlMap = map[string]bool{
	"kernel.msgmax":          true,
	"kernel.msgmnb":          true,
	"kernel.msgmni":          true,
	"kernel.sem":             true,
	"kernel.shmall":          true,
	"kernel.shmmax":          true,
	"kernel.shmmni":          true,
	"kernel.shm_rmid_forced": true,
}

func applyPodSysctls(sysctls, podSysctls map[string]string, pod *v1.Pod) {
	// From https://github.com/opencontainers/runc/blob/master/libcontainer/configs/validate/validator.go
	for k, v := range podSysctls {
		if validSysctlMap[k] || strings.HasPrefix(k, "fs.mqueue.") {
			if !pod.Spec.HostIPC {
				// Set the IPC parameters unless the pod runs in the host IPC
				// namespace.
				sysctls[k] = v
			}
			continue
		}
		if strings.HasPrefix(k, "net.") {
			if !pod.Spec.HostNetwork {
				// Set the networking parameters unless the pod runs in the
				// host network namespace.
				sysctls[k] = v
			}
			continue
		}
		// Docker creates per-container UTS namespace by default, and uses host
		// UTS namespace for host network.
		// Containerd creates per-pod UTS namespace by default, and uses host
		// UTS namespace for host network.
		if !pod.Spec.HostNetwork {
			switch k {
			case "kernel.domainname":
				// This is namespaced and there's no explicit OCI field for it.
				sysctls[k] = v
			case "kernel.hostname":
				// This is namespaced but there's a conflicting (dedicated) OCI
				// field for it.
			}
			continue
		}
	}
}
