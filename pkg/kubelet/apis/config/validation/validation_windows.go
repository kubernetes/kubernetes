//go:build windows
// +build windows

/*
Copyright 2018 The Kubernetes Authors.

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

package validation

import (
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// validateKubeletOSConfiguration validates os specific kubelet configuration and returns an error if it is invalid.
func validateKubeletOSConfiguration(kc *kubeletconfig.KubeletConfiguration, warningf func(string, ...interface{})) error {
	message := "invalid configuration: %v (%v) %v is not supported on Windows"

	if kc.CgroupsPerQOS {
		warningf(message, "CgroupsPerQOS", "--cgroups-per-qos", kc.CgroupsPerQOS)
	}

	if len(kc.EnforceNodeAllocatable) > 0 {
		warningf(message, "EnforceNodeAllocatable", "--enforce-node-allocatable", kc.EnforceNodeAllocatable)
	}

	return nil
}
