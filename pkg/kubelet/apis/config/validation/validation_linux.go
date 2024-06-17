//go:build linux
// +build linux

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

package validation

import (
	"fmt"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

// validateKubeletOSConfiguration validates os specific kubelet configuration and returns an error if it is invalid.
func validateKubeletOSConfiguration(kc *kubeletconfig.KubeletConfiguration) error {
	if kc.FailCgroupV1 && !libcontainercgroups.IsCgroup2UnifiedMode() {
		return fmt.Errorf("kubelet is configured to not run on a host using cgroup v1. cgroup v1 support is in maintenance mode")
	}

	return nil
}
