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
	"math"

	libcontainercgroups "github.com/opencontainers/cgroups"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/utils/ptr"
)

const userNsUnitLength = 65536

// validateKubeletOSConfiguration validates os specific kubelet configuration and returns an error if it is invalid.
func validateKubeletOSConfiguration(kc *kubeletconfig.KubeletConfiguration) error {
	isCgroup1 := !libcontainercgroups.IsCgroup2UnifiedMode()
	if kc.FailCgroupV1 && isCgroup1 {
		return fmt.Errorf("kubelet is configured to not run on a host using cgroup v1. cgroup v1 support is in maintenance mode")
	}

	if isCgroup1 && kc.SingleProcessOOMKill != nil && !ptr.Deref(kc.SingleProcessOOMKill, true) {
		return fmt.Errorf("invalid configuration: singleProcessOOMKill must not be explicitly set to false when using cgroup v1")
	}

	if userNs := kc.UserNamespaces; userNs != nil {
		if idsPerPod := userNs.IDsPerPod; idsPerPod != nil {
			if *idsPerPod < userNsUnitLength {
				return fmt.Errorf("invalid configuration: userNamespaces.idsPerPod must not be less than %d", userNsUnitLength)
			}
			if *idsPerPod%userNsUnitLength != 0 {
				return fmt.Errorf("invalid configuration: userNamespaces.idsPerPod must be a multiple of %d", userNsUnitLength)
			}
			if *idsPerPod > math.MaxUint32 {
				// int64() is needed for 32-bit targets
				return fmt.Errorf("invalid configuration: userNamespaces.idsPerPod must not be more than %d", int64(math.MaxUint32))
			}
		}
	}

	return nil
}
