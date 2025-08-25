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
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// validateKubeletOSConfiguration validates os specific kubelet configuration and returns an error if it is invalid.
func validateKubeletOSConfiguration(kc *kubeletconfig.KubeletConfiguration) error {
	message := "ignored configuration option: %v (%v) %v is not supported on Windows"

	if kc.CgroupsPerQOS {
		klog.Warningf(message, "CgroupsPerQOS", "--cgroups-per-qos", kc.CgroupsPerQOS)
	}

	if kc.SingleProcessOOMKill != nil {
		return fmt.Errorf("invalid configuration: singleProcessOOMKill is not supported on Windows")
	}

	enforceNodeAllocatableWithoutNone := sets.New(kc.EnforceNodeAllocatable...).Delete(kubetypes.NodeAllocatableNoneKey)
	if len(enforceNodeAllocatableWithoutNone) > 0 {
		klog.Warningf(message, "EnforceNodeAllocatable", "--enforce-node-allocatable", kc.EnforceNodeAllocatable)
	}

	if kc.UserNamespaces != nil {
		return fmt.Errorf("invalid configuration: userNamespaces is not supported on Windows")
	}

	return nil
}
