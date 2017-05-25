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

package nodeconfig

import (
	"fmt"

	ccv1a1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	containermanager "k8s.io/kubernetes/pkg/kubelet/cm"
)

// validateConfig checks for invalid configuration and returns an error if `kc` fails validation
// TODO(mtaufen): keep this up to date with cmd/kubelet/app/server.go, until the nodeconfig controller
// goes GA, we maintain an extended copy of the validation in cmd/kubelet/app/server.go:validateConfig here.
// This is because, today, the controller is only ever used if dynamic config is enabled, so validation with the
// feature off temporarily needs to live in another codepath.
func validateConfig(kc *ccv1a1.KubeletConfiguration) error {
	// restrict crashloop threshold to between 2 and `maxStartups`, inclusive
	// more than `maxStartups` adds unnecessary bloat to the .startups.json file, and negative values would be silly
	if *kc.CrashLoopThreshold < 0 || *kc.CrashLoopThreshold > maxCrashLoopThreshold {
		return fmt.Errorf("CrashLoopThreshold must be between 2 and %d, inclusive.", maxStartups)
	}

	// validation from server.go
	if !*kc.CgroupsPerQOS && len(kc.EnforceNodeAllocatable) > 0 {
		return fmt.Errorf("Node Allocatable enforcement is not supported unless Cgroups Per QOS feature is turned on")
	}
	if kc.SystemCgroups != "" && kc.CgroupRoot == "" {
		return fmt.Errorf("invalid configuration: system container was specified and cgroup root was not specified")
	}
	for _, val := range kc.EnforceNodeAllocatable {
		switch val {
		case containermanager.NodeAllocatableEnforcementKey:
		case containermanager.SystemReservedEnforcementKey:
		case containermanager.KubeReservedEnforcementKey:
			continue
		default:
			return fmt.Errorf("invalid option %q specified for EnforceNodeAllocatable setting. Valid options are %q, %q or %q",
				val, containermanager.NodeAllocatableEnforcementKey, containermanager.SystemReservedEnforcementKey, containermanager.KubeReservedEnforcementKey)
		}
	}
	return nil
}
