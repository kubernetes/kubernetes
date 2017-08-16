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

package validation

import (
	"fmt"

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	containermanager "k8s.io/kubernetes/pkg/kubelet/cm"
)

// MaxCrashLoopThreshold is the maximum allowed KubeletConfiguraiton.CrashLoopThreshold
const MaxCrashLoopThreshold = 10

// ValidateKubeletConfiguration validates `kc` and returns an error if it is invalid
func ValidateKubeletConfiguration(kc *kubeletconfig.KubeletConfiguration) error {
	// restrict crashloop threshold to between 0 and `maxCrashLoopThreshold`, inclusive
	// more than `maxStartups=maxCrashLoopThreshold` adds unnecessary bloat to the .startups.json file,
	// and negative values would be silly.
	if kc.CrashLoopThreshold < 0 || kc.CrashLoopThreshold > MaxCrashLoopThreshold {
		return fmt.Errorf("field `CrashLoopThreshold` must be between 0 and %d, inclusive", MaxCrashLoopThreshold)
	}

	if !kc.CgroupsPerQOS && len(kc.EnforceNodeAllocatable) > 0 {
		return fmt.Errorf("node allocatable enforcement is not supported unless Cgroups Per QOS feature is turned on")
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
