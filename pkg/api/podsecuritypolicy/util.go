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

package podsecuritypolicy

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledFields removes disabled fields from the pod security policy spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a pod security policy spec.
func DropDisabledFields(pspSpec, oldPSPSpec *policy.PodSecurityPolicySpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ProcMountType) && !allowedProcMountTypesInUse(oldPSPSpec) {
		pspSpec.AllowedProcMountTypes = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.RunAsGroup) && (oldPSPSpec == nil || oldPSPSpec.RunAsGroup == nil) {
		pspSpec.RunAsGroup = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.Sysctls) && !sysctlsInUse(oldPSPSpec) {
		pspSpec.AllowedUnsafeSysctls = nil
		pspSpec.ForbiddenSysctls = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIInlineVolume) {
		pspSpec.AllowedCSIDrivers = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.RuntimeClass) &&
		(oldPSPSpec == nil || oldPSPSpec.RuntimeClass == nil) {
		pspSpec.RuntimeClass = nil
	}
}

func allowedProcMountTypesInUse(oldPSPSpec *policy.PodSecurityPolicySpec) bool {
	if oldPSPSpec == nil {
		return false
	}

	if oldPSPSpec.AllowedProcMountTypes != nil {
		return true
	}

	return false

}

func sysctlsInUse(oldPSPSpec *policy.PodSecurityPolicySpec) bool {
	if oldPSPSpec == nil {
		return false
	}
	if oldPSPSpec.AllowedUnsafeSysctls != nil || oldPSPSpec.ForbiddenSysctls != nil {
		return true
	}
	return false
}
