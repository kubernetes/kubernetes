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

// DropDisabledAlphaFields removes disabled fields from the pod security policy spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a od security policy spec.
func DropDisabledAlphaFields(pspSpec *policy.PodSecurityPolicySpec) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ProcMountType) {
		pspSpec.AllowedProcMountTypes = nil
	}
	if !utilfeature.DefaultFeatureGate.Enabled(features.RunAsGroup) {
		pspSpec.RunAsGroup = nil
	}
}
