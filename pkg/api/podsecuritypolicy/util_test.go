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
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropAlphaProcMountType(t *testing.T) {
	// PodSecurityPolicy with AllowedProcMountTypes set
	psp := policy.PodSecurityPolicy{
		Spec: policy.PodSecurityPolicySpec{
			AllowedProcMountTypes: []api.ProcMountType{api.UnmaskedProcMount},
		},
	}

	// Enable alpha feature ProcMountType
	err1 := utilfeature.DefaultFeatureGate.Set("ProcMountType=true")
	if err1 != nil {
		t.Fatalf("Failed to enable feature gate for ProcMountType: %v", err1)
	}

	// now test dropping the fields - should not be dropped
	DropDisabledAlphaFields(&psp.Spec)

	// check to make sure AllowedProcMountTypes is still present
	// if featureset is set to true
	if utilfeature.DefaultFeatureGate.Enabled(features.ProcMountType) {
		if psp.Spec.AllowedProcMountTypes == nil {
			t.Error("AllowedProcMountTypes in pvc.Spec should not have been dropped based on feature-gate")
		}
	}

	// Disable alpha feature ProcMountType
	err := utilfeature.DefaultFeatureGate.Set("ProcMountType=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for ProcMountType: %v", err)
	}

	// now test dropping the fields
	DropDisabledAlphaFields(&psp.Spec)

	// check to make sure AllowedProcMountTypes is nil
	// if featureset is set to false
	if utilfeature.DefaultFeatureGate.Enabled(features.ProcMountType) {
		if psp.Spec.AllowedProcMountTypes != nil {
			t.Error("DropDisabledAlphaFields AllowedProcMountTypes for psp.Spec failed")
		}
	}
}
