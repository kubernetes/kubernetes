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
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropAllowedProcMountTypes(t *testing.T) {
	allowedProcMountTypes := []api.ProcMountType{api.UnmaskedProcMount}
	scWithoutAllowedProcMountTypes := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{}
	}
	scWithAllowedProcMountTypes := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{
			AllowedProcMountTypes: allowedProcMountTypes,
		}
	}

	scInfo := []struct {
		description              string
		hasAllowedProcMountTypes bool
		sc                       func() *policy.PodSecurityPolicySpec
	}{
		{
			description:              "PodSecurityPolicySpec Without AllowedProcMountTypes",
			hasAllowedProcMountTypes: false,
			sc:                       scWithoutAllowedProcMountTypes,
		},
		{
			description:              "PodSecurityPolicySpec With AllowedProcMountTypes",
			hasAllowedProcMountTypes: true,
			sc:                       scWithAllowedProcMountTypes,
		},
		{
			description:              "is nil",
			hasAllowedProcMountTypes: false,
			sc:                       func() *policy.PodSecurityPolicySpec { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPSPSpecInfo := range scInfo {
			for _, newPSPSpecInfo := range scInfo {
				oldPSPSpecHasAllowedProcMountTypes, oldPSPSpec := oldPSPSpecInfo.hasAllowedProcMountTypes, oldPSPSpecInfo.sc()
				newPSPSpecHasAllowedProcMountTypes, newPSPSpec := newPSPSpecInfo.hasAllowedProcMountTypes, newPSPSpecInfo.sc()
				if newPSPSpec == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old PodSecurityPolicySpec %v, new PodSecurityPolicySpec %v", enabled, oldPSPSpecInfo.description, newPSPSpecInfo.description), func(t *testing.T) {
					defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProcMountType, enabled)()

					DropDisabledFields(newPSPSpec, oldPSPSpec)

					// old PodSecurityPolicySpec should never be changed
					if !reflect.DeepEqual(oldPSPSpec, oldPSPSpecInfo.sc()) {
						t.Errorf("old PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(oldPSPSpec, oldPSPSpecInfo.sc()))
					}

					switch {
					case enabled || oldPSPSpecHasAllowedProcMountTypes:
						// new PodSecurityPolicySpec should not be changed if the feature is enabled, or if the old PodSecurityPolicySpec had AllowedProcMountTypes
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.sc()))
						}
					case newPSPSpecHasAllowedProcMountTypes:
						// new PodSecurityPolicySpec should be changed
						if reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec was not changed")
						}
						// new PodSecurityPolicySpec should not have AllowedProcMountTypes
						if !reflect.DeepEqual(newPSPSpec, scWithoutAllowedProcMountTypes()) {
							t.Errorf("new PodSecurityPolicySpec had PodSecurityPolicySpecAllowedProcMountTypes: %v", diff.ObjectReflectDiff(newPSPSpec, scWithoutAllowedProcMountTypes()))
						}
					default:
						// new PodSecurityPolicySpec should not need to be changed
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.sc()))
						}
					}
				})
			}
		}
	}
}

func TestDropRunAsGroup(t *testing.T) {
	group := func() *policy.RunAsGroupStrategyOptions {
		return &policy.RunAsGroupStrategyOptions{}
	}
	scWithoutRunAsGroup := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{}
	}
	scWithRunAsGroup := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{
			RunAsGroup: group(),
		}
	}
	scInfo := []struct {
		description   string
		hasRunAsGroup bool
		sc            func() *policy.PodSecurityPolicySpec
	}{
		{
			description:   "PodSecurityPolicySpec Without RunAsGroup",
			hasRunAsGroup: false,
			sc:            scWithoutRunAsGroup,
		},
		{
			description:   "PodSecurityPolicySpec With RunAsGroup",
			hasRunAsGroup: true,
			sc:            scWithRunAsGroup,
		},
		{
			description:   "is nil",
			hasRunAsGroup: false,
			sc:            func() *policy.PodSecurityPolicySpec { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPSPSpecInfo := range scInfo {
			for _, newPSPSpecInfo := range scInfo {
				oldPSPSpecHasRunAsGroup, oldPSPSpec := oldPSPSpecInfo.hasRunAsGroup, oldPSPSpecInfo.sc()
				newPSPSpecHasRunAsGroup, newPSPSpec := newPSPSpecInfo.hasRunAsGroup, newPSPSpecInfo.sc()
				if newPSPSpec == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old PodSecurityPolicySpec %v, new PodSecurityPolicySpec %v", enabled, oldPSPSpecInfo.description, newPSPSpecInfo.description), func(t *testing.T) {
					defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RunAsGroup, enabled)()

					DropDisabledFields(newPSPSpec, oldPSPSpec)

					// old PodSecurityPolicySpec should never be changed
					if !reflect.DeepEqual(oldPSPSpec, oldPSPSpecInfo.sc()) {
						t.Errorf("old PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(oldPSPSpec, oldPSPSpecInfo.sc()))
					}

					switch {
					case enabled || oldPSPSpecHasRunAsGroup:
						// new PodSecurityPolicySpec should not be changed if the feature is enabled, or if the old PodSecurityPolicySpec had RunAsGroup
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.sc()))
						}
					case newPSPSpecHasRunAsGroup:
						// new PodSecurityPolicySpec should be changed
						if reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec was not changed")
						}
						// new PodSecurityPolicySpec should not have RunAsGroup
						if !reflect.DeepEqual(newPSPSpec, scWithoutRunAsGroup()) {
							t.Errorf("new PodSecurityPolicySpec had RunAsGroup: %v", diff.ObjectReflectDiff(newPSPSpec, scWithoutRunAsGroup()))
						}
					default:
						// new PodSecurityPolicySpec should not need to be changed
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.sc()))
						}
					}
				})
			}
		}
	}
}
