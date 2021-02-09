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
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProcMountType, enabled)()

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

func TestDropEphemeralVolumeType(t *testing.T) {
	allowedVolumeTypes := []policy.FSType{policy.Ephemeral}
	pspWithoutGenericVolume := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{}
	}
	pspWithGenericVolume := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{
			Volumes: allowedVolumeTypes,
		}
	}

	pspInfo := []struct {
		description      string
		hasGenericVolume bool
		psp              func() *policy.PodSecurityPolicySpec
	}{
		{
			description:      "PodSecurityPolicySpec Without GenericVolume",
			hasGenericVolume: false,
			psp:              pspWithoutGenericVolume,
		},
		{
			description:      "PodSecurityPolicySpec With GenericVolume",
			hasGenericVolume: true,
			psp:              pspWithGenericVolume,
		},
		{
			description:      "is nil",
			hasGenericVolume: false,
			psp:              func() *policy.PodSecurityPolicySpec { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPSPSpecInfo := range pspInfo {
			for _, newPSPSpecInfo := range pspInfo {
				oldPSPSpecHasGenericVolume, oldPSPSpec := oldPSPSpecInfo.hasGenericVolume, oldPSPSpecInfo.psp()
				newPSPSpecHasGenericVolume, newPSPSpec := newPSPSpecInfo.hasGenericVolume, newPSPSpecInfo.psp()
				if newPSPSpec == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old PodSecurityPolicySpec %v, new PodSecurityPolicySpec %v", enabled, oldPSPSpecInfo.description, newPSPSpecInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericEphemeralVolume, enabled)()

					DropDisabledFields(newPSPSpec, oldPSPSpec)

					// old PodSecurityPolicySpec should never be changed
					if !reflect.DeepEqual(oldPSPSpec, oldPSPSpecInfo.psp()) {
						t.Errorf("old PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(oldPSPSpec, oldPSPSpecInfo.psp()))
					}

					switch {
					case enabled || oldPSPSpecHasGenericVolume:
						// new PodSecurityPolicySpec should not be changed if the feature is enabled, or if the old PodSecurityPolicySpec had GenericVolume
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.psp()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.psp()))
						}
					case newPSPSpecHasGenericVolume:
						// new PodSecurityPolicySpec should be changed
						if reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.psp()) {
							t.Errorf("new PodSecurityPolicySpec was not changed")
						}
						// new PodSecurityPolicySpec should not have GenericVolume
						if !reflect.DeepEqual(newPSPSpec, pspWithoutGenericVolume()) {
							t.Errorf("new PodSecurityPolicySpec had PodSecurityPolicySpecGenericVolume: %v", diff.ObjectReflectDiff(newPSPSpec, pspWithoutGenericVolume()))
						}
					default:
						// new PodSecurityPolicySpec should not need to be changed
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.psp()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.psp()))
						}
					}
				})
			}
		}
	}
}
