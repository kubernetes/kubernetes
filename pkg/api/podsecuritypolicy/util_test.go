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

	"github.com/stretchr/testify/assert"

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
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RunAsGroup, enabled)()

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

func TestDropSysctls(t *testing.T) {
	scWithSysctls := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{
			AllowedUnsafeSysctls: []string{"foo/*"},
			ForbiddenSysctls:     []string{"bar.*"},
		}
	}
	scWithOneSysctls := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{
			AllowedUnsafeSysctls: []string{"foo/*"},
		}
	}
	scWithoutSysctls := func() *policy.PodSecurityPolicySpec {
		return &policy.PodSecurityPolicySpec{}
	}

	scInfo := []struct {
		description string
		hasSysctls  bool
		sc          func() *policy.PodSecurityPolicySpec
	}{
		{
			description: "has Sysctls",
			hasSysctls:  true,
			sc:          scWithSysctls,
		},
		{
			description: "has one Sysctl",
			hasSysctls:  true,
			sc:          scWithOneSysctls,
		},
		{
			description: "does not have Sysctls",
			hasSysctls:  false,
			sc:          scWithoutSysctls,
		},
		{
			description: "is nil",
			hasSysctls:  false,
			sc:          func() *policy.PodSecurityPolicySpec { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPSPSpecInfo := range scInfo {
			for _, newPSPSpecInfo := range scInfo {
				oldPSPSpecHasSysctls, oldPSPSpec := oldPSPSpecInfo.hasSysctls, oldPSPSpecInfo.sc()
				newPSPSpecHasSysctls, newPSPSpec := newPSPSpecInfo.hasSysctls, newPSPSpecInfo.sc()
				if newPSPSpec == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old PodSecurityPolicySpec %v, new PodSecurityPolicySpec %v", enabled, oldPSPSpecInfo.description, newPSPSpecInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.Sysctls, enabled)()

					DropDisabledFields(newPSPSpec, oldPSPSpec)

					// old PodSecurityPolicySpec should never be changed
					if !reflect.DeepEqual(oldPSPSpec, oldPSPSpecInfo.sc()) {
						t.Errorf("old PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(oldPSPSpec, oldPSPSpecInfo.sc()))
					}

					switch {
					case enabled || oldPSPSpecHasSysctls:
						// new PodSecurityPolicySpec should not be changed if the feature is enabled, or if the old PodSecurityPolicySpec had Sysctls
						if !reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec changed: %v", diff.ObjectReflectDiff(newPSPSpec, newPSPSpecInfo.sc()))
						}
					case newPSPSpecHasSysctls:
						// new PodSecurityPolicySpec should be changed
						if reflect.DeepEqual(newPSPSpec, newPSPSpecInfo.sc()) {
							t.Errorf("new PodSecurityPolicySpec was not changed")
						}
						// new PodSecurityPolicySpec should not have Sysctls
						if !reflect.DeepEqual(newPSPSpec, scWithoutSysctls()) {
							t.Errorf("new PodSecurityPolicySpec had Sysctls: %v", diff.ObjectReflectDiff(newPSPSpec, scWithoutSysctls()))
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

func TestDropRuntimeClass(t *testing.T) {
	type testcase struct {
		name                string
		featureEnabled      bool
		pspSpec, oldPSPSpec *policy.PodSecurityPolicySpec
		expectRuntimeClass  bool
	}
	tests := []testcase{}
	pspGenerator := func(withRuntimeClass bool) *policy.PodSecurityPolicySpec {
		psp := &policy.PodSecurityPolicySpec{}
		if withRuntimeClass {
			psp.RuntimeClass = &policy.RuntimeClassStrategyOptions{
				AllowedRuntimeClassNames: []string{policy.AllowAllRuntimeClassNames},
			}
		}
		return psp
	}
	for _, enabled := range []bool{true, false} {
		for _, hasRuntimeClass := range []bool{true, false} {
			tests = append(tests, testcase{
				name:               fmt.Sprintf("create feature:%t hasRC:%t", enabled, hasRuntimeClass),
				featureEnabled:     enabled,
				pspSpec:            pspGenerator(hasRuntimeClass),
				expectRuntimeClass: enabled && hasRuntimeClass,
			})
			for _, hadRuntimeClass := range []bool{true, false} {
				tests = append(tests, testcase{
					name:               fmt.Sprintf("update feature:%t hasRC:%t hadRC:%t", enabled, hasRuntimeClass, hadRuntimeClass),
					featureEnabled:     enabled,
					pspSpec:            pspGenerator(hasRuntimeClass),
					oldPSPSpec:         pspGenerator(hadRuntimeClass),
					expectRuntimeClass: hasRuntimeClass && (enabled || hadRuntimeClass),
				})
			}
		}
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClass, test.featureEnabled)()

			DropDisabledFields(test.pspSpec, test.oldPSPSpec)

			if test.expectRuntimeClass {
				assert.NotNil(t, test.pspSpec.RuntimeClass)
			} else {
				assert.Nil(t, test.pspSpec.RuntimeClass)
			}
		})
	}
}
