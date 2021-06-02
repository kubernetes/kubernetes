/*
Copyright 2021 The Kubernetes Authors.

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
	"context"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/features"
)

func TestAllowEphemeralVolumeType(t *testing.T) {
	pspWithoutGenericVolume := func() *policy.PodSecurityPolicy {
		return &policy.PodSecurityPolicy{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "psp",
				ResourceVersion: "1",
			},
			Spec: policy.PodSecurityPolicySpec{
				RunAsUser: policy.RunAsUserStrategyOptions{
					Rule: policy.RunAsUserStrategyMustRunAs,
				},
				SupplementalGroups: policy.SupplementalGroupsStrategyOptions{
					Rule: policy.SupplementalGroupsStrategyMustRunAs,
				},
				SELinux: policy.SELinuxStrategyOptions{
					Rule: policy.SELinuxStrategyMustRunAs,
				},
				FSGroup: policy.FSGroupStrategyOptions{
					Rule: policy.FSGroupStrategyMustRunAs,
				},
			},
		}
	}
	pspWithGenericVolume := func() *policy.PodSecurityPolicy {
		psp := pspWithoutGenericVolume()
		psp.Spec.Volumes = append(psp.Spec.Volumes, policy.Ephemeral)
		return psp
	}
	pspNil := func() *policy.PodSecurityPolicy {
		return nil
	}

	pspInfo := []struct {
		description      string
		hasGenericVolume bool
		psp              func() *policy.PodSecurityPolicy
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
			psp:              pspNil,
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPSPInfo := range pspInfo {
			for _, newPSPInfo := range pspInfo {
				oldPSP := oldPSPInfo.psp()
				newPSP := newPSPInfo.psp()
				if newPSP == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old PodSecurityPolicySpec %v, new PodSecurityPolicySpec %v", enabled, oldPSPInfo.description, newPSPInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.GenericEphemeralVolume, enabled)()

					var errs field.ErrorList
					var expectErrors bool
					if oldPSP == nil {
						errs = Strategy.Validate(context.Background(), newPSP)
						expectErrors = newPSPInfo.hasGenericVolume && !enabled
					} else {
						errs = Strategy.ValidateUpdate(context.Background(), newPSP, oldPSP)
						expectErrors = !oldPSPInfo.hasGenericVolume && newPSPInfo.hasGenericVolume && !enabled
					}
					if expectErrors && len(errs) == 0 {
						t.Error("expected errors, got none")
					}
					if !expectErrors && len(errs) > 0 {
						t.Errorf("expected no errors, got: %v", errs)
					}
				})
			}
		}
	}
}
