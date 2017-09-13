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

package gce

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/stretchr/testify/assert"
)

func TestGetServiceNetworkTier(t *testing.T) {
	createTestService := func() *v1.Service {
		return &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "randome-uid",
				Name:      "test-svc",
				Namespace: "test-ns",
			},
		}
	}

	for testName, testCase := range map[string]struct {
		annotations  map[string]string
		expectedTier NetworkTier
	}{
		"Use the default when the annotation does not exist": {
			annotations:  nil,
			expectedTier: NetworkTierDefault,
		},
		"Standard tier": {
			annotations:  map[string]string{NetworkTierAnnotationKey: "Standard"},
			expectedTier: NetworkTierStandard,
		},
		"Premium tier": {
			annotations:  map[string]string{NetworkTierAnnotationKey: "Premium"},
			expectedTier: NetworkTierPremium,
		},
	} {
		t.Run(testName, func(t *testing.T) {
			svc := createTestService()
			svc.Annotations = testCase.annotations
			actualTier := GetServiceNetworkTier(svc)
			assert.Equal(t, testCase.expectedTier, actualTier)
		})
	}
}

func TestAllKnownServiceLBAnnoationsAreValid(t *testing.T) {
	for k, v := range allServiceLBAnnotations {
		t.Logf("Verifying annotation %q", k)
		assert.NotEqual(t, 0, v.values.Len(), "annotation must have at least one valid value")
		assert.True(t, v.values.Has(v.defaultValue), "default value must be a valid value")
		assert.NotEqual(t, 0, len(v.defaultValue), "default value must be non-empty")
	}
}

func TestVerifySericeAnnotations(t *testing.T) {
	createTestService := func() *v1.Service {
		return &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "randome-uid",
				Name:      "test-svc",
				Namespace: "test-ns",
			},
			Spec: v1.ServiceSpec{
				Type: v1.ServiceTypeLoadBalancer,
			},
		}
	}

	for testName, testCase := range map[string]struct {
		annotations map[string]string
	}{
		"Unknown network tiers": {
			annotations: map[string]string{
				NetworkTierAnnotationKey: "foobar",
			},
		},
		"Network tiers is incompatible with ILB": {
			annotations: map[string]string{
				NetworkTierAnnotationKey:          "Premium",
				ServiceAnnotationLoadBalancerType: "Internal",
			},
		},
		"Shared backends is incompatible with ELB (default)": {
			annotations: map[string]string{
				ServiceAnnotationILBBackendShare: "true",
			},
		},
	} {
		t.Run(testName, func(t *testing.T) {
			svc := createTestService()
			svc.Annotations = testCase.annotations
			err := validateServiceLBAnnotations(svc)
			assert.True(t, err != nil)
		})
	}
}
