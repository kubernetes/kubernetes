/*
Copyright 2023 The Kubernetes Authors.

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

package core

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/resource"
)

func testResourceClaim(name string, namespace string, spec api.ResourceClaimSpec) *api.ResourceClaim {
	return &api.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
}

func TestResourceClaimEvaluatorUsage(t *testing.T) {
	classGpu := "gpu"
	validClaim := testResourceClaim("foo", "ns", api.ResourceClaimSpec{ResourceClassName: classGpu})

	evaluator := NewResourceClaimEvaluator(nil)
	testCases := map[string]struct {
		claim *api.ResourceClaim
		usage corev1.ResourceList
	}{
		"claim-usage": {
			claim: validClaim,
			usage: corev1.ResourceList{
				corev1.ResourceClaims:                              resource.MustParse("1"),
				"count/resourceclaims.resource.k8s.io":             resource.MustParse("1"),
				"gpu.resourceclass.resource.k8s.io/resourceclaims": resource.MustParse("1"),
			},
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			actual, err := evaluator.Usage(testCase.claim)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(testCase.usage, actual); diff != "" {
				t.Errorf("Unexpected usage (-want, +got):\n%s", diff)
			}
		})

	}
}

func TestResourceClaimEvaluatorMatchingResources(t *testing.T) {
	evaluator := NewResourceClaimEvaluator(nil)
	testCases := map[string]struct {
		items []corev1.ResourceName
		want  []corev1.ResourceName
	}{
		"supported-resources": {
			items: []corev1.ResourceName{
				"count/resourceclaims.resource.k8s.io",
				"resourceclaims",
				"gpu.resourceclass.resource.k8s.io/resourceclaims",
			},

			want: []corev1.ResourceName{
				"count/resourceclaims.resource.k8s.io",
				"resourceclaims",
				"gpu.resourceclass.resource.k8s.io/resourceclaims",
			},
		},
		"unsupported-resources": {
			items: []corev1.ResourceName{
				"storage",
				"ephemeral-storage",
				"bronze.resourceclass.resource.k8s.io/storage",
				"gpu.storage.k8s.io/requests.storage",
			},
			want: []corev1.ResourceName{},
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			actual := evaluator.MatchingResources(testCase.items)

			if diff := cmp.Diff(testCase.want, actual); diff != "" {
				t.Errorf("Unexpected response (-want, +got):\n%s", diff)
			}
		})
	}
}
