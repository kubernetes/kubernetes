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
	"strings"
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
	validClaim := testResourceClaim("foo", "ns", api.ResourceClaimSpec{Devices: api.DeviceClaim{Requests: []api.DeviceRequest{{Name: "req-0", DeviceClassName: classGpu, AllocationMode: api.DeviceAllocationModeExactCount, Count: 1}}}})

	evaluator := NewResourceClaimEvaluator(nil)
	testCases := map[string]struct {
		claim  *api.ResourceClaim
		usage  corev1.ResourceList
		errMsg string
	}{
		"simple": {
			claim: validClaim,
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("1"),
			},
		},
		"many-requests": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				for i := 0; i < 4; i++ {
					claim.Spec.Devices.Requests = append(claim.Spec.Devices.Requests, claim.Spec.Devices.Requests[0])
				}
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("5"),
			},
		},
		"count": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Requests[0].Count = 5
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("5"),
			},
		},
		"all": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Requests[0].AllocationMode = api.DeviceAllocationModeAll
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": *resource.NewQuantity(api.AllocationResultsMaxSize, resource.DecimalSI),
			},
		},
		"unknown-count-mode": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				claim.Spec.Devices.Requests[0].AllocationMode = "future-mode"
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("0"),
			},
		},
		"admin": {
			claim: func() *api.ResourceClaim {
				claim := validClaim.DeepCopy()
				// Admins are *not* exempt from quota.
				claim.Spec.Devices.Requests[0].AdminAccess = true
				return claim
			}(),
			usage: corev1.ResourceList{
				"count/resourceclaims.resource.k8s.io":    resource.MustParse("1"),
				"gpu.deviceclass.resource.k8s.io/devices": resource.MustParse("1"),
			},
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			actual, err := evaluator.Usage(testCase.claim)
			if err != nil {
				if testCase.errMsg == "" {
					t.Fatalf("Unexpected error: %v", err)
				}
				if !strings.Contains(err.Error(), testCase.errMsg) {
					t.Fatalf("Expected error %q, got error: %v", testCase.errMsg, err.Error())
				}
			}
			if err == nil && testCase.errMsg != "" {
				t.Fatalf("Expected error %q, got none", testCase.errMsg)
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
				"gpu.deviceclass.resource.k8s.io/devices",
			},

			want: []corev1.ResourceName{
				"count/resourceclaims.resource.k8s.io",
				"gpu.deviceclass.resource.k8s.io/devices",
			},
		},
		"unsupported-resources": {
			items: []corev1.ResourceName{
				"resourceclaims", // no such alias
				"storage",
				"ephemeral-storage",
				"bronze.deviceclass.resource.k8s.io/storage",
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
