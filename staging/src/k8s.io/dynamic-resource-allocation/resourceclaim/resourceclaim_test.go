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

package resourceclaim

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestResourceClaimIsForPod(t *testing.T) {
	uid := 0
	newUID := func() types.UID {
		uid++
		return types.UID(fmt.Sprintf("%d", uid))
	}
	isController := true

	podNotOwner := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podNotOwner",
			UID:       newUID(),
		},
	}
	podOwner := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podOwner",
			UID:       newUID(),
		},
	}
	claimNoOwner := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimNoOwner",
			UID:       newUID(),
		},
	}
	claimWithOwner := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimNoOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podOwner.UID,
					Controller: &isController,
				},
			},
		},
	}
	userClaimWithOwner := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "user-namespace",
			Name:      "userClaimWithOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podOwner.UID,
					Controller: &isController,
				},
			},
		},
	}

	testcases := map[string]struct {
		pod           *v1.Pod
		claim         *resourceapi.ResourceClaim
		expectedError string
	}{
		"owned": {
			pod:   podOwner,
			claim: claimWithOwner,
		},
		"other-pod": {
			pod:           podNotOwner,
			claim:         claimWithOwner,
			expectedError: `ResourceClaim kube-system/claimNoOwner was not created for pod kube-system/podNotOwner (pod is not owner)`,
		},
		"no-owner": {
			pod:           podOwner,
			claim:         claimNoOwner,
			expectedError: `ResourceClaim kube-system/claimNoOwner was not created for pod kube-system/podOwner (pod is not owner)`,
		},
		"different-namespace": {
			pod:           podOwner,
			claim:         userClaimWithOwner,
			expectedError: `ResourceClaim user-namespace/userClaimWithOwner was not created for pod kube-system/podOwner (pod is not owner)`,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			err := IsForPod(tc.pod, tc.claim)
			if tc.expectedError == "" {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.EqualError(t, err, tc.expectedError)
			}
		})
	}
}

func TestIsReservedForPod(t *testing.T) {
	uid := 0
	newUID := func() types.UID {
		uid++
		return types.UID(fmt.Sprintf("%d", uid))
	}

	podNotReserved := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podNotReserved",
			UID:       newUID(),
		},
	}
	podReserved := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podReserved",
			UID:       newUID(),
		},
	}
	podOtherReserved := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podOtherReserved",
			UID:       newUID(),
		},
	}

	claimNoReservation := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimNoReservation",
			UID:       newUID(),
		},
		Status: resourceapi.ResourceClaimStatus{},
	}

	claimWithReservation := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithReservation",
			UID:       newUID(),
		},
		Status: resourceapi.ResourceClaimStatus{
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{
					UID: podReserved.UID,
				},
			},
		},
	}

	claimWithMultipleReservations := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithMultipleReservations",
			UID:       newUID(),
		},
		Status: resourceapi.ResourceClaimStatus{
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{
					UID: podReserved.UID,
				},
				{
					UID: podOtherReserved.UID,
				},
			},
		},
	}

	testcases := map[string]struct {
		pod            *v1.Pod
		claim          *resourceapi.ResourceClaim
		expectedResult bool
	}{
		"not-reserved": {
			pod:            podNotReserved,
			claim:          claimNoReservation,
			expectedResult: false,
		},
		"reserved-for-pod": {
			pod:            podReserved,
			claim:          claimWithReservation,
			expectedResult: true,
		},
		"reserved-for-other-pod": {
			pod:            podNotReserved,
			claim:          claimWithReservation,
			expectedResult: false,
		},
		"multiple-reservations-including-pod": {
			pod:            podReserved,
			claim:          claimWithMultipleReservations,
			expectedResult: true,
		},
		"multiple-reservations-excluding-pod": {
			pod:            podNotReserved,
			claim:          claimWithMultipleReservations,
			expectedResult: false,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			result := IsReservedForPod(tc.pod, tc.claim)
			assert.Equal(t, tc.expectedResult, result)
		})
	}
}

func TestName(t *testing.T) {
	testcases := map[string]struct {
		pod           *v1.Pod
		podClaim      *v1.PodResourceClaim
		expectedName  *string
		expectedCheck bool
		expectedError error
	}{
		"resource-claim-name-set": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
			},
			podClaim: &v1.PodResourceClaim{
				ResourceClaimName: func() *string { s := "existing-claim"; return &s }(),
			},
			expectedName:  func() *string { s := "existing-claim"; return &s }(),
			expectedCheck: false,
			expectedError: nil,
		},
		"resource-claim-template-name-set-and-status-found": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Status: v1.PodStatus{
					ResourceClaimStatuses: []v1.PodResourceClaimStatus{
						{
							Name:              "template-claim",
							ResourceClaimName: func() *string { s := "created-claim"; return &s }(),
						},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name:                      "template-claim",
				ResourceClaimTemplateName: func() *string { s := "template-claim-template"; return &s }(),
			},
			expectedName:  func() *string { s := "created-claim"; return &s }(),
			expectedCheck: true,
			expectedError: nil,
		},
		"resource-claim-template-name-set-but-status-not-found": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Status: v1.PodStatus{
					ResourceClaimStatuses: []v1.PodResourceClaimStatus{
						{
							Name:              "other-claim",
							ResourceClaimName: func() *string { s := "other-created-claim"; return &s }(),
						},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name:                      "template-claim",
				ResourceClaimTemplateName: func() *string { s := "template-claim-template"; return &s }(),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf(`pod "default/test-pod": %w`, ErrClaimNotFound),
		},
		"neither-resource-claim-name-nor-template-name-set": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name: "invalid-claim",
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf(`pod "default/test-pod", spec.resourceClaim "invalid-claim": %w`, ErrAPIUnsupported),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			name, check, err := Name(tc.pod, tc.podClaim)
			if tc.expectedError == nil {
				require.NoError(t, err)
				assert.Equal(t, tc.expectedName, name)
				assert.Equal(t, tc.expectedCheck, check)
			} else {
				require.EqualError(t, err, tc.expectedError.Error())
			}
		})
	}
}

func TestBaseRequestRef(t *testing.T) {
	testcases := map[string]struct {
		requestRef             string
		expectedBaseRequestRef string
	}{
		"valid-no-subrequest": {
			requestRef:             "foo",
			expectedBaseRequestRef: "foo",
		},
		"valid-subrequest": {
			requestRef:             "foo/bar",
			expectedBaseRequestRef: "foo",
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			baseRequestRef := BaseRequestRef(tc.requestRef)
			assert.Equal(t, tc.expectedBaseRequestRef, baseRequestRef)
		})
	}
}

func TestConfigForResult(t *testing.T) {
	testcases := map[string]struct {
		deviceConfigurations []resourcev1beta2.DeviceAllocationConfiguration
		result               resourcev1beta2.DeviceRequestAllocationResult
		expectedConfigs      []resourcev1beta2.DeviceAllocationConfiguration
	}{
		"opaque-nil": {
			deviceConfigurations: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source:   resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: nil,
					},
				},
			},
			result: resourcev1beta2.DeviceRequestAllocationResult{
				Request: "foo",
				Device:  "device-1",
			},
			expectedConfigs: nil,
		},
		"empty-requests-match-all": {
			deviceConfigurations: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source:   resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourcev1beta2.DeviceRequestAllocationResult{
				Request: "foo",
				Device:  "device-1",
			},
			expectedConfigs: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source:   resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-regular-request": {
			deviceConfigurations: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourcev1beta2.DeviceRequestAllocationResult{
				Request: "foo",
				Device:  "device-1",
			},
			expectedConfigs: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-parent-request-for-subrequest": {
			deviceConfigurations: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourcev1beta2.DeviceRequestAllocationResult{
				Request: "foo/bar",
				Device:  "device-1",
			},
			expectedConfigs: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-subrequest": {
			deviceConfigurations: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo/not-bar",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourcev1beta2.DeviceRequestAllocationResult{
				Request: "foo/bar",
				Device:  "device-1",
			},
			expectedConfigs: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-both-source-class-and-claim": {
			deviceConfigurations: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
				{
					Source: resourcev1beta2.AllocationConfigSourceClaim,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourcev1beta2.DeviceRequestAllocationResult{
				Request: "foo/bar",
				Device:  "device-1",
			},
			expectedConfigs: []resourcev1beta2.DeviceAllocationConfiguration{
				{
					Source: resourcev1beta2.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
				{
					Source: resourcev1beta2.AllocationConfigSourceClaim,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourcev1beta2.DeviceConfiguration{
						Opaque: &resourcev1beta2.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			configs := ConfigForResult(tc.deviceConfigurations, tc.result)
			assert.Equal(t, tc.expectedConfigs, configs)
		})
	}
}
