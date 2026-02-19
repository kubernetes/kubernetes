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
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
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

	podWithPodGroup := func(pod *v1.Pod, podGroupName string) *v1.Pod {
		pod = pod.DeepCopy()
		pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &podGroupName,
		}
		return pod
	}

	podNotOwner := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podNotOwner",
			UID:       newUID(),
		},
	}
	podGroupNotOwner := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podGroupNotOwner",
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
	podGroupOwner := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podGroupOwner",
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
			Name:      "claimWithOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podOwner.UID,
					Controller: &isController,
				},
			},
		},
	}
	claimWithPodGroupOwner := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithPodGroupOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podGroupOwner.UID,
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
	userClaimWithPodGroupOwner := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "user-namespace",
			Name:      "userClaimWithPodGroupOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					UID:        podGroupOwner.UID,
					Controller: &isController,
				},
			},
		},
	}

	testcases := map[string]struct {
		pod           *v1.Pod
		podGroup      *schedulingapi.PodGroup
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
			expectedError: `ResourceClaim kube-system/claimWithOwner was not created for Pod kube-system/podNotOwner (Pod is not owner)`,
		},
		"no-owner": {
			pod:           podOwner,
			claim:         claimNoOwner,
			expectedError: `ResourceClaim kube-system/claimNoOwner was not created for Pod kube-system/podOwner (Pod is not owner)`,
		},
		"different-namespace": {
			pod:           podOwner,
			claim:         userClaimWithOwner,
			expectedError: `ResourceClaim user-namespace/userClaimWithOwner was not created for Pod kube-system/podOwner (Pod is not owner)`,
		},
		"owned-by-podgroup": {
			pod:      podWithPodGroup(podOwner, podGroupOwner.Name),
			podGroup: podGroupOwner,
			claim:    claimWithPodGroupOwner,
		},
		"other-podgroup": {
			pod:           podWithPodGroup(podOwner, podGroupNotOwner.Name),
			podGroup:      podGroupNotOwner,
			claim:         claimWithPodGroupOwner,
			expectedError: `ResourceClaim kube-system/claimWithPodGroupOwner was not created for Pod kube-system/podOwner or PodGroup kube-system/podGroupNotOwner (neither Pod nor PodGroup is the owner)`,
		},
		"no-podgroup-owner": {
			pod:           podWithPodGroup(podOwner, podGroupOwner.Name),
			podGroup:      podGroupOwner,
			claim:         claimNoOwner,
			expectedError: `ResourceClaim kube-system/claimNoOwner was not created for Pod kube-system/podOwner or PodGroup kube-system/podGroupOwner (neither Pod nor PodGroup is the owner)`,
		},
		"different-podgroup-namespace": {
			pod:           podWithPodGroup(podOwner, podGroupOwner.Name),
			podGroup:      podGroupOwner,
			claim:         userClaimWithPodGroupOwner,
			expectedError: `ResourceClaim user-namespace/userClaimWithPodGroupOwner was not created for Pod kube-system/podOwner or PodGroup kube-system/podGroupOwner (neither Pod nor PodGroup is the owner)`,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			err := IsForPod(tc.pod, tc.podGroup, tc.claim)
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
	podGroupNotReserved := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podGroupNotReserved",
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
	podGroupReserved := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podGroupReserved",
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
	podGroupOtherReserved := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "podGroupOtherReserved",
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

	podWithPodGroup := func(pod *v1.Pod, podGroupName string) *v1.Pod {
		pod = pod.DeepCopy()
		pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
			PodGroupName: &podGroupName,
		}
		return pod
	}

	claimWithPodGroupReservation := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithPodGroupReservation",
			UID:       newUID(),
		},
		Status: resourceapi.ResourceClaimStatus{
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{
					UID: podGroupReserved.UID,
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

	claimWithMultiplePodGroupReservations := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithMultiplePodGroupReservations",
			UID:       newUID(),
		},
		Status: resourceapi.ResourceClaimStatus{
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{
					UID: podGroupReserved.UID,
				},
				{
					UID: podGroupOtherReserved.UID,
				},
			},
		},
	}

	claimWithReservationsForPodAndPodGroup := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithReservationsForPodAndPodGroup",
			UID:       newUID(),
		},
		Status: resourceapi.ResourceClaimStatus{
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{
					UID: podReserved.UID,
				},
				{
					UID: podGroupReserved.UID,
				},
			},
		},
	}

	testcases := map[string]struct {
		pod            *v1.Pod
		podGroup       *schedulingapi.PodGroup
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
		"reserved-for-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupReserved.Name),
			podGroup:       podGroupReserved,
			claim:          claimWithPodGroupReservation,
			expectedResult: true,
		},
		"reserved-for-pod-and-podgroup": {
			pod:            podWithPodGroup(podReserved, podGroupReserved.Name),
			podGroup:       podGroupReserved,
			claim:          claimWithReservationsForPodAndPodGroup,
			expectedResult: true,
		},
		"reserved-for-other-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupNotReserved.Name),
			podGroup:       podGroupNotReserved,
			claim:          claimWithReservation,
			expectedResult: false,
		},
		"multiple-reservations-including-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupReserved.Name),
			podGroup:       podGroupReserved,
			claim:          claimWithMultiplePodGroupReservations,
			expectedResult: true,
		},
		"multiple-reservations-excluding-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupNotReserved.Name),
			podGroup:       podGroupNotReserved,
			claim:          claimWithMultiplePodGroupReservations,
			expectedResult: false,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			result := IsReservedForPod(tc.pod, tc.podGroup, tc.claim)
			assert.Equal(t, tc.expectedResult, result)
		})
	}
}

func TestName(t *testing.T) {
	testcases := map[string]struct {
		pod           *v1.Pod
		podGroup      *schedulingapi.PodGroup
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
			expectedError: fmt.Errorf("Pod default/test-pod: %w", ErrClaimNotFound),
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
			expectedError: fmt.Errorf(`Pod default/test-pod, spec.resourceClaim invalid-claim: %w`, ErrAPIUnsupported),
		},
		"podgroup-claim-ungrouped-pod": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
			},
			podGroup: &schedulingapi.PodGroup{},
			podClaim: &v1.PodResourceClaim{
				Name:                  "pod-claim",
				PodGroupResourceClaim: new("podgroup-claim"),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf("Pod default/test-pod does not belong to a PodGroup"),
		},
		"podgroup-claim-without-podgroup": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
			},
			podGroup: nil,
			podClaim: &v1.PodResourceClaim{
				Name:                  "pod-claim",
				PodGroupResourceClaim: new("podgroup-claim"),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf(`Pod default/test-pod, spec.resourceClaim pod-claim: %w`, ErrAPIUnsupported),
		},
		"podgroup-claim-wrong-podgroup": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{
						PodGroupName: new("correct-podgroup"),
					},
				},
			},
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "wrong-podgroup",
				},
			},
			podClaim: &v1.PodResourceClaim{
				PodGroupResourceClaim: new("claim"),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf("Pod default/test-pod belongs to PodGroup default/correct-podgroup, not PodGroup default/wrong-podgroup"),
		},
		"podgroup-claim-not-found": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{
						PodGroupName: new("podgroup"),
					},
				},
			},
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "podgroup",
				},
				Spec: schedulingapi.PodGroupSpec{
					ResourceClaims: []schedulingapi.PodGroupResourceClaim{
						{Name: "not-this-one"},
						{Name: "not-this-one-either"},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				PodGroupResourceClaim: new("claim"),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf("PodGroup default/podgroup does not have claim claim requested by Pod test-pod"),
		},
		"podgroup-claim-name": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{
						PodGroupName: new("podgroup"),
					},
				},
			},
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "podgroup",
				},
				Spec: schedulingapi.PodGroupSpec{
					ResourceClaims: []schedulingapi.PodGroupResourceClaim{
						{
							Name:              "podgroup-claim",
							ResourceClaimName: new("resource-claim"),
						},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name:                  "pod-claim",
				PodGroupResourceClaim: new("podgroup-claim"),
			},
			expectedName:  new("resource-claim"),
			expectedCheck: false,
			expectedError: nil,
		},
		"podgroup-claim-template-name": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{
						PodGroupName: new("podgroup"),
					},
				},
				Status: v1.PodStatus{
					ResourceClaimStatuses: []v1.PodResourceClaimStatus{
						{
							Name: "not-this-one",
						},
						{
							Name:              "pod-claim",
							ResourceClaimName: ptr.To("generated-claim"),
						},
						{
							Name: "nor-this-one",
						},
					},
				},
			},
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "podgroup",
				},
				Spec: schedulingapi.PodGroupSpec{
					ResourceClaims: []schedulingapi.PodGroupResourceClaim{
						{
							Name:                      "podgroup-claim",
							ResourceClaimTemplateName: new("resource-claim-template"),
						},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name:                  "pod-claim",
				PodGroupResourceClaim: new("podgroup-claim"),
			},
			expectedName:  new("generated-claim"),
			expectedCheck: true,
			expectedError: nil,
		},
		"podgroup-claim-template-no-status": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{
						PodGroupName: new("podgroup"),
					},
				},
			},
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "podgroup",
				},
				Spec: schedulingapi.PodGroupSpec{
					ResourceClaims: []schedulingapi.PodGroupResourceClaim{
						{
							Name:                      "podgroup-claim",
							ResourceClaimTemplateName: new("resource-claim-template"),
						},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name:                  "pod-claim",
				PodGroupResourceClaim: new("podgroup-claim"),
			},
			expectedName:  new("resource-claim"),
			expectedCheck: false,
			expectedError: fmt.Errorf("PodGroup default/podgroup: %w", ErrClaimNotFound),
		},
		"podgroup-claim-template-unsupported-api": {
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-pod",
				},
				Spec: v1.PodSpec{
					SchedulingGroup: &v1.PodSchedulingGroup{
						PodGroupName: new("podgroup"),
					},
				},
			},
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "podgroup",
				},
				Spec: schedulingapi.PodGroupSpec{
					ResourceClaims: []schedulingapi.PodGroupResourceClaim{
						{
							Name: "podgroup-claim",
						},
					},
				},
			},
			podClaim: &v1.PodResourceClaim{
				Name:                  "pod-claim",
				PodGroupResourceClaim: new("podgroup-claim"),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf("PodGroup default/podgroup, spec.resourceClaim podgroup-claim: %w", ErrAPIUnsupported),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			name, check, err := Name(tc.pod, tc.podGroup, tc.podClaim)
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
		deviceConfigurations []resourceapi.DeviceAllocationConfiguration
		result               resourceapi.DeviceRequestAllocationResult
		expectedConfigs      []resourceapi.DeviceAllocationConfiguration
	}{
		"opaque-nil": {
			deviceConfigurations: []resourceapi.DeviceAllocationConfiguration{
				{
					Source:   resourceapi.AllocationConfigSourceClass,
					Requests: []string{},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: nil,
					},
				},
			},
			result: resourceapi.DeviceRequestAllocationResult{
				Request: "foo",
				Device:  "device-1",
			},
			expectedConfigs: nil,
		},
		"empty-requests-match-all": {
			deviceConfigurations: []resourceapi.DeviceAllocationConfiguration{
				{
					Source:   resourceapi.AllocationConfigSourceClass,
					Requests: []string{},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourceapi.DeviceRequestAllocationResult{
				Request: "foo",
				Device:  "device-1",
			},
			expectedConfigs: []resourceapi.DeviceAllocationConfiguration{
				{
					Source:   resourceapi.AllocationConfigSourceClass,
					Requests: []string{},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-regular-request": {
			deviceConfigurations: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourceapi.DeviceRequestAllocationResult{
				Request: "foo",
				Device:  "device-1",
			},
			expectedConfigs: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-parent-request-for-subrequest": {
			deviceConfigurations: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourceapi.DeviceRequestAllocationResult{
				Request: "foo/bar",
				Device:  "device-1",
			},
			expectedConfigs: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-subrequest": {
			deviceConfigurations: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo/not-bar",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourceapi.DeviceRequestAllocationResult{
				Request: "foo/bar",
				Device:  "device-1",
			},
			expectedConfigs: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
		},
		"match-both-source-class-and-claim": {
			deviceConfigurations: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
				{
					Source: resourceapi.AllocationConfigSourceClaim,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
			},
			result: resourceapi.DeviceRequestAllocationResult{
				Request: "foo/bar",
				Device:  "device-1",
			},
			expectedConfigs: []resourceapi.DeviceAllocationConfiguration{
				{
					Source: resourceapi.AllocationConfigSourceClass,
					Requests: []string{
						"foo",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver: "driver-a",
						},
					},
				},
				{
					Source: resourceapi.AllocationConfigSourceClaim,
					Requests: []string{
						"foo/bar",
					},
					DeviceConfiguration: resourceapi.DeviceConfiguration{
						Opaque: &resourceapi.OpaqueDeviceConfiguration{
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
