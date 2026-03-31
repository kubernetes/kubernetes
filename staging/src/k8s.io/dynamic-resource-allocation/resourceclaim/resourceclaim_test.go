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

func TestResourceClaimIsFor(t *testing.T) {
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
					APIVersion: schedulingapi.SchemeGroupVersion.String(),
					Kind:       "PodGroup",
					Name:       podGroupOwner.Name,
					UID:        podGroupOwner.UID,
					Controller: &isController,
				},
			},
		},
	}
	claimWithOtherVersionPodGroupOwner := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "claimWithPodGroupOwner",
			UID:       newUID(),
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "scheduling.k8s.io/v25",
					Kind:       "PodGroup",
					Name:       podGroupOwner.Name,
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

	t.Run("Pod", func(t *testing.T) {
		testcases := map[string]struct {
			pod            *v1.Pod
			claim          *resourceapi.ResourceClaim
			acceptPodGroup bool
			expectedError  string
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
				expectedError: `ResourceClaim user-namespace/userClaimWithOwner is not in the same namespace as Pod kube-system/podOwner`,
			},
			"owned-by-podgroup": {
				pod:            podWithPodGroup(podOwner, podGroupOwner.Name),
				claim:          claimWithPodGroupOwner,
				acceptPodGroup: true,
			},
			"owned-by-podgroup-other-api-version": {
				pod:            podWithPodGroup(podOwner, podGroupOwner.Name),
				claim:          claimWithOtherVersionPodGroupOwner,
				acceptPodGroup: true,
			},
			"other-podgroup": {
				pod:            podWithPodGroup(podOwner, podGroupNotOwner.Name),
				claim:          claimWithPodGroupOwner,
				expectedError:  `ResourceClaim kube-system/claimWithPodGroupOwner was not created for Pod kube-system/podOwner (neither Pod nor PodGroup podGroupNotOwner is the owner)`,
				acceptPodGroup: true,
			},
			"no-podgroup-owner": {
				pod:            podWithPodGroup(podOwner, podGroupOwner.Name),
				claim:          claimNoOwner,
				expectedError:  `ResourceClaim kube-system/claimNoOwner was not created for Pod kube-system/podOwner (neither Pod nor PodGroup podGroupOwner is the owner)`,
				acceptPodGroup: true,
			},
			"different-podgroup-namespace": {
				pod:            podWithPodGroup(podOwner, podGroupOwner.Name),
				claim:          userClaimWithPodGroupOwner,
				expectedError:  `ResourceClaim user-namespace/userClaimWithPodGroupOwner is not in the same namespace as Pod kube-system/podOwner`,
				acceptPodGroup: true,
			},
		}

		for name, tc := range testcases {
			t.Run(name, func(t *testing.T) {
				err := IsForPod(tc.pod, tc.claim, tc.acceptPodGroup)
				if tc.expectedError == "" {
					require.NoError(t, err)
				} else {
					require.Error(t, err)
					require.EqualError(t, err, tc.expectedError)
				}
			})
		}
	})

	t.Run("PodGroup", func(t *testing.T) {
		testcases := map[string]struct {
			pod           *schedulingapi.PodGroup
			claim         *resourceapi.ResourceClaim
			expectedError string
		}{
			"owned": {
				pod:   podGroupOwner,
				claim: claimWithPodGroupOwner,
			},
			"other-pod": {
				pod:           podGroupNotOwner,
				claim:         claimWithPodGroupOwner,
				expectedError: `ResourceClaim kube-system/claimWithPodGroupOwner was not created for PodGroup kube-system/podGroupNotOwner (PodGroup is not owner)`,
			},
			"no-owner": {
				pod:           podGroupOwner,
				claim:         claimNoOwner,
				expectedError: `ResourceClaim kube-system/claimNoOwner was not created for PodGroup kube-system/podGroupOwner (PodGroup is not owner)`,
			},
			"different-namespace": {
				pod:           podGroupOwner,
				claim:         userClaimWithPodGroupOwner,
				expectedError: `ResourceClaim user-namespace/userClaimWithPodGroupOwner is not in the same namespace as PodGroup kube-system/podGroupOwner`,
			},
		}

		for name, tc := range testcases {
			t.Run(name, func(t *testing.T) {
				err := IsForPodGroup(tc.pod, tc.claim)
				if tc.expectedError == "" {
					require.NoError(t, err)
				} else {
					require.Error(t, err)
					require.EqualError(t, err, tc.expectedError)
				}
			})
		}
	})
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
					APIGroup: schedulingapi.GroupName,
					Resource: "podgroups",
					Name:     podGroupReserved.Name,
					UID:      podGroupReserved.UID,
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
					APIGroup: schedulingapi.GroupName,
					Resource: "podgroups",
					Name:     podGroupReserved.Name,
					UID:      podGroupReserved.UID,
				},
				{
					APIGroup: schedulingapi.GroupName,
					Resource: "podgroups",
					Name:     podGroupOtherReserved.Name,
					UID:      podGroupOtherReserved.UID,
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
		claim          *resourceapi.ResourceClaim
		acceptPodGroup bool
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
			claim:          claimWithPodGroupReservation,
			acceptPodGroup: true,
			expectedResult: true,
		},
		"podgroup-disabled": {
			pod:            podWithPodGroup(podNotReserved, podGroupReserved.Name),
			claim:          claimWithPodGroupReservation,
			acceptPodGroup: false,
			expectedResult: false,
		},
		"reserved-for-pod-and-podgroup": {
			pod:            podWithPodGroup(podReserved, podGroupReserved.Name),
			claim:          claimWithReservationsForPodAndPodGroup,
			acceptPodGroup: true,
			expectedResult: true,
		},
		"reserved-for-other-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupNotReserved.Name),
			claim:          claimWithReservation,
			acceptPodGroup: true,
			expectedResult: false,
		},
		"multiple-reservations-including-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupReserved.Name),
			claim:          claimWithMultiplePodGroupReservations,
			acceptPodGroup: true,
			expectedResult: true,
		},
		"multiple-reservations-excluding-podgroup": {
			pod:            podWithPodGroup(podNotReserved, podGroupNotReserved.Name),
			claim:          claimWithMultiplePodGroupReservations,
			acceptPodGroup: true,
			expectedResult: false,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			result := IsReservedForPod(tc.pod, tc.claim, tc.acceptPodGroup)
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

func TestNameFromPodGroup(t *testing.T) {
	testcases := map[string]struct {
		podGroup      *schedulingapi.PodGroup
		podGroupClaim *schedulingapi.PodGroupResourceClaim
		expectedName  *string
		expectedCheck bool
		expectedError error
	}{
		"resource-claim-name-set": {
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-podgroup",
				},
			},
			podGroupClaim: &schedulingapi.PodGroupResourceClaim{
				ResourceClaimName: new("existing-claim"),
			},
			expectedName:  new("existing-claim"),
			expectedCheck: false,
			expectedError: nil,
		},
		"resource-claim-template-name-set-and-status-found": {
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-podgroup",
				},
				Status: schedulingapi.PodGroupStatus{
					ResourceClaimStatuses: []schedulingapi.PodGroupResourceClaimStatus{
						{
							Name:              "template-claim",
							ResourceClaimName: new("created-claim"),
						},
					},
				},
			},
			podGroupClaim: &schedulingapi.PodGroupResourceClaim{
				Name:                      "template-claim",
				ResourceClaimTemplateName: new("template-claim-template"),
			},
			expectedName:  new("created-claim"),
			expectedCheck: true,
			expectedError: nil,
		},
		"resource-claim-template-name-set-but-status-not-found": {
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-podgroup",
				},
				Status: schedulingapi.PodGroupStatus{
					ResourceClaimStatuses: []schedulingapi.PodGroupResourceClaimStatus{
						{
							Name:              "other-claim",
							ResourceClaimName: new("other-created-claim"),
						},
					},
				},
			},
			podGroupClaim: &schedulingapi.PodGroupResourceClaim{
				Name:                      "template-claim",
				ResourceClaimTemplateName: new("template-claim-template"),
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf("PodGroup default/test-podgroup: %w", ErrClaimNotFound),
		},
		"neither-resource-claim-name-nor-template-name-set": {
			podGroup: &schedulingapi.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "test-podgroup",
				},
			},
			podGroupClaim: &schedulingapi.PodGroupResourceClaim{
				Name: "invalid-claim",
			},
			expectedName:  nil,
			expectedCheck: false,
			expectedError: fmt.Errorf("PodGroup default/test-podgroup, spec.resourceClaim invalid-claim: %w", ErrAPIUnsupported),
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			name, check, err := NameFromPodGroup(tc.podGroup, tc.podGroupClaim)
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

func TestBindTo(t *testing.T) {
	namespace := "my-namespace"
	podName := "my-pod"
	podUID := types.UID("pod-uid")

	podGroupName := "my-podgroup"
	podGroupUID := types.UID("podgroup-uid")

	claimName := "my-claim"
	resourceClaimName := new("my-resource-claim")

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      podName,
			UID:       podUID,
		},
		Spec: v1.PodSpec{
			SchedulingGroup: &v1.PodSchedulingGroup{
				PodGroupName: &podGroupName,
			},
		},
	}
	podReservation := resourceapi.ResourceClaimConsumerReference{
		APIGroup: v1.GroupName,
		Resource: "pods",
		Name:     podName,
		UID:      podUID,
	}

	podGroupWithClaims := func(claims ...schedulingapi.PodGroupResourceClaim) *schedulingapi.PodGroup {
		podGroup := &schedulingapi.PodGroup{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: namespace,
				Name:      podGroupName,
				UID:       podGroupUID,
			},
			Spec: schedulingapi.PodGroupSpec{
				ResourceClaims: claims,
			},
		}
		return podGroup
	}
	podGroupReservation := resourceapi.ResourceClaimConsumerReference{
		APIGroup: schedulingapi.GroupName,
		Resource: "podgroups",
		Name:     podGroupName,
		UID:      podGroupUID,
	}

	tests := map[string]struct {
		pod           *v1.Pod
		podGroup      *schedulingapi.PodGroup
		podClaim      *v1.PodResourceClaim
		expected      resourceapi.ResourceClaimConsumerReference
		expectedError error
	}{
		"unsupported API in Pod": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name: claimName,
				// Some future extension
			},
			expectedError: fmt.Errorf("Pod my-namespace/my-pod claim my-claim: %w", ErrAPIUnsupported),
		},
		"Pod claim with ResourceClaimName": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			expected: podReservation,
		},
		"Pod claim with ResourceClaimTemplateName": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:                      claimName,
				ResourceClaimTemplateName: resourceClaimName,
			},
			expected: podReservation,
		},
		"Pod claim with ResourceClaimName matching PodGroup": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(schedulingapi.PodGroupResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			}),
			expected: podGroupReservation,
		},
		"Pod claim with ResourceClaimTemplateName matching PodGroup": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:                      claimName,
				ResourceClaimTemplateName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(schedulingapi.PodGroupResourceClaim{
				Name:                      claimName,
				ResourceClaimTemplateName: resourceClaimName,
			}),
			expected: podGroupReservation,
		},
		"Wrong PodGroup namespace for Pod": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup:      &schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: "not-" + namespace, Name: podGroupName}},
			expectedError: fmt.Errorf("Pod my-namespace/my-pod belongs to PodGroup my-namespace/my-podgroup, not PodGroup not-my-namespace/my-podgroup"),
		},
		"Wrong PodGroup name for Pod": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup:      &schedulingapi.PodGroup{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: "not-" + podGroupName}},
			expectedError: fmt.Errorf("Pod my-namespace/my-pod belongs to PodGroup my-namespace/my-podgroup, not PodGroup my-namespace/not-my-podgroup"),
		},
		"No matching claim in PodGroup": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(
				schedulingapi.PodGroupResourceClaim{Name: "not-" + claimName, ResourceClaimName: resourceClaimName},
				schedulingapi.PodGroupResourceClaim{Name: "nor-" + claimName, ResourceClaimName: resourceClaimName},
			),
			expected: podReservation,
		},
		"Find matching claim in PodGroup": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(
				schedulingapi.PodGroupResourceClaim{Name: "not-" + claimName, ResourceClaimName: resourceClaimName},
				schedulingapi.PodGroupResourceClaim{Name: "nor-" + claimName, ResourceClaimName: resourceClaimName},
				schedulingapi.PodGroupResourceClaim{Name: claimName, ResourceClaimName: resourceClaimName},
			),
			expected: podGroupReservation,
		},
		"Claim name mismatch": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(
				schedulingapi.PodGroupResourceClaim{
					Name:              "not-" + claimName,
					ResourceClaimName: resourceClaimName,
				},
			),
			expected: podReservation,
		},
		"Claim resourceClaimName mismatch": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:              claimName,
				ResourceClaimName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(
				schedulingapi.PodGroupResourceClaim{
					Name:              claimName,
					ResourceClaimName: new("not-" + *resourceClaimName),
				},
			),
			expected: podReservation,
		},
		"Claim resourceClaimTemplateName mismatch": {
			pod: pod,
			podClaim: &v1.PodResourceClaim{
				Name:                      claimName,
				ResourceClaimTemplateName: resourceClaimName,
			},
			podGroup: podGroupWithClaims(
				schedulingapi.PodGroupResourceClaim{
					Name:                      claimName,
					ResourceClaimTemplateName: new("not-" + *resourceClaimName),
				},
			),
			expected: podReservation,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			actual, err := BindTo(test.pod, test.podGroup, test.podClaim)
			if test.expectedError == nil {
				require.NoError(t, err)
				assert.Equal(t, test.expected, actual)
			} else {
				assert.EqualError(t, err, test.expectedError.Error())
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
