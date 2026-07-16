/*
Copyright 2020 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"slices"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	resourcelisters "k8s.io/client-go/listers/resource/v1"
	k8stesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	resourceclaimmetrics "k8s.io/kubernetes/pkg/controller/resourceclaim/metrics"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	testPodName          = "test-pod"
	testPodGroupName     = "test-podgroup"
	testNamespace        = "my-namespace"
	testPodUID           = types.UID("uidpod1")
	testPodGroupUID      = types.UID("uidpodgroup1")
	otherNamespace       = "not-my-namespace"
	podResourceClaimName = "acme-resource"
	templateName         = "my-template"
	className            = "my-resource-class"
	nodeName             = "worker"

	testPod                     = makePod(testPodName, testNamespace, testPodUID)
	testPodWithResource         = makePod(testPodName, testNamespace, testPodUID, *makePodResourceClaim(podResourceClaimName, templateName))
	testPodWithPodGroupResource = podInPodGroup(testPodWithResource, testPodName, testPodGroupName)

	otherTestPod = makePod(testPodName+"-II", testNamespace, testPodUID+"-II")

	testPodGroupWithResource         = makePodGroup(testPodGroupName, testNamespace, testPodGroupUID, *makePodGroupResourceClaim(podResourceClaimName, templateName))
	testPodGroupWithResourceInStatus = func() *schedulingapi.PodGroup {
		podGroup := testPodGroupWithResource.DeepCopy()
		podGroup.Status.ResourceClaimStatuses = []schedulingapi.PodGroupResourceClaimStatus{
			{Name: podResourceClaimName, ResourceClaimName: &testPodGroupClaim.Name},
		}
		return podGroup
	}()

	testClaim              = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, makeOwnerReference(testPodWithResource, true))
	testPodGroupClaim      = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, makeOwnerReference(testPodGroupWithResource, true))
	testClaimAllocated     = allocateClaim(testClaim)
	testClaimReserved      = reserveClaim(testClaimAllocated, testPodWithResource)
	testClaimReservedTwice = reserveClaim(testClaimReserved, otherTestPod)
	testClaimKey           = claimKeyPrefix + testClaim.Namespace + "/" + testClaim.Name
	testPodKey             = podKeyPrefix + testNamespace + "/" + testPodName
	testPodGroupKey        = podGroupKeyPrefix + testNamespace + "/" + testPodGroupName

	testClaimReservedForPodGroup = reserveClaim(testClaimAllocated, testPodGroupWithResource)

	templatedTestClaim                    = makeTemplatedClaim(podResourceClaimName, testPodName+"-"+podResourceClaimName+"-", testNamespace, className, 1, makeOwnerReference(testPodWithResource, true), nil)
	templatedTestClaimAllocated           = allocateClaim(templatedTestClaim)
	templatedTestClaimReserved            = reserveClaim(templatedTestClaimAllocated, testPodWithResource)
	templatedTestClaimReservedForPodGroup = reserveClaim(templatedTestClaimAllocated, testPodGroupWithResource)
	templatedTestPodGroupClaim            = makeTemplatedClaim(podResourceClaimName, testPodGroupName+"-"+podResourceClaimName+"-", testNamespace, className, 1, makeOwnerReference(testPodGroupWithResource, true), nil)

	templatedTestClaimWithAdmin          = makeTemplatedClaim(podResourceClaimName, testPodName+"-"+podResourceClaimName+"-", testNamespace, className, 1, makeOwnerReference(testPodWithResource, true), new(true))
	templatedTestClaimWithAdminAllocated = allocateClaim(templatedTestClaimWithAdmin)

	extendedTestClaim          = makeExtendedResourceClaim(testPodName, testNamespace, 1, makeOwnerReference(testPodWithResource, true))
	extendedTestClaimAllocated = allocateClaim(extendedTestClaim)

	conflictingClaim         = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, nil)
	conflictingPodGroupClaim = makeClaim(testPodGroupName+"-"+podResourceClaimName, testNamespace, className, nil)
	otherNamespaceClaim      = makeClaim(testPodName+"-"+podResourceClaimName, otherNamespace, className, nil)
	template                 = makeTemplate(templateName, testNamespace, className, nil)
	templateWithAdminAccess  = makeTemplate(templateName, testNamespace, className, new(true))

	testPodWithNodeName = func() *v1.Pod {
		pod := testPodWithResource.DeepCopy()
		pod.Spec.NodeName = nodeName
		pod.Status.ResourceClaimStatuses = append(pod.Status.ResourceClaimStatuses, v1.PodResourceClaimStatus{
			Name:              pod.Spec.ResourceClaims[0].Name,
			ResourceClaimName: &templatedTestClaim.Name,
		})
		return pod
	}()
	testPodWithPodGroupAndNodeName = podInPodGroup(testPodWithNodeName, testPodName, testPodGroupName)
	adminAccessFeatureOffError     = "admin access is requested, but the feature is disabled"
)

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name                          string
		key                           string
		adminAccessEnabled            bool
		prioritizedListEnabled        bool
		workloadResourceClaimsEnabled bool
		claims                        []*resourceapi.ResourceClaim
		claimsInCache                 []*resourceapi.ResourceClaim
		pods                          []*v1.Pod
		podsLater                     []*v1.Pod
		podGroups                     []*schedulingapi.PodGroup
		templates                     []*resourceapi.ResourceClaimTemplate
		expectedClaims                []resourceapi.ResourceClaim
		expectedStatuses              map[string][]v1.PodResourceClaimStatus
		expectedPodGroupStatuses      map[string][]schedulingapi.PodGroupResourceClaimStatus
		expectedError                 string
		expectedMetrics               expectedMetrics
	}{
		{
			name:           "create",
			pods:           []*v1.Pod{testPodWithResource},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			name:          "create with admin and feature gate off",
			pods:          []*v1.Pod{testPodWithResource},
			templates:     []*resourceapi.ResourceClaimTemplate{templateWithAdminAccess},
			key:           podKey(testPodWithResource),
			expectedError: adminAccessFeatureOffError,
		},
		{
			name:           "create with admin and feature gate on",
			pods:           []*v1.Pod{testPodWithResource},
			templates:      []*resourceapi.ResourceClaimTemplate{templateWithAdminAccess},
			key:            podKey(testPodWithResource),
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestClaimWithAdmin},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaimWithAdmin.Name},
				},
			},
			adminAccessEnabled: true,
			expectedMetrics:    expectedMetrics{0, 1, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "create for PodGroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			key:                           podGroupKey(testPodGroupWithResource),
			expectedClaims:                []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "skip create for Pod with PodGroup before template exists",
			pods:                          []*v1.Pod{testPodWithPodGroupResource},
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:                     []*resourceapi.ResourceClaimTemplate{},
			key:                           podKey(testPodWithPodGroupResource),
			expectedClaims:                nil,
			expectedMetrics:               expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "skip create for Pod with PodGroup after template exists",
			pods:                          []*v1.Pod{testPodWithPodGroupResource},
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			key:                           podKey(testPodWithPodGroupResource),
			expectedClaims:                nil,
			expectedMetrics:               expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "update Pod status with PodGroup claim",
			pods:                          []*v1.Pod{testPodWithPodGroupResource},
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			claims:                        []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			key:                           podKey(testPodWithPodGroupResource),
			expectedClaims:                []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupResource.Name: {
					{Name: testPodWithPodGroupResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: false,
			name:                          "create ResourceClaim for Pod with PodGroup claim when feature is disabled",
			pods:                          []*v1.Pod{testPodWithPodGroupResource},
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			claims:                        []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			key:                           podKey(testPodWithPodGroupResource),
			expectedClaims:                []resourceapi.ResourceClaim{*templatedTestPodGroupClaim, *templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupResource.Name: {
					{Name: testPodWithPodGroupResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			name: "nop",
			pods: []*v1.Pod{func() *v1.Pod {
				pod := testPodWithResource.DeepCopy()
				pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				}
				return pod
			}()},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			claims:         []*resourceapi.ResourceClaim{templatedTestClaim},
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "nop for PodGroup",
			podGroups: []*schedulingapi.PodGroup{func() *schedulingapi.PodGroup {
				podGroup := testPodGroupWithResource.DeepCopy()
				podGroup.Status.ResourceClaimStatuses = []schedulingapi.PodGroupResourceClaimStatus{
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				}
				return podGroup
			}()},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podGroupKey(testPodGroupWithResource),
			claims:         []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "recreate",
			pods: []*v1.Pod{func() *v1.Pod {
				pod := testPodWithResource.DeepCopy()
				pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				}
				return pod
			}()},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "recreate for PodGroup",
			podGroups: []*schedulingapi.PodGroup{func() *schedulingapi.PodGroup {
				pod := testPodGroupWithResource.DeepCopy()
				pod.Status.ResourceClaimStatuses = []schedulingapi.PodGroupResourceClaimStatus{
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				}
				return pod
			}()},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podGroupKey(testPodGroupWithResource),
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			name:          "missing-template",
			pods:          []*v1.Pod{testPodWithResource},
			templates:     nil,
			key:           podKey(testPodWithResource),
			expectedError: "resource claim template \"my-template\": resourceclaimtemplate.resource.k8s.io \"my-template\" not found",
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "missing-template-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:                     nil,
			key:                           podGroupKey(testPodGroupWithResource),
			expectedError:                 "resource claim template \"my-template\": resourceclaimtemplate.resource.k8s.io \"my-template\" not found",
		},
		{
			name:           "find-existing-claim-by-label",
			pods:           []*v1.Pod{testPodWithResource},
			key:            podKey(testPodWithResource),
			claims:         []*resourceapi.ResourceClaim{templatedTestClaim},
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "find-existing-claim-by-label-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                           podGroupKey(testPodGroupWithResource),
			claims:                        []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			expectedClaims:                []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:          "find-created-claim-in-cache",
			pods:          []*v1.Pod{testPodWithResource},
			key:           podKey(testPodWithResource),
			claimsInCache: []*resourceapi.ResourceClaim{templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "find-created-claim-in-cache-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                           podGroupKey(testPodGroupWithResource),
			claimsInCache:                 []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "no-such-pod",
			key:  podKey(testPodWithResource),
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "no-such-podgroup",
			key:                           podGroupKey(testPodGroupWithResource),
		},
		{
			name: "pod-deleted",
			pods: func() []*v1.Pod {
				deleted := metav1.Now()
				pods := []*v1.Pod{testPodWithResource.DeepCopy()}
				pods[0].DeletionTimestamp = &deleted
				return pods
			}(),
			key: podKey(testPodWithResource),
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "podgroup-deleted",
			podGroups: func() []*schedulingapi.PodGroup {
				deleted := metav1.Now()
				podGroups := []*schedulingapi.PodGroup{testPodGroupWithResource.DeepCopy()}
				podGroups[0].DeletionTimestamp = &deleted
				return podGroups
			}(),
			key: podGroupKey(testPodGroupWithResource),
		},
		{
			name: "no-volumes",
			pods: []*v1.Pod{testPod},
			key:  podKey(testPod),
		},
		{
			name:           "create-with-other-claim",
			pods:           []*v1.Pod{testPodWithResource},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			claims:         []*resourceapi.ResourceClaim{otherNamespaceClaim},
			expectedClaims: []resourceapi.ResourceClaim{*otherNamespaceClaim, *templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "create-with-other-claim-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			key:                           podGroupKey(testPodGroupWithResource),
			claims:                        []*resourceapi.ResourceClaim{otherNamespaceClaim},
			expectedClaims:                []resourceapi.ResourceClaim{*otherNamespaceClaim, *templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			name:           "wrong-claim-owner",
			pods:           []*v1.Pod{testPodWithResource},
			key:            podKey(testPodWithResource),
			claims:         []*resourceapi.ResourceClaim{conflictingClaim},
			expectedClaims: []resourceapi.ResourceClaim{*conflictingClaim},
			expectedError:  "resource claim template \"my-template\": resourceclaimtemplate.resource.k8s.io \"my-template\" not found",
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "wrong-claim-owner-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                           podGroupKey(testPodGroupWithResource),
			claims:                        []*resourceapi.ResourceClaim{conflictingPodGroupClaim},
			expectedClaims:                []resourceapi.ResourceClaim{*conflictingPodGroupClaim},
			expectedError:                 "resource claim template \"my-template\": resourceclaimtemplate.resource.k8s.io \"my-template\" not found",
		},
		{
			name:            "create-conflict",
			pods:            []*v1.Pod{testPodWithResource},
			templates:       []*resourceapi.ResourceClaimTemplate{template},
			key:             podKey(testPodWithResource),
			expectedMetrics: expectedMetrics{1, 0, 1, 0},
			expectedError:   "create ResourceClaim : Operation cannot be fulfilled on resourceclaims.resource.k8s.io \"fake name\": fake conflict",
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "create-conflict-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			key:                           podGroupKey(testPodGroupWithResource),
			expectedMetrics:               expectedMetrics{1, 0, 1, 0},
			expectedError:                 "create ResourceClaim : Operation cannot be fulfilled on resourceclaims.resource.k8s.io \"fake name\": fake conflict",
		},
		{
			name:            "stay-reserved-seen",
			pods:            []*v1.Pod{testPodWithResource},
			key:             claimKey(testClaimReserved),
			claims:          []*resourceapi.ResourceClaim{testClaimReserved},
			expectedClaims:  []resourceapi.ResourceClaim{*testClaimReserved},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:            "stay-reserved-not-seen",
			podsLater:       []*v1.Pod{testPodWithResource},
			key:             claimKey(testClaimReserved),
			claims:          []*resourceapi.ResourceClaim{testClaimReserved},
			expectedClaims:  []resourceapi.ResourceClaim{*testClaimReserved},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:   "clear-reserved-structured",
			pods:   []*v1.Pod{},
			key:    claimKey(testClaimReserved),
			claims: []*resourceapi.ResourceClaim{structuredParameters(testClaimReserved)},
			expectedClaims: func() []resourceapi.ResourceClaim {
				claim := testClaimAllocated.DeepCopy()
				claim.Finalizers = []string{}
				claim.Status.Allocation = nil
				return []resourceapi.ResourceClaim{*claim}
			}(),
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "clear-reserved-podgroup",
			podGroups:                     []*schedulingapi.PodGroup{},
			key:                           claimKey(testClaimReservedForPodGroup),
			claims:                        []*resourceapi.ResourceClaim{structuredParameters(testClaimReservedForPodGroup)},
			expectedClaims: func() []resourceapi.ResourceClaim {
				claim := testClaimAllocated.DeepCopy()
				claim.Finalizers = []string{}
				claim.Status.Allocation = nil
				return []resourceapi.ResourceClaim{*claim}
			}(),
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: false,
			name:                          "dont-clear-reserved-podgroup-feature-disabled",
			podGroups:                     []*schedulingapi.PodGroup{},
			key:                           claimKey(testClaimReservedForPodGroup),
			claims:                        []*resourceapi.ResourceClaim{structuredParameters(testClaimReservedForPodGroup)},
			expectedClaims:                []resourceapi.ResourceClaim{*structuredParameters(testClaimReservedForPodGroup)},
			expectedMetrics:               expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "dont-clear-reserved-structured",
			pods: []*v1.Pod{testPodWithResource},
			key:  claimKey(testClaimReserved),
			claims: func() []*resourceapi.ResourceClaim {
				claim := structuredParameters(testClaimReserved)
				claim = reserveClaim(claim, otherTestPod)
				return []*resourceapi.ResourceClaim{claim}
			}(),
			expectedClaims:  []resourceapi.ResourceClaim{*structuredParameters(testClaimReserved)},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "clear-reserved-structured-deleted",
			pods: []*v1.Pod{},
			key:  claimKey(testClaimReserved),
			claims: func() []*resourceapi.ResourceClaim {
				claim := structuredParameters(testClaimReserved.DeepCopy())
				claim.DeletionTimestamp = &metav1.Time{}
				return []*resourceapi.ResourceClaim{claim}
			}(),
			expectedClaims: func() []resourceapi.ResourceClaim {
				claim := structuredParameters(testClaimAllocated.DeepCopy())
				claim.DeletionTimestamp = &metav1.Time{}
				claim.Finalizers = []string{}
				claim.Status.Allocation = nil
				return []resourceapi.ResourceClaim{*claim}
			}(),
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "structured-deleted",
			pods: []*v1.Pod{},
			key:  claimKey(testClaimReserved),
			claims: func() []*resourceapi.ResourceClaim {
				claim := structuredParameters(testClaimAllocated.DeepCopy())
				claim.DeletionTimestamp = &metav1.Time{}
				return []*resourceapi.ResourceClaim{claim}
			}(),
			expectedClaims: func() []resourceapi.ResourceClaim {
				claim := structuredParameters(testClaimAllocated.DeepCopy())
				claim.DeletionTimestamp = &metav1.Time{}
				claim.Finalizers = []string{}
				claim.Status.Allocation = nil
				return []resourceapi.ResourceClaim{*claim}
			}(),
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "clear-reserved-when-done",
			pods: func() []*v1.Pod {
				pods := []*v1.Pod{testPodWithResource.DeepCopy()}
				pods[0].Status.Phase = v1.PodSucceeded
				return pods
			}(),
			key: claimKey(testClaimReserved),
			claims: func() []*resourceapi.ResourceClaim {
				claims := []*resourceapi.ResourceClaim{testClaimReserved.DeepCopy()}
				claims[0].OwnerReferences = nil
				return claims
			}(),
			expectedClaims: func() []resourceapi.ResourceClaim {
				claims := []resourceapi.ResourceClaim{*testClaimAllocated.DeepCopy()}
				claims[0].OwnerReferences = nil
				return claims
			}(),
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:            "remove-reserved",
			pods:            []*v1.Pod{testPod},
			key:             claimKey(testClaimReservedTwice),
			claims:          []*resourceapi.ResourceClaim{testClaimReservedTwice},
			expectedClaims:  []resourceapi.ResourceClaim{*testClaimReserved},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "delete-claim-when-done",
			pods: func() []*v1.Pod {
				pods := []*v1.Pod{testPodWithResource.DeepCopy()}
				pods[0].Status.Phase = v1.PodSucceeded
				return pods
			}(),
			key:             claimKey(testClaimReserved),
			claims:          []*resourceapi.ResourceClaim{testClaimReserved},
			expectedClaims:  nil,
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:           "add-reserved",
			pods:           []*v1.Pod{testPodWithNodeName},
			key:            podKey(testPodWithNodeName),
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			claims:         []*resourceapi.ResourceClaim{templatedTestClaimAllocated},
			expectedClaims: []resourceapi.ResourceClaim{*templatedTestClaimReserved},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithNodeName.Name: {
					{Name: testPodWithNodeName.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: true,
			name:                          "add-reserved-podgroup",
			pods:                          []*v1.Pod{testPodWithPodGroupAndNodeName},
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                           podKey(testPodWithPodGroupAndNodeName),
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			claims:                        []*resourceapi.ResourceClaim{templatedTestClaimAllocated},
			expectedClaims:                []resourceapi.ResourceClaim{*templatedTestClaimReservedForPodGroup},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupAndNodeName.Name: {
					{Name: testPodWithPodGroupAndNodeName.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			workloadResourceClaimsEnabled: false,
			name:                          "add-reserved-podgroup-feature-disabled",
			pods:                          []*v1.Pod{testPodWithPodGroupAndNodeName},
			podGroups:                     []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                           podKey(testPodWithPodGroupAndNodeName),
			templates:                     []*resourceapi.ResourceClaimTemplate{template},
			claims:                        []*resourceapi.ResourceClaim{templatedTestClaimAllocated},
			expectedClaims:                []resourceapi.ResourceClaim{*templatedTestClaimReserved},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupAndNodeName.Name: {
					{Name: testPodWithPodGroupAndNodeName.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "clean up pod reservation with non-pod reservation present",
			pods: func() []*v1.Pod {
				pod := testPodWithResource.DeepCopy()
				pod.Status.Phase = v1.PodSucceeded
				return []*v1.Pod{pod}
			}(),
			claims: func() []*resourceapi.ResourceClaim {
				claim := testClaimReserved.DeepCopy()
				nonPodRef := resourceapi.ResourceClaimConsumerReference{
					APIGroup: "foo.com",
					Resource: "foo",
					Name:     "foo",
					UID:      "123",
				}
				claim.Status.ReservedFor = append(claim.Status.ReservedFor, nonPodRef)
				return []*resourceapi.ResourceClaim{claim}
			}(),
			key: testClaimKey,
			expectedClaims: []resourceapi.ResourceClaim{func() resourceapi.ResourceClaim {
				claim := testClaimReserved.DeepCopy()
				nonPodRef := resourceapi.ResourceClaimConsumerReference{
					APIGroup: "foo.com",
					Resource: "foo",
					Name:     "foo",
					UID:      "123",
				}
				claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{nonPodRef}
				return *claim
			}()},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
	}

	for _, tc := range tests {
		// Run sequentially because of global logging and global metrics.
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)

			var objects []runtime.Object
			for _, pod := range tc.pods {
				objects = append(objects, pod)
			}
			for _, podGroup := range tc.podGroups {
				objects = append(objects, podGroup)
			}
			for _, claim := range tc.claims {
				objects = append(objects, claim)
			}
			for _, template := range tc.templates {
				objects = append(objects, template)
			}

			fakeKubeClient := createTestClient(objects...)
			if tc.expectedMetrics.numFailures > 0 {
				fakeKubeClient.PrependReactor("create", "resourceclaims", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, apierrors.NewConflict(action.GetResource().GroupResource(), "fake name", errors.New("fake conflict"))
				})
			}
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
			claimInformer := informerFactory.Resource().V1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
			setupMetrics()

			features := Features{
				AdminAccess:            tc.adminAccessEnabled,
				PrioritizedList:        tc.prioritizedListEnabled,
				WorkloadResourceClaims: tc.workloadResourceClaimsEnabled,
			}
			ec, err := NewController(tCtx.Logger(), features, fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer)
			if err != nil {
				t.Fatalf("error creating ephemeral controller : %v", err)
			}

			// Ensure informers are up-to-date.
			informerFactory.Start(tCtx.Done())
			stopInformers := func() {
				tCtx.Cancel("stopping informers")
				informerFactory.Shutdown()
			}
			defer stopInformers()
			informerFactory.WaitForCacheSync(tCtx.Done())

			// Add claims that only exist in the mutation cache.
			for _, claim := range tc.claimsInCache {
				ec.claimCache.Mutation(claim)
			}

			// Simulate race: stop informers, add more pods that the controller doesn't know about.
			stopInformers()
			for _, pod := range tc.podsLater {
				_, err := fakeKubeClient.CoreV1().Pods(pod.Namespace).Create(tCtx, pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unexpected error while creating pod: %v", err)
				}
			}

			err = ec.syncHandler(tCtx, tc.key)
			if err != nil {
				assert.ErrorContains(t, err, tc.expectedError, "the error message should have contained the expected error message")
				return
			}
			if tc.expectedError != "" {
				t.Fatalf("expected error, got none")
			}

			claims, err := fakeKubeClient.ResourceV1().ResourceClaims("").List(tCtx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("unexpected error while listing claims: %v", err)
			}
			assert.Equal(t, normalizeClaims(tc.expectedClaims), normalizeClaims(claims.Items))

			pods, err := fakeKubeClient.CoreV1().Pods("").List(tCtx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("unexpected error while listing pods: %v", err)
			}
			var actualStatuses map[string][]v1.PodResourceClaimStatus
			for _, pod := range pods.Items {
				if len(pod.Status.ResourceClaimStatuses) == 0 {
					continue
				}
				if actualStatuses == nil {
					actualStatuses = make(map[string][]v1.PodResourceClaimStatus)
				}
				actualStatuses[pod.Name] = pod.Status.ResourceClaimStatuses
			}
			assert.Equal(t, tc.expectedStatuses, actualStatuses, "pod resource claim statuses")

			podGroups, err := fakeKubeClient.SchedulingV1alpha2().PodGroups("").List(tCtx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("unexpected error while listing podgroups: %v", err)
			}
			var actualPodGroupStatuses map[string][]schedulingapi.PodGroupResourceClaimStatus
			for _, podGroup := range podGroups.Items {
				if len(podGroup.Status.ResourceClaimStatuses) == 0 {
					continue
				}
				if actualPodGroupStatuses == nil {
					actualPodGroupStatuses = make(map[string][]schedulingapi.PodGroupResourceClaimStatus)
				}
				actualPodGroupStatuses[podGroup.Name] = podGroup.Status.ResourceClaimStatuses
			}
			assert.Equal(t, tc.expectedPodGroupStatuses, actualPodGroupStatuses, "podgroup resource claim statuses")

			expectMetrics(t, tc.expectedMetrics)
		})
	}
}

func TestResourceClaimTemplateEventHandler(t *testing.T) {
	tCtx := ktesting.Init(t)

	fakeKubeClient := createTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	podInformer := informerFactory.Core().V1().Pods()
	podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
	claimInformer := informerFactory.Resource().V1().ResourceClaims()
	templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
	claimTemplateClient := fakeKubeClient.ResourceV1().ResourceClaimTemplates(testNamespace)
	claimTemplateTmpClient := fakeKubeClient.ResourceV1().ResourceClaimTemplates("tmp")
	podClient := fakeKubeClient.CoreV1().Pods(testNamespace)
	podTmpClient := fakeKubeClient.CoreV1().Pods("tmp")

	ec, err := NewController(tCtx.Logger(), Features{}, fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer)
	tCtx.ExpectNoError(err, "creating ephemeral controller")

	informerFactory.Start(tCtx.Done())
	stopInformers := func() {
		tCtx.Cancel("stopping informers")
		informerFactory.Shutdown()
	}
	defer stopInformers()

	expectQueue := func(tCtx ktesting.TContext, expectedKeys []string, expectedIndexerKeys []string) {
		g := gomega.NewWithT(tCtx)
		tCtx.Helper()

		lenDiffMessage := func() string {
			actualKeys := []string{}
			for ec.queue.Len() > 0 {
				actual, _ := ec.queue.Get()
				actualKeys = append(actualKeys, actual)
				ec.queue.Forget(actual)
				ec.queue.Done(actual)
			}
			return "Workqueue does not contain expected number of elements\n" +
				"Diff of elements (- expected, + actual):\n" +
				diff.Diff(expectedKeys, actualKeys)
		}

		g.Eventually(ec.queue.Len).
			WithTimeout(5*time.Second).
			Should(gomega.Equal(len(expectedKeys)), lenDiffMessage)
		g.Consistently(ec.queue.Len).
			WithTimeout(1*time.Second).
			Should(gomega.Equal(len(expectedKeys)), lenDiffMessage)

		g.Eventually(func() int { return len(ec.podIndexer.ListIndexFuncValues(podResourceClaimTemplateIndex)) }).
			WithTimeout(5 * time.Second).
			Should(gomega.Equal(len(expectedIndexerKeys)))
		g.Consistently(func() int { return len(ec.podIndexer.ListIndexFuncValues(podResourceClaimTemplateIndex)) }).
			WithTimeout(1 * time.Second).
			Should(gomega.Equal(len(expectedIndexerKeys)))

		for _, expected := range expectedKeys {
			actual, shuttingDown := ec.queue.Get()
			g.Expect(shuttingDown).To(gomega.BeFalseBecause("workqueue is unexpectedly shutting down"))
			g.Expect(actual).To(gomega.Equal(expected))
			ec.queue.Forget(actual)
			ec.queue.Done(actual)
		}

		for _, src := range expectedIndexerKeys {
			objects, err := ec.podIndexer.ByIndex(podResourceClaimTemplateIndex, src)
			g.Expect(err).NotTo(gomega.HaveOccurred(), "should not error when getting objects by index for key %s", src)
			g.Expect(objects).NotTo(gomega.BeEmpty(), "should have at least one object indexed for key %s", src)

			// Verify that the indexed objects are the expected pods
			found := false
			for _, obj := range objects {
				pod, ok := obj.(*v1.Pod)
				if !ok {
					continue
				}
				// Check if this pod matches the expected template reference
				for _, claim := range pod.Spec.ResourceClaims {
					if claim.ResourceClaimTemplateName != nil {
						// Build the expected index key for this pod
						expectedKey := pod.Namespace + "/" + *claim.ResourceClaimTemplateName
						if expectedKey == src {
							found = true
							break
						}
					}
				}
			}
			g.Expect(found).To(gomega.BeTrueBecause("should find a pod with template %s in index for key %s", templateName, src))
		}
	}

	tmpNamespace := "tmp"

	expectQueue(tCtx, []string{}, []string{})

	// Create two pods:
	// - testPodWithResource in the my-namespace namespace
	// - fake-1 in the tmp namespace
	_, err = podClient.Create(tCtx, testPodWithResource, metav1.CreateOptions{})
	_, err1 := podTmpClient.Create(tCtx, makePod("fake-1", tmpNamespace, "uidpod2", *makePodResourceClaim(podResourceClaimName, templateName)), metav1.CreateOptions{})
	tCtx.Step("create pod", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		tCtx.ExpectNoError(err1)
		expectQueue(tCtx, []string{testPodKey, podKeyPrefix + tmpNamespace + "/" + "fake-1"}, []string{testNamespace + "/" + templateName, tmpNamespace + "/" + templateName})
	})

	// The item  has been forgotten and marked as done in the workqueue,so queue is nil
	tCtx.Step("expect queue is nil", func(tCtx ktesting.TContext) {
		expectQueue(tCtx, []string{}, []string{testNamespace + "/" + templateName, tmpNamespace + "/" + templateName})
	})

	// After create claim template,queue should have test pod key
	_, err = claimTemplateClient.Create(tCtx, template, metav1.CreateOptions{})
	tCtx.Step("create claim template after pod backoff", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		expectQueue(tCtx, []string{testPodKey}, []string{testNamespace + "/" + templateName, tmpNamespace + "/" + templateName})
	})

	// The item  has been forgotten and marked as done in the workqueue,so queue is nil
	tCtx.Step("expect queue is nil", func(tCtx ktesting.TContext) {
		expectQueue(tCtx, []string{}, []string{testNamespace + "/" + templateName, tmpNamespace + "/" + templateName})
	})

	// After create tmp namespace claim template,queue should have fake pod key
	TmpNamespaceTemplate := makeTemplate(templateName, "tmp", className, nil)
	_, err = claimTemplateTmpClient.Create(tCtx, TmpNamespaceTemplate, metav1.CreateOptions{})
	tCtx.Step("create  claim template in tmp namespace after  pod backoff in test namespace", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		expectQueue(tCtx, []string{podKeyPrefix + tmpNamespace + "/" + "fake-1"}, []string{testNamespace + "/" + templateName, tmpNamespace + "/" + templateName})
	})

}

func TestResourceClaimEventHandler(t *testing.T) {
	tCtx := ktesting.Init(t)

	fakeKubeClient := createTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	podInformer := informerFactory.Core().V1().Pods()
	podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
	claimInformer := informerFactory.Resource().V1().ResourceClaims()
	templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
	setupMetrics()
	claimClient := fakeKubeClient.ResourceV1().ResourceClaims(testNamespace)

	ec, err := NewController(tCtx.Logger(), Features{}, fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer)
	tCtx.ExpectNoError(err, "creating ephemeral controller")

	informerFactory.Start(tCtx.Done())
	stopInformers := func() {
		tCtx.Cancel("stopping informers")
		informerFactory.Shutdown()
	}
	defer stopInformers()

	em := newNumMetrics(claimInformer.Lister())

	expectQueue := func(tCtx ktesting.TContext, expectedKeys []string) {
		g := gomega.NewWithT(tCtx)
		tCtx.Helper()

		lenDiffMessage := func() string {
			actualKeys := []string{}
			for ec.queue.Len() > 0 {
				actual, _ := ec.queue.Get()
				actualKeys = append(actualKeys, actual)
				ec.queue.Forget(actual)
				ec.queue.Done(actual)
			}
			return "Workqueue does not contain expected number of elements\n" +
				"Diff of elements (- expected, + actual):\n" +
				diff.Diff(expectedKeys, actualKeys)
		}

		g.Eventually(ec.queue.Len).
			WithTimeout(5*time.Second).
			Should(gomega.Equal(len(expectedKeys)), lenDiffMessage)
		g.Consistently(ec.queue.Len).
			WithTimeout(1*time.Second).
			Should(gomega.Equal(len(expectedKeys)), lenDiffMessage)

		for _, expected := range expectedKeys {
			actual, shuttingDown := ec.queue.Get()
			g.Expect(shuttingDown).To(gomega.BeFalseBecause("workqueue is unexpectedly shutting down"))
			g.Expect(actual).To(gomega.Equal(expected))
			ec.queue.Forget(actual)
			ec.queue.Done(actual)
		}
	}

	expectQueue(tCtx, []string{})

	_, err = claimClient.Create(tCtx, testClaim, metav1.CreateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: ""}, 1)
	tCtx.Step("create claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	modifiedClaim := testClaim.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	tCtx.Step("modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	_, err = claimClient.Update(tCtx, testClaimAllocated, metav1.UpdateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: ""}, -1)
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "false", Source: ""}, 1)
	tCtx.Step("allocate claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	modifiedClaim = testClaimAllocated.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar2"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	tCtx.Step("modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	otherClaimAllocated := testClaimAllocated.DeepCopy()
	otherClaimAllocated.Name += "2"
	_, err = claimClient.Create(tCtx, otherClaimAllocated, metav1.CreateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "false", Source: ""}, 1)
	tCtx.Step("create allocated claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey + "2"})
	})

	_, err = claimClient.Update(tCtx, testClaim, metav1.UpdateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: ""}, 1)
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "false", Source: ""}, -1)
	tCtx.Step("deallocate claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	err = claimClient.Delete(tCtx, testClaim.Name, metav1.DeleteOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: ""}, -1)
	tCtx.Step("delete deallocated claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{})
	})

	err = claimClient.Delete(tCtx, otherClaimAllocated.Name, metav1.DeleteOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "false", Source: ""}, -1)
	tCtx.Step("delete allocated claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{})
	})

	_, err = claimClient.Create(tCtx, templatedTestClaimWithAdmin, metav1.CreateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}, 1)
	tCtx.Step("create claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	modifiedClaim = templatedTestClaimWithAdmin.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	tCtx.Step("modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
	})

	_, err = claimClient.Update(tCtx, templatedTestClaimWithAdminAllocated, metav1.UpdateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}, -1)
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}, 1)
	tCtx.Step("allocate claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	modifiedClaim = templatedTestClaimWithAdminAllocated.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar2"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	tCtx.Step("modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
	})

	otherClaimAllocated = templatedTestClaimWithAdminAllocated.DeepCopy()
	otherClaimAllocated.Name += "2"
	_, err = claimClient.Create(tCtx, otherClaimAllocated, metav1.CreateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}, 1)
	tCtx.Step("create allocated claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	_, err = claimClient.Update(tCtx, templatedTestClaimWithAdmin, metav1.UpdateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}, 1)
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}, -1)
	tCtx.Step("deallocate claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	err = claimClient.Delete(tCtx, templatedTestClaimWithAdmin.Name, metav1.DeleteOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}, -1)
	tCtx.Step("delete deallocated claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	err = claimClient.Delete(tCtx, otherClaimAllocated.Name, metav1.DeleteOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}, -1)
	tCtx.Step("delete allocated claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	_, err = claimClient.Create(tCtx, extendedTestClaim, metav1.CreateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: "extended_resource"}, 1)
	tCtx.Step("create extended resource claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	_, err = claimClient.Update(tCtx, extendedTestClaimAllocated, metav1.UpdateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: "extended_resource"}, -1)
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "false", Source: "extended_resource"}, 1)
	tCtx.Step("allocate extended resource claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	_, err = claimClient.Update(tCtx, extendedTestClaim, metav1.UpdateOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: "extended_resource"}, 1)
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "true", AdminAccess: "false", Source: "extended_resource"}, -1)
	tCtx.Step("deallocate extended resource claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	err = claimClient.Delete(tCtx, extendedTestClaim.Name, metav1.DeleteOptions{})
	em = em.withUpdates(resourceclaimmetrics.NumResourceClaimLabels{Allocated: "false", AdminAccess: "false", Source: "extended_resource"}, -1)
	tCtx.Step("delete extended resource claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	em.Consistently(tCtx)
}

func TestEventHandlers(t *testing.T) { testEventHandlers(ktesting.Init(t)) }
func testEventHandlers(tCtx ktesting.TContext) {
	type object interface {
		runtime.Object
		metav1.Object
	}

	tests := map[string]struct {
		features        Features
		initialObjects  []runtime.Object
		createObjects   []object
		updateObjects   []object
		deleteObjects   []object
		expectedKeys    []string
		expectedMetrics map[resourceclaimmetrics.NumResourceClaimLabels]float64
	}{
		"nothing": {},
		"new-podgroup-feature-disabled": {
			features:      Features{WorkloadResourceClaims: false},
			createObjects: []object{testPodGroupWithResourceInStatus},
			expectedKeys:  []string{},
		},
		"new-podgroup": {
			features:      Features{WorkloadResourceClaims: true},
			createObjects: []object{testPodGroupWithResourceInStatus},
			expectedKeys:  []string{testPodGroupKey},
		},
		"new-podgroup-templated-claim-already-exists": {
			features:       Features{WorkloadResourceClaims: true},
			initialObjects: []runtime.Object{testPodGroupClaim},
			createObjects:  []object{testPodGroupWithResourceInStatus},
			expectedKeys:   []string{},
			expectedMetrics: map[resourceclaimmetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"new-templated-claim-for-podgroup": {
			features:       Features{WorkloadResourceClaims: true},
			initialObjects: []runtime.Object{testPodGroupWithResourceInStatus},
			createObjects:  []object{testPodGroupClaim},
			expectedKeys:   []string{testClaimKey},
			expectedMetrics: map[resourceclaimmetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"new-template-for-podgroup": {
			features:       Features{WorkloadResourceClaims: true},
			initialObjects: []runtime.Object{testPodGroupWithResource},
			createObjects:  []object{template},
			expectedKeys:   []string{testPodGroupKey},
		},
		"podgroup-claim-status-update": {
			features: Features{WorkloadResourceClaims: true},
			initialObjects: []runtime.Object{
				testPodGroupWithResource,
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-1", testPodGroupName),
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-2", testPodGroupName),
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-3", testPodGroupName+"-2"),
				podInPodGroup(testPod, testPodName+"-4", testPodGroupName),
			},
			updateObjects: []object{testPodGroupWithResourceInStatus},
			expectedKeys: []string{
				testPodGroupKey,
				testPodKey + "-1",
				testPodKey + "-2",
			},
		},
		"podgroup-claim-status-update-feature-disabled": {
			features: Features{WorkloadResourceClaims: false},
			initialObjects: []runtime.Object{
				testPodGroupWithResource,
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-1", testPodGroupName),
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-2", testPodGroupName),
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-3", testPodGroupName+"-2"),
				podInPodGroup(testPod, testPodName+"-4", testPodGroupName),
			},
			updateObjects: []object{testPodGroupWithResourceInStatus},
			expectedKeys:  []string{},
		},
	}
	for name, test := range tests {
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			fakeKubeClient := createTestClient(test.initialObjects...)
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
			claimInformer := informerFactory.Resource().V1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
			setupMetrics()

			ec, err := NewController(tCtx.Logger(), test.features, fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer)
			tCtx.ExpectNoError(err, "creating ephemeral controller")
			tCtx.Cleanup(ec.queue.ShutDown)

			informerFactory.Start(tCtx.Done())
			stopInformers := func() {
				tCtx.Cancel("stopping informers")
				informerFactory.Shutdown()
			}
			tCtx.Cleanup(stopInformers)

			drainQueue := func() []string {
				tCtx.Wait()
				actualKeys := []string{}
				for ec.queue.Len() > 0 {
					actual, shuttingDown := ec.queue.Get()
					tCtx.Expect(shuttingDown).To(gomega.BeFalseBecause("workqueue should not be shutting down"))
					actualKeys = append(actualKeys, actual)
					ec.queue.Forget(actual)
					ec.queue.Done(actual)
				}
				return actualKeys
			}
			gvr := func(obj metav1.Object) schema.GroupVersionResource {
				switch obj.(type) {
				case *v1.Pod:
					return v1.SchemeGroupVersion.WithResource("pods")
				case *schedulingapi.PodGroup:
					return schedulingapi.SchemeGroupVersion.WithResource("podgroups")
				case *resourceapi.ResourceClaim:
					return resourceapi.SchemeGroupVersion.WithResource("resourceclaims")
				case *resourceapi.ResourceClaimTemplate:
					return resourceapi.SchemeGroupVersion.WithResource("resourceclaimtemplates")
				}
				tCtx.Fatalf("invalid object type %T", obj)
				return schema.GroupVersionResource{}
			}

			// Not checking after initial objects added. Waiting for the
			// interesting operations.
			_ = drainQueue()

			for _, object := range test.createObjects {
				err := fakeKubeClient.Tracker().Create(gvr(object), object, object.GetNamespace(), metav1.CreateOptions{})
				tCtx.ExpectNoError(err)
			}
			for _, object := range test.updateObjects {
				err := fakeKubeClient.Tracker().Update(gvr(object), object, object.GetNamespace(), metav1.UpdateOptions{})
				tCtx.ExpectNoError(err)
			}
			for _, object := range test.deleteObjects {
				err := fakeKubeClient.Tracker().Delete(gvr(object), object.GetName(), object.GetNamespace(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err)
			}

			actualKeys := drainQueue()
			tCtx.Expect(actualKeys).To(gomega.ConsistOf(test.expectedKeys), "Workqueue does not contain expected elements")

			em := newNumMetrics(claimInformer.Lister())
			for labels, val := range test.expectedMetrics {
				em = em.withUpdates(labels, val)
			}
			em.verify(tCtx)
		})
	}
}

func TestGetAdminAccessMetricLabel(t *testing.T) {
	tests := []struct {
		name  string
		claim *resourceapi.ResourceClaim
		want  string
	}{
		{
			name:  "nil claim",
			claim: nil,
			want:  "false",
		},
		{
			name: "no requests",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: nil,
					},
				},
			},
			want: "false",
		},
		{
			name: "admin access false",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: ptr.To(false),
								},
							},
						},
					},
				},
			},
			want: "false",
		},
		{
			name: "admin access true",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: ptr.To(true),
								},
							},
						},
					},
				},
			},
			want: "true",
		},
		{
			name: "prioritized list",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								FirstAvailable: []resourceapi.DeviceSubRequest{{}},
							},
						},
					},
				},
			},
			want: "false",
		},
		{
			name: "multiple requests, one with admin access true",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: ptr.To(false),
								},
							},
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: ptr.To(true),
								},
							},
						},
					},
				},
			},
			want: "true",
		},
		{
			name: "multiple requests, all admin access false or nil",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: nil,
								},
							},
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: ptr.To(false),
								},
							},
						},
					},
				},
			},
			want: "false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getAdminAccessMetricLabel(tt.claim)
			if got != tt.want {
				t.Errorf("GetAdminAccessMetricLabel() = %v, want %v", got, tt.want)
			}
		})
	}
}

func makeClaim(name, namespace, classname string, owner *metav1.OwnerReference) *resourceapi.ResourceClaim {
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	if owner != nil {
		claim.OwnerReferences = []metav1.OwnerReference{*owner}
	}

	return claim
}

func makeTemplatedClaim(podClaimName, generateName, namespace, classname string, createCounter int, owner *metav1.OwnerReference, adminAccess *bool) *resourceapi.ResourceClaim {
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:         fmt.Sprintf("%s-%d", generateName, createCounter),
			GenerateName: generateName,
			Namespace:    namespace,
			Annotations:  map[string]string{resourceapi.PodResourceClaimAnnotation: podClaimName},
		},
	}
	if owner != nil {
		claim.OwnerReferences = []metav1.OwnerReference{*owner}
	}
	if adminAccess != nil {
		claim.Spec = resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{
					{
						Name: "req-0",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: "class",
							AdminAccess:     adminAccess,
						},
					},
				},
			},
		}
	}

	return claim
}

func makeExtendedResourceClaim(podName, namespace string, createCounter int, owner *metav1.OwnerReference) *resourceapi.ResourceClaim {
	generateName := podName + "-extended-resources-"
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:         fmt.Sprintf("%s-%d", generateName, createCounter),
			GenerateName: generateName,
			Namespace:    namespace,
			Annotations:  map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"},
		},
	}
	if owner != nil {
		claim.OwnerReferences = []metav1.OwnerReference{*owner}
	}

	return claim
}

func allocateClaim(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	claim.Status.Allocation = &resourceapi.AllocationResult{}
	return claim
}

func structuredParameters(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	// As far the controller is concerned, a claim was allocated by us if it has
	// this finalizer. For testing we don't need to update the allocation result.
	claim.Finalizers = append(claim.Finalizers, resourceapi.Finalizer)
	return claim
}

func reserveClaim(claim *resourceapi.ResourceClaim, obj metav1.Object) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	var apiGroup, resource string
	switch obj.(type) {
	case *v1.Pod:
		apiGroup = v1.GroupName
		resource = "pods"
	case *schedulingapi.PodGroup:
		apiGroup = schedulingapi.GroupName
		resource = "podgroups"
	default:
		panic(fmt.Sprintf("invalid type: %T", obj))
	}
	claim.Status.ReservedFor = append(claim.Status.ReservedFor,
		resourceapi.ResourceClaimConsumerReference{
			APIGroup: apiGroup,
			Resource: resource,
			Name:     obj.GetName(),
			UID:      obj.GetUID(),
		},
	)
	return claim
}

func makePodResourceClaim(name, templateName string) *v1.PodResourceClaim {
	return &v1.PodResourceClaim{
		Name:                      name,
		ResourceClaimTemplateName: &templateName,
	}
}

func makePod(name, namespace string, uid types.UID, podClaims ...v1.PodResourceClaim) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, UID: uid},
		Spec: v1.PodSpec{
			ResourceClaims: podClaims,
		},
	}

	return pod
}

func makePodGroupResourceClaim(name, templateName string) *schedulingapi.PodGroupResourceClaim {
	return &schedulingapi.PodGroupResourceClaim{
		Name:                      name,
		ResourceClaimTemplateName: &templateName,
	}
}

func makePodGroup(name, namespace string, uid types.UID, podGroupClaims ...schedulingapi.PodGroupResourceClaim) *schedulingapi.PodGroup {
	podGroup := &schedulingapi.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, UID: uid},
		Spec: schedulingapi.PodGroupSpec{
			ResourceClaims: podGroupClaims,
		},
	}

	return podGroup
}
func podInPodGroup(pod *v1.Pod, podName, podGroupName string) *v1.Pod {
	pod = pod.DeepCopy()
	pod.Name = podName
	pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{
		PodGroupName: &podGroupName,
	}
	return pod
}

func makeTemplate(name, namespace, classname string, adminAccess *bool) *resourceapi.ResourceClaimTemplate {
	template := &resourceapi.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	if adminAccess != nil {
		template.Spec = resourceapi.ResourceClaimTemplateSpec{
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{
						{
							Name: "req-0",
							Exactly: &resourceapi.ExactDeviceRequest{
								DeviceClassName: "class",
								AdminAccess:     adminAccess,
							},
						},
					},
				},
			},
		}
	}
	return template
}

func podKey(pod *v1.Pod) string {
	return podKeyPrefix + pod.Namespace + "/" + pod.Name
}

func podGroupKey(podGroup *schedulingapi.PodGroup) string {
	return podGroupKeyPrefix + podGroup.Namespace + "/" + podGroup.Name
}

func claimKey(claim *resourceapi.ResourceClaim) string {
	return claimKeyPrefix + claim.Namespace + "/" + claim.Name
}

func makeOwnerReference(obj metav1.Object, isController bool) *metav1.OwnerReference {
	var apiVersion, kind string
	switch obj.(type) {
	case *v1.Pod:
		apiVersion = v1.SchemeGroupVersion.String()
		kind = "Pod"
	case *schedulingapi.PodGroup:
		apiVersion = schedulingapi.SchemeGroupVersion.String()
		kind = "PodGroup"
	default:
		panic(fmt.Sprintf("invalid type %T", obj))
	}
	return &metav1.OwnerReference{
		APIVersion: apiVersion,
		Kind:       kind,
		Name:       obj.GetName(),
		UID:        obj.GetUID(),
		Controller: &isController,
	}
}

func normalizeClaims(claims []resourceapi.ResourceClaim) []resourceapi.ResourceClaim {
	sort.Slice(claims, func(i, j int) bool {
		if claims[i].Namespace < claims[j].Namespace {
			return true
		}
		if claims[i].Namespace > claims[j].Namespace {
			return false
		}
		return claims[i].Name < claims[j].Name
	})
	for i := range claims {
		if len(claims[i].Status.ReservedFor) == 0 {
			claims[i].Status.ReservedFor = nil
		}
	}
	return claims
}

func createTestClient(objects ...runtime.Object) *fake.Clientset {
	fakeClient := fake.NewSimpleClientset(objects...)
	fakeClient.PrependReactor("create", "resourceclaims", createResourceClaimReactor())
	return fakeClient
}

// createResourceClaimReactor implements the logic required for the GenerateName field to work when using
// the fake client. Add it with client.PrependReactor to your fake client.
func createResourceClaimReactor() func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
	nameCounter := 1
	var mutex sync.Mutex
	return func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
		mutex.Lock()
		defer mutex.Unlock()
		claim := action.(k8stesting.CreateAction).GetObject().(*resourceapi.ResourceClaim)
		if claim.Name == "" && claim.GenerateName != "" {
			claim.Name = fmt.Sprintf("%s-%d", claim.GenerateName, nameCounter)
		}
		nameCounter++
		return false, nil, nil
	}
}

type numMetrics struct {
	metrics map[resourceclaimmetrics.NumResourceClaimLabels]float64
	lister  resourcelisters.ResourceClaimLister
}

func getNumMetric(lister resourcelisters.ResourceClaimLister, logger klog.Logger) (em numMetrics, err error) {
	if lister == nil {
		return numMetrics{}, nil
	}

	// Create a fresh collector instance for each call to avoid registration conflicts
	freshCollector := newCustomCollector(lister, getAdminAccessMetricLabel, logger)
	testRegistry := metrics.NewKubeRegistry()
	testRegistry.CustomMustRegister(freshCollector)

	gatheredMetrics, err := testRegistry.Gather()
	if err != nil {
		return numMetrics{}, fmt.Errorf("failed to gather metrics: %w", err)
	}

	metricName := "resourceclaim_controller_resource_claims"

	em = newNumMetrics(lister)

	for _, mf := range gatheredMetrics {
		if mf.GetName() != metricName {
			continue
		}
		for _, metric := range mf.GetMetric() {
			labels := make(map[string]string)
			for _, labelPair := range metric.GetLabel() {
				labels[labelPair.GetName()] = labelPair.GetValue()
			}

			allocated := labels["allocated"]
			adminAccess := labels["admin_access"]
			source := labels["source"]
			value := metric.GetGauge().GetValue()

			em.metrics[resourceclaimmetrics.NumResourceClaimLabels{
				Allocated:   allocated,
				AdminAccess: adminAccess,
				Source:      source,
			}] = value
		}
	}

	return em, nil
}

func (em numMetrics) verify(tCtx ktesting.TContext) {
	tCtx.Helper()
	result, err := getNumMetric(em.lister, tCtx.Logger())
	tCtx.ExpectNoError(err)
	tCtx.Expect(result.metrics).To(gomega.Equal(em.metrics))
}

func (em numMetrics) Eventually(tCtx ktesting.TContext) {
	g := gomega.NewWithT(tCtx)
	tCtx.Helper()

	g.Eventually(func() (numMetrics, error) {
		result, err := getNumMetric(em.lister, tCtx.Logger())
		result.lister = em.lister
		return result, err
	}).WithTimeout(5 * time.Second).Should(gomega.Equal(em))
}

func (em numMetrics) Consistently(tCtx ktesting.TContext) {
	g := gomega.NewWithT(tCtx)
	tCtx.Helper()

	g.Consistently(func() (numMetrics, error) {
		result, err := getNumMetric(em.lister, tCtx.Logger())
		result.lister = em.lister
		return result, err
	}).WithTimeout(time.Second).Should(gomega.Equal(em))
}

type expectedMetrics struct {
	numCreated          int
	numCreatedWithAdmin int
	numFailures         int
	numFailureWithAdmin int
}

func expectMetrics(t *testing.T, em expectedMetrics) {
	t.Helper()

	// Check created claims
	actualCreated, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("success", "false"))
	handleErr(t, err, "ResourceClaimCreateSuccesses")
	if actualCreated != float64(em.numCreated) {
		t.Errorf("Expected claims to be created %d, got %v", em.numCreated, actualCreated)
	}

	// Check created claims with admin access
	actualCreatedWithAdmin, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("success", "true"))
	handleErr(t, err, "ResourceClaimCreateSuccessesWithAdminAccess")
	if actualCreatedWithAdmin != float64(em.numCreatedWithAdmin) {
		t.Errorf("Expected claims with admin access to be created %d, got %v", em.numCreatedWithAdmin, actualCreatedWithAdmin)
	}

	// Check failed claims
	actualFailed, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("failure", "false"))
	handleErr(t, err, "ResourceClaimCreateFailures")
	if actualFailed != float64(em.numFailures) {
		t.Errorf("Expected claims to have failed %d, got %v", em.numFailures, actualFailed)
	}

	// Check failed claims with admin access
	actualFailedWithAdmin, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("failure", "true"))
	handleErr(t, err, "ResourceClaimCreateFailuresWithAdminAccess")
	if actualFailedWithAdmin != float64(em.numFailureWithAdmin) {
		t.Errorf("Expected claims with admin access to have failed %d, got %v", em.numFailureWithAdmin, actualFailedWithAdmin)
	}
}
func handleErr(t *testing.T, err error, metricName string) {
	if err != nil {
		t.Errorf("Failed to get %s value, err: %v", metricName, err)
	}
}
func setupMetrics() {
	// Enable test mode to prevent global custom collector registration
	resourceclaimmetrics.SetTestMode(true)

	// Reset counter metrics for each test (they are registered by the controller itself)
	resourceclaimmetrics.ResourceClaimCreate.Reset()
}

func newNumMetrics(lister resourcelisters.ResourceClaimLister) numMetrics {
	metrics := make(map[resourceclaimmetrics.NumResourceClaimLabels]float64)
	for _, allocated := range []string{"false", "true"} {
		for _, adminAccess := range []string{"false", "true"} {
			for _, source := range []string{"", "extended_resource", "resource_claim_template"} {
				metrics[resourceclaimmetrics.NumResourceClaimLabels{
					Allocated:   allocated,
					AdminAccess: adminAccess,
					Source:      source,
				}] = 0
			}
		}
	}
	return numMetrics{
		metrics: metrics,
		lister:  lister,
	}
}

func (em numMetrics) withUpdates(rcLabels resourceclaimmetrics.NumResourceClaimLabels, n float64) numMetrics {
	em.metrics[rcLabels] += n
	return numMetrics{
		metrics: em.metrics,
		lister:  em.lister,
	}
}

func TestEnqueuePodExtendedResourceClaims(t *testing.T) {
	tests := []struct {
		name                        string
		pod                         *v1.Pod
		featureGateEnabled          bool
		deleted                     bool
		expectEarlyReturn           bool
		expectExtendedClaimEnqueued bool
	}{
		{
			name: "pod with no resource claims",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec:       v1.PodSpec{},
				Status:     v1.PodStatus{},
			},
			featureGateEnabled:          true,
			expectEarlyReturn:           true,
			expectExtendedClaimEnqueued: false,
		},
		{
			name: "pod with regular resource claims only",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{{Name: "regular-claim"}},
				},
				Status: v1.PodStatus{Phase: v1.PodRunning},
			},
			featureGateEnabled:          true,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: false,
		},
		{
			name: "pod with extended resource claim, feature enabled, running pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec:       v1.PodSpec{},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						ResourceClaimName: "test-extended-claim",
					},
				},
			},
			featureGateEnabled:          true,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: false,
		},
		{
			name: "pod with extended resource claim, feature enabled, completed pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec:       v1.PodSpec{},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						ResourceClaimName: "test-extended-claim",
					},
				},
			},
			featureGateEnabled:          true,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: true,
		},
		{
			name: "pod with extended resource claim, feature enabled, failed pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec:       v1.PodSpec{},
				Status: v1.PodStatus{
					Phase: v1.PodFailed, // Failed pod
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						ResourceClaimName: "test-extended-claim",
					},
				},
			},
			featureGateEnabled:          true,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: true,
		},
		{
			name: "pod with extended resource claim, feature disabled",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec:       v1.PodSpec{},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						ResourceClaimName: "test-extended-claim",
					},
				},
			},
			featureGateEnabled:          false,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: true,
		},
		{
			name: "deleted pod with extended resource claim",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec:       v1.PodSpec{},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						ResourceClaimName: "test-extended-claim",
					},
				},
			},
			featureGateEnabled:          true,
			deleted:                     true,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: true,
		},
		{
			name: "pod with both regular and extended resource claims, completed",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{{Name: "regular-claim"}},
				},
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
						ResourceClaimName: "test-extended-claim",
					},
				},
			},
			featureGateEnabled:          true,
			expectEarlyReturn:           false,
			expectExtendedClaimEnqueued: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, test.featureGateEnabled)

			tCtx := ktesting.Init(t)

			fakeKubeClient := createTestClient()
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			podGroupInformer := informerFactory.Scheduling().V1alpha2().PodGroups()
			claimInformer := informerFactory.Resource().V1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()

			setupMetrics()

			ec, err := NewController(tCtx.Logger(), Features{}, fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer)
			if err != nil {
				t.Fatalf("error creating controller: %v", err)
			}

			ec.enqueuePod(tCtx.Logger(), test.pod, test.deleted)

			var keys []string
			for ec.queue.Len() > 0 {
				k, _ := ec.queue.Get()
				keys = append(keys, k)
				ec.queue.Forget(k)
				ec.queue.Done(k)
			}

			if test.expectEarlyReturn {
				if len(keys) != 0 {
					t.Errorf("expected no keys enqueued on early return, got: %v", keys)
				}
				return
			}

			var expectedClaimKey string
			if test.pod.Status.ExtendedResourceClaimStatus != nil {
				expectedClaimKey = claimKeyPrefix + test.pod.Namespace + "/" + test.pod.Status.ExtendedResourceClaimStatus.ResourceClaimName
			}
			found := slices.Contains(keys, expectedClaimKey)
			if test.expectExtendedClaimEnqueued && !found {
				t.Errorf("expected extended claim key %q to be enqueued, got keys: %v", expectedClaimKey, keys)
			}
			if !test.expectExtendedClaimEnqueued && found {
				t.Errorf("did not expect extended claim key %q to be enqueued, got keys: %v", expectedClaimKey, keys)
			}
		})
	}
}
