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
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	resourcelisters "k8s.io/client-go/listers/resource/v1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	resourceclaimmetrics "k8s.io/dynamic-resource-allocation/resourceclaim/metrics"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	controllermetrics "k8s.io/kubernetes/pkg/controller/resourceclaim/metrics"
	"k8s.io/kubernetes/test/utils/ktesting"
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
	nodeName             = "worker"

	testPod                     = makePod(testPodName, testNamespace, testPodUID)
	testPodWithResource         = makePod(testPodName, testNamespace, testPodUID, *makePodResourceClaim(podResourceClaimName, templateName))
	testPodWithPodGroupResource = podInPodGroup(testPodWithResource, testPodName, testPodGroupName)

	otherTestPod = makePod(testPodName+"-II", testNamespace, testPodUID+"-II")

	testPodGroup                     = makePodGroup(testPodGroupName, testNamespace, testPodGroupUID)
	testPodGroupWithResource         = makePodGroup(testPodGroupName, testNamespace, testPodGroupUID, *makePodGroupResourceClaim(podResourceClaimName, templateName))
	testPodGroupWithResourceInStatus = func() *schedulingapi.PodGroup {
		podGroup := testPodGroupWithResource.DeepCopy()
		podGroup.Status.ResourceClaimStatuses = []schedulingapi.PodGroupResourceClaimStatus{
			{Name: podResourceClaimName, ResourceClaimName: &testPodGroupClaim.Name},
		}
		return podGroup
	}()

	testClaim              = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, makeOwnerReference(testPodWithResource, true))
	testPodGroupClaim      = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, makeOwnerReference(testPodGroupWithResource, true))
	testClaimAllocated     = allocateClaim(testClaim)
	testClaimReserved      = reserveClaim(testClaimAllocated, testPodWithResource)
	testClaimReservedTwice = reserveClaim(testClaimReserved, otherTestPod)
	testClaimKey           = claimKeyPrefix + testClaim.Namespace + "/" + testClaim.Name
	testPodKey             = podKeyPrefix + testNamespace + "/" + testPodName
	testPodGroupKey        = podGroupKeyPrefix + testNamespace + "/" + testPodGroupName

	testClaimReservedForPodGroup = reserveClaim(testClaimAllocated, testPodGroupWithResource)

	templatedTestClaim                    = makeTemplatedClaim(podResourceClaimName, testPodName+"-"+podResourceClaimName+"-", testNamespace, 1, makeOwnerReference(testPodWithResource, true), nil)
	templatedTestClaimAllocated           = allocateClaim(templatedTestClaim)
	templatedTestClaimReserved            = reserveClaim(templatedTestClaimAllocated, testPodWithResource)
	templatedTestClaimReservedForPodGroup = reserveClaim(templatedTestClaimAllocated, testPodGroupWithResource)
	templatedTestPodGroupClaim            = makeTemplatedClaim(podResourceClaimName, testPodGroupName+"-"+podResourceClaimName+"-", testNamespace, 1, makeOwnerReference(testPodGroupWithResource, true), nil)

	templatedTestClaimWithAdmin          = makeTemplatedClaim(podResourceClaimName, testPodName+"-"+podResourceClaimName+"-", testNamespace, 1, makeOwnerReference(testPodWithResource, true), new(true))
	templatedTestClaimWithAdminAllocated = allocateClaim(templatedTestClaimWithAdmin)

	extendedTestClaim          = makeExtendedResourceClaim(testPodName, testNamespace, 1, makeOwnerReference(testPodWithResource, true))
	extendedTestClaimAllocated = allocateClaim(extendedTestClaim)

	conflictingClaim         = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, nil)
	conflictingPodGroupClaim = makeClaim(testPodGroupName+"-"+podResourceClaimName, testNamespace, nil)
	otherNamespaceClaim      = makeClaim(testPodName+"-"+podResourceClaimName, otherNamespace, nil)
	template                 = makeTemplate(templateName, testNamespace, nil)
	templateWithAdminAccess  = makeTemplate(templateName, testNamespace, new(true))

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

	// WorkloadResourceClaims depends on GenericWorkload
	allPossibleFeatures = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
	}
	adminAccessDisabled = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
	}
	adminAccessEnabled = []controllerFeatures{
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
	}
	genericWorkloadDisabled = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
	}
	genericWorkloadEnabled = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
	}
	workloadResourceClaimsDisabled = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: false, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
	}
	workloadResourceClaimsEnabled = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: true},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: true},
	}
	workloadResourceClaimsDisabledGenericWorkloadEnabled = []controllerFeatures{
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: false, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: false, WorkloadResourceClaims: false},
		{AdminAccess: true, GenericWorkload: true, PrioritizedList: true, WorkloadResourceClaims: false},
	}
)

func TestSyncHandler(t *testing.T) { testSyncHandler(ktesting.Init(t)) }
func testSyncHandler(tCtx ktesting.TContext) {
	tests := []struct {
		name                     string
		key                      string
		featureCombinations      []controllerFeatures
		claims                   []*resourceapi.ResourceClaim
		claimsInCache            []*resourceapi.ResourceClaim
		pods                     []*v1.Pod
		podsLater                []*v1.Pod
		podGroups                []*schedulingapi.PodGroup
		templates                []*resourceapi.ResourceClaimTemplate
		expectedClaims           []resourceapi.ResourceClaim
		expectedStatuses         map[string][]v1.PodResourceClaimStatus
		expectedPodGroupStatuses map[string][]schedulingapi.PodGroupResourceClaimStatus
		expectedError            string
		expectedMetrics          expectedMetrics
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
			name:                "create with admin and feature gate off",
			featureCombinations: adminAccessDisabled,
			pods:                []*v1.Pod{testPodWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{templateWithAdminAccess},
			key:                 podKey(testPodWithResource),
			expectedError:       adminAccessFeatureOffError,
		},
		{
			name:                "create with admin and feature gate on",
			featureCombinations: adminAccessEnabled,
			pods:                []*v1.Pod{testPodWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{templateWithAdminAccess},
			key:                 podKey(testPodWithResource),
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestClaimWithAdmin},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaimWithAdmin.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 1, 0, 0},
		},
		{
			name:                "create for Pod in PodGroup",
			featureCombinations: workloadResourceClaimsEnabled,
			pods:                []*v1.Pod{podInPodGroup(testPodWithResource, testPodName, testPodGroupName)},
			podGroups:           []*schedulingapi.PodGroup{testPodGroup},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithResource),
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			name:                "skip-create-for-pod-podgroup-does-not-exist",
			featureCombinations: genericWorkloadEnabled,
			pods:                []*v1.Pod{podInPodGroup(testPodWithResource, testPodName, testPodGroupName)},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithResource),
			expectedError:       `podgroup.scheduling.k8s.io "test-podgroup" not found`,
		},
		{
			name:                "create for PodGroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podGroupKey(testPodGroupWithResource),
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedPodGroupStatuses: map[string][]schedulingapi.PodGroupResourceClaimStatus{
				testPodGroupWithResource.Name: {
					{Name: testPodGroupWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
		{
			name:                "skip create for Pod with PodGroup before template exists",
			featureCombinations: workloadResourceClaimsEnabled,
			pods:                []*v1.Pod{testPodWithPodGroupResource},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{},
			key:                 podKey(testPodWithPodGroupResource),
			expectedClaims:      nil,
			expectedMetrics:     expectedMetrics{0, 0, 0, 0},
		},
		{
			name:                "skip create for Pod with PodGroup after template exists",
			featureCombinations: workloadResourceClaimsEnabled,
			pods:                []*v1.Pod{testPodWithPodGroupResource},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithPodGroupResource),
			expectedClaims:      nil,
			expectedMetrics:     expectedMetrics{0, 0, 0, 0},
		},
		{
			name:                "update Pod status with PodGroup claim",
			featureCombinations: workloadResourceClaimsEnabled,
			pods:                []*v1.Pod{testPodWithPodGroupResource},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			claims:              []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithPodGroupResource),
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupResource.Name: {
					{Name: testPodWithPodGroupResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestPodGroupClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:                "skip create ResourceClaim for Pod with PodGroup claim when GenericWorkload feature is disabled",
			featureCombinations: genericWorkloadDisabled,
			pods:                []*v1.Pod{testPodWithPodGroupResource},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithPodGroupResource),
			expectedError:       "GenericWorkload feature is disabled",
		},
		{
			name:                "skip create ResourceClaim for Pod with PodGroup claim when DRAWorkloadResourceClaims feature is disabled",
			featureCombinations: workloadResourceClaimsDisabledGenericWorkloadEnabled,
			pods:                []*v1.Pod{testPodWithPodGroupResource},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithPodGroupResource),
			expectedError:       "DRAWorkloadResourceClaims feature is disabled",
		},
		{
			name:                "create ResourceClaim for grouped Pod with claim when DRAWorkloadResourceClaims feature is disabled",
			featureCombinations: workloadResourceClaimsDisabledGenericWorkloadEnabled,
			pods:                []*v1.Pod{podInPodGroup(testPodWithResource, testPodName, testPodGroupName)},
			podGroups:           []*schedulingapi.PodGroup{testPodGroup},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podKey(testPodWithResource),
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodName: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
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
			name:                "nop for PodGroup",
			featureCombinations: workloadResourceClaimsEnabled,
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
			name:                "recreate for PodGroup",
			featureCombinations: workloadResourceClaimsEnabled,
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
			name:                "missing-template-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           nil,
			key:                 podGroupKey(testPodGroupWithResource),
			expectedError:       "resource claim template \"my-template\": resourceclaimtemplate.resource.k8s.io \"my-template\" not found",
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
			name:                "find-existing-claim-by-label-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                 podGroupKey(testPodGroupWithResource),
			claims:              []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestPodGroupClaim},
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
			name:                "find-created-claim-in-cache-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                 podGroupKey(testPodGroupWithResource),
			claimsInCache:       []*resourceapi.ResourceClaim{templatedTestPodGroupClaim},
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
			name:                "no-such-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			key:                 podGroupKey(testPodGroupWithResource),
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
			name:                "podgroup-deleted",
			featureCombinations: workloadResourceClaimsEnabled,
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
			name:                "create-with-other-claim-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podGroupKey(testPodGroupWithResource),
			claims:              []*resourceapi.ResourceClaim{otherNamespaceClaim},
			expectedClaims:      []resourceapi.ResourceClaim{*otherNamespaceClaim, *templatedTestPodGroupClaim},
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
			name:                "wrong-claim-owner-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                 podGroupKey(testPodGroupWithResource),
			claims:              []*resourceapi.ResourceClaim{conflictingPodGroupClaim},
			expectedClaims:      []resourceapi.ResourceClaim{*conflictingPodGroupClaim},
			expectedError:       "resource claim template \"my-template\": resourceclaimtemplate.resource.k8s.io \"my-template\" not found",
		},
		{
			name:            "create-conflict",
			pods:            []*v1.Pod{testPodWithResource},
			templates:       []*resourceapi.ResourceClaimTemplate{template},
			key:             podKey(testPodWithResource),
			expectedMetrics: expectedMetrics{0, 0, 1, 0},
			expectedError:   "create ResourceClaim : Operation cannot be fulfilled on resourceclaims.resource.k8s.io \"fake name\": fake conflict",
		},
		{
			name:                "create-conflict-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			key:                 podGroupKey(testPodGroupWithResource),
			expectedMetrics:     expectedMetrics{0, 0, 1, 0},
			expectedError:       "create ResourceClaim : Operation cannot be fulfilled on resourceclaims.resource.k8s.io \"fake name\": fake conflict",
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
			name:                "clear-reserved-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			podGroups:           []*schedulingapi.PodGroup{},
			key:                 claimKey(testClaimReservedForPodGroup),
			claims:              []*resourceapi.ResourceClaim{structuredParameters(testClaimReservedForPodGroup)},
			expectedClaims: func() []resourceapi.ResourceClaim {
				claim := testClaimAllocated.DeepCopy()
				claim.Finalizers = []string{}
				claim.Status.Allocation = nil
				return []resourceapi.ResourceClaim{*claim}
			}(),
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:                "dont-clear-reserved-podgroup-feature-disabled",
			featureCombinations: workloadResourceClaimsDisabled,
			podGroups:           []*schedulingapi.PodGroup{},
			key:                 claimKey(testClaimReservedForPodGroup),
			claims:              []*resourceapi.ResourceClaim{structuredParameters(testClaimReservedForPodGroup)},
			expectedClaims:      []resourceapi.ResourceClaim{*structuredParameters(testClaimReservedForPodGroup)},
			expectedMetrics:     expectedMetrics{0, 0, 0, 0},
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
			name:                "add-reserved-podgroup",
			featureCombinations: workloadResourceClaimsEnabled,
			pods:                []*v1.Pod{testPodWithPodGroupAndNodeName},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                 podKey(testPodWithPodGroupAndNodeName),
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			claims:              []*resourceapi.ResourceClaim{templatedTestClaimAllocated},
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestClaimReservedForPodGroup},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupAndNodeName.Name: {
					{Name: testPodWithPodGroupAndNodeName.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:                "skip-add-reserved-podgroup-genericworkload-disabled",
			featureCombinations: genericWorkloadDisabled,
			pods:                []*v1.Pod{testPodWithPodGroupAndNodeName},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                 podKey(testPodWithPodGroupAndNodeName),
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			claims:              []*resourceapi.ResourceClaim{templatedTestClaimAllocated},
			expectedError:       "GenericWorkload feature is disabled",
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestClaimAllocated},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithPodGroupAndNodeName.Name: {
					{Name: testPodWithPodGroupAndNodeName.Spec.ResourceClaims[0].Name, ResourceClaimName: &templatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:                "skip-add-reserved-podgroup-workloadresourceclaims-disabled",
			featureCombinations: workloadResourceClaimsDisabledGenericWorkloadEnabled,
			pods:                []*v1.Pod{testPodWithPodGroupAndNodeName},
			podGroups:           []*schedulingapi.PodGroup{testPodGroupWithResource},
			key:                 podKey(testPodWithPodGroupAndNodeName),
			templates:           []*resourceapi.ResourceClaimTemplate{template},
			claims:              []*resourceapi.ResourceClaim{templatedTestClaimAllocated},
			expectedClaims:      []resourceapi.ResourceClaim{*templatedTestClaimAllocated},
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
		{
			name: "flapping-resourceclaim-statuses",
			pods: func() []*v1.Pod {
				pod := makePod(testPodName, testNamespace, testPodUID,
					*makePodResourceClaim("claimA", templateName),
					*makePodResourceClaim("claimB", templateName),
				)
				// Initially only claimA is in status
				pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
					{Name: "claimA", ResourceClaimName: new("claimA-object")},
				}
				return []*v1.Pod{pod}
			}(),
			templates: []*resourceapi.ResourceClaimTemplate{template},
			claims: []*resourceapi.ResourceClaim{
				makeClaim("claimA-object", testNamespace, makeOwnerReference(testPod, true)),
			},
			key: podKeyPrefix + testNamespace + "/" + testPodName,
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodName: {
					{Name: "claimA", ResourceClaimName: new("claimA-object")},
					{Name: "claimB", ResourceClaimName: new("test-pod-claimB--1")},
				},
			},
			expectedClaims: []resourceapi.ResourceClaim{
				*makeClaim("claimA-object", testNamespace, makeOwnerReference(testPod, true)),
				*makeTemplatedClaim("claimB", testPodName+"-claimB-", testNamespace, 1, makeOwnerReference(testPod, true), nil),
			},
			expectedMetrics: expectedMetrics{1, 0, 0, 0},
		},
	}

	for _, tc := range tests {
		// Run sequentially because of global logging and global metrics.
		run := func(tCtx ktesting.TContext, features controllerFeatures) {
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
			var appliedPatches []string
			fakeKubeClient.PrependReactor("patch", "pods", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
				patchAction := action.(k8stesting.PatchAction)
				if patchAction.GetSubresource() == "status" {
					appliedPatches = append(appliedPatches, string(patchAction.GetPatch()))
				}
				return false, nil, nil
			})
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			podGroupInformer := informerFactory.Scheduling().V1alpha3().PodGroups()
			claimInformer := informerFactory.Resource().V1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
			setupMetrics()

			ec, err := newControllerWithFeatures(tCtx.Logger(), fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer, features)
			if err != nil {
				tCtx.Fatalf("error creating ephemeral controller : %v", err)
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
					tCtx.Fatalf("unexpected error while creating pod: %v", err)
				}
			}

			err = ec.syncHandler(tCtx, tc.key)
			if tc.expectedError != "" {
				assert.ErrorContains(tCtx, err, tc.expectedError, "the error message should have contained the expected error message")
			} else if err != nil {
				tCtx.Errorf("unexpected sync handler error: %v", err)
			}
			// keep going to check other side effects

			if tc.name == "flapping-resourceclaim-statuses" {
				assert.Len(tCtx, appliedPatches, 1, "should have applied status once")
				assert.Contains(tCtx, appliedPatches[0], `"name":"claimA"`, "patch should contain claimA")
				assert.Contains(tCtx, appliedPatches[0], `"name":"claimB"`, "patch should contain claimB")
			}

			claims, err := fakeKubeClient.ResourceV1().ResourceClaims("").List(tCtx, metav1.ListOptions{})
			if err != nil {
				tCtx.Fatalf("unexpected error while listing claims: %v", err)
			}
			assert.Equal(tCtx, normalizeClaims(tc.expectedClaims), normalizeClaims(claims.Items))

			pods, err := fakeKubeClient.CoreV1().Pods("").List(tCtx, metav1.ListOptions{})
			if err != nil {
				tCtx.Fatalf("unexpected error while listing pods: %v", err)
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
			assert.Equal(tCtx, tc.expectedStatuses, actualStatuses, "pod resource claim statuses")

			podGroups, err := fakeKubeClient.SchedulingV1alpha3().PodGroups("").List(tCtx, metav1.ListOptions{})
			if err != nil {
				tCtx.Fatalf("unexpected error while listing podgroups: %v", err)
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
			assert.Equal(tCtx, tc.expectedPodGroupStatuses, actualPodGroupStatuses, "podgroup resource claim statuses")

			expectMetrics(tCtx, tc.expectedMetrics)
		}
		tCtx.Run(tc.name, func(tCtx ktesting.TContext) {
			if len(tc.featureCombinations) == 0 {
				tc.featureCombinations = allPossibleFeatures
			}
			for _, features := range tc.featureCombinations {
				tCtx.Run(features.String(), func(tCtx ktesting.TContext) {
					run(tCtx, features)
				})
			}
		})
	}
}

func TestEventHandlers(t *testing.T) { testEventHandlers(ktesting.Init(t)) }
func testEventHandlers(tCtx ktesting.TContext) {
	type object interface {
		runtime.Object
		metav1.Object
	}

	modifiedClaim := testClaim.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar"}

	modifiedClaimAdminAccess := templatedTestClaimWithAdmin.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar"}

	otherClaimAllocated := testClaimAllocated.DeepCopy()
	otherClaimAllocated.Name += "2"
	otherClaimKey := testClaimKey + "2"

	otherClaimAllocatedAdminAccess := templatedTestClaimWithAdminAllocated.DeepCopy()
	otherClaimAllocatedAdminAccess.Name += "2"

	templatedTestClaimKey := testClaimKey + "--1"
	templatedOtherClaimKey := templatedTestClaimKey + "2"
	extendedResourceTemplatedClaimKey := claimKeyPrefix + testNamespace + "/" + testPodName + "-extended-resources--1"

	otherNSTemplate := makeTemplate(templateName, otherNamespace, nil)

	otherNSPod := makePod("fake-1", otherNamespace, "uidpod2", *makePodResourceClaim(podResourceClaimName, templateName))
	otherNSPodKey := podKeyPrefix + otherNamespace + "/" + otherNSPod.Name

	extendedResourceClaimName := "test-extended-claim"
	podWithExtendedResourceClaim := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: testPodName, Namespace: testNamespace},
		Spec:       v1.PodSpec{},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
				ResourceClaimName: extendedResourceClaimName,
			},
		},
	}

	completedPodWithExtendedResourceClaim := podWithExtendedResourceClaim.DeepCopy()
	completedPodWithExtendedResourceClaim.Status.Phase = v1.PodSucceeded

	completedPodWithRegularAndExtendedResourceClaim := podWithExtendedResourceClaim.DeepCopy()
	completedPodWithRegularAndExtendedResourceClaim.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "regular-claim"}}
	completedPodWithRegularAndExtendedResourceClaim.Status.Phase = v1.PodSucceeded

	failedPodWithExtendedResourceClaim := podWithExtendedResourceClaim.DeepCopy()
	failedPodWithExtendedResourceClaim.Status.Phase = v1.PodFailed

	extendedResourceClaimKey := claimKeyPrefix + testNamespace + "/" + extendedResourceClaimName

	tests := map[string]struct {
		featureCombinations []controllerFeatures
		initialObjects      []runtime.Object
		createObjects       []object
		updateObjects       []object
		deleteObjects       []object

		expectedKeys    []string
		expectedMetrics map[controllermetrics.NumResourceClaimLabels]float64

		expectedIndexedPodsByResourceClaimTemplate []string
	}{
		"nothing": {},
		"new-claim": {
			createObjects: []object{testClaim},
			expectedKeys:  []string{testClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"update-claim": {
			initialObjects: []runtime.Object{testClaim},
			updateObjects:  []object{modifiedClaim},
			expectedKeys:   []string{testClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"allocate-claim": {
			initialObjects: []runtime.Object{testClaim},
			updateObjects:  []object{testClaimAllocated},
			expectedKeys:   []string{testClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "false"}: 1,
			},
		},
		"allocate-another-claim": {
			initialObjects: []runtime.Object{testClaimAllocated},
			createObjects:  []object{otherClaimAllocated},
			expectedKeys:   []string{otherClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "false"}: 2,
			},
		},
		"deallocate-claim": {
			initialObjects: []runtime.Object{testClaimAllocated, otherClaimAllocated},
			updateObjects:  []object{testClaim},
			expectedKeys:   []string{testClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "false"}:  1,
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"delete-deallocated-claim": {
			initialObjects: []runtime.Object{testClaim, otherClaimAllocated},
			deleteObjects:  []object{testClaim},
			expectedKeys:   []string{},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "false"}: 1,
			},
		},
		"delete-allocated-claim": {
			initialObjects:  []runtime.Object{otherClaimAllocated},
			deleteObjects:   []object{otherClaimAllocated},
			expectedKeys:    []string{},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{},
		},
		"new-claim-admin-access": {
			createObjects: []object{templatedTestClaimWithAdmin},
			expectedKeys:  []string{templatedTestClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}: 1,
			},
		},
		"update-claim-admin-access": {
			initialObjects: []runtime.Object{templatedTestClaimWithAdmin},
			updateObjects:  []object{modifiedClaimAdminAccess},
			expectedKeys:   []string{templatedTestClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}: 1,
			},
		},
		"allocate-claim-admin-access": {
			initialObjects: []runtime.Object{modifiedClaimAdminAccess},
			updateObjects:  []object{templatedTestClaimWithAdminAllocated},
			expectedKeys:   []string{templatedTestClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}: 1,
			},
		},
		"allocate-another-claim-admin-access": {
			initialObjects: []runtime.Object{templatedTestClaimWithAdminAllocated},
			createObjects:  []object{otherClaimAllocatedAdminAccess},
			expectedKeys:   []string{templatedOtherClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}: 2,
			},
		},
		"deallocate-claim-admin-access": {
			initialObjects: []runtime.Object{templatedTestClaimWithAdminAllocated, otherClaimAllocatedAdminAccess},
			updateObjects:  []object{templatedTestClaimWithAdmin},
			expectedKeys:   []string{templatedTestClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}:  1,
				{Allocated: "false", AdminAccess: "true", Source: "resource_claim_template"}: 1,
			},
		},
		"delete-deallocated-claim-admin-access": {
			initialObjects: []runtime.Object{templatedTestClaimWithAdmin, otherClaimAllocatedAdminAccess},
			deleteObjects:  []object{templatedTestClaimWithAdmin},
			expectedKeys:   []string{},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "true", Source: "resource_claim_template"}: 1,
			},
		},
		"delete-allocated-claim-admin-access": {
			initialObjects:  []runtime.Object{otherClaimAllocatedAdminAccess},
			deleteObjects:   []object{otherClaimAllocatedAdminAccess},
			expectedKeys:    []string{},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{},
		},
		"new-claim-extended-resources": {
			createObjects: []object{extendedTestClaim},
			expectedKeys:  []string{extendedResourceTemplatedClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false", Source: "extended_resource"}: 1,
			},
		},
		"allocate-claim-extended-resources": {
			initialObjects: []runtime.Object{extendedTestClaim},
			updateObjects:  []object{extendedTestClaimAllocated},
			expectedKeys:   []string{extendedResourceTemplatedClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "true", AdminAccess: "false", Source: "extended_resource"}: 1,
			},
		},
		"deallocate-claim-extended-resources": {
			initialObjects: []runtime.Object{extendedTestClaimAllocated},
			updateObjects:  []object{extendedTestClaim},
			expectedKeys:   []string{extendedResourceTemplatedClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false", Source: "extended_resource"}: 1,
			},
		},
		"delete-claim-extended-resources": {
			initialObjects:  []runtime.Object{extendedTestClaimAllocated},
			deleteObjects:   []object{extendedTestClaimAllocated},
			expectedKeys:    []string{},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{},
		},
		"new-pods": {
			createObjects: []object{testPodWithResource, otherNSPod},
			expectedKeys:  []string{testPodKey, otherNSPodKey},
			expectedIndexedPodsByResourceClaimTemplate: []string{
				testNamespace + "/" + templateName,
				otherNamespace + "/" + templateName,
			},
		},
		"new-template-for-pod": {
			initialObjects: []runtime.Object{testPodWithResource, otherNSPod},
			createObjects:  []object{template},
			expectedKeys:   []string{testPodKey},
			expectedIndexedPodsByResourceClaimTemplate: []string{
				testNamespace + "/" + templateName,
				otherNamespace + "/" + templateName,
			},
		},
		"new-template-for-pod-other-namespace": {
			initialObjects: []runtime.Object{testPodWithResource, otherNSPod, template},
			createObjects:  []object{otherNSTemplate},
			expectedKeys:   []string{otherNSPodKey},
			expectedIndexedPodsByResourceClaimTemplate: []string{
				testNamespace + "/" + templateName,
				otherNamespace + "/" + templateName,
			},
		},
		"pod-without-resource-claims": {
			createObjects: []object{testPod},
			expectedKeys:  []string{},
		},
		"running-pod-with-extended-resource-claim": {
			createObjects: []object{podWithExtendedResourceClaim},
			expectedKeys:  []string{},
		},
		"completed-pod-with-extended-resource-claim": {
			createObjects: []object{completedPodWithExtendedResourceClaim},
			expectedKeys:  []string{extendedResourceClaimKey},
		},
		"faled-pod-with-extended-resource-claim": {
			createObjects: []object{failedPodWithExtendedResourceClaim},
			expectedKeys:  []string{extendedResourceClaimKey},
		},
		"delete-pod-with-extended-resource-claim": {
			initialObjects: []runtime.Object{podWithExtendedResourceClaim},
			deleteObjects:  []object{podWithExtendedResourceClaim},
			expectedKeys:   []string{extendedResourceClaimKey},
		},
		"completed-pod-with-regular-and-extended-resource-claim": {
			createObjects: []object{completedPodWithRegularAndExtendedResourceClaim},
			expectedKeys:  []string{extendedResourceClaimKey, testPodKey},
		},
		"new-podgroup-feature-disabled": {
			featureCombinations: workloadResourceClaimsDisabled,
			createObjects:       []object{testPodGroupWithResourceInStatus},
			expectedKeys:        []string{},
		},
		"new-podgroup": {
			featureCombinations: workloadResourceClaimsEnabled,
			createObjects:       []object{testPodGroupWithResourceInStatus},
			expectedKeys:        []string{testPodGroupKey},
		},
		"new-podgroup-templated-claim-already-exists": {
			featureCombinations: workloadResourceClaimsEnabled,
			initialObjects:      []runtime.Object{testPodGroupClaim},
			createObjects:       []object{testPodGroupWithResourceInStatus},
			expectedKeys:        []string{},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"new-templated-claim-for-podgroup": {
			featureCombinations: workloadResourceClaimsEnabled,
			initialObjects:      []runtime.Object{testPodGroupWithResourceInStatus},
			createObjects:       []object{testPodGroupClaim},
			expectedKeys:        []string{testClaimKey},
			expectedMetrics: map[controllermetrics.NumResourceClaimLabels]float64{
				{Allocated: "false", AdminAccess: "false"}: 1,
			},
		},
		"new-template-for-podgroup": {
			featureCombinations: workloadResourceClaimsEnabled,
			initialObjects:      []runtime.Object{testPodGroupWithResource},
			createObjects:       []object{template},
			expectedKeys:        []string{testPodGroupKey},
		},
		"podgroup-claim-status-update": {
			featureCombinations: workloadResourceClaimsEnabled,
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
			expectedIndexedPodsByResourceClaimTemplate: []string{testNamespace + "/" + templateName},
		},
		"podgroup-claim-status-update-feature-disabled": {
			featureCombinations: workloadResourceClaimsDisabled,
			initialObjects: []runtime.Object{
				testPodGroupWithResource,
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-1", testPodGroupName),
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-2", testPodGroupName),
				podInPodGroup(testPodWithPodGroupResource, testPodName+"-3", testPodGroupName+"-2"),
				podInPodGroup(testPod, testPodName+"-4", testPodGroupName),
			},
			updateObjects: []object{testPodGroupWithResourceInStatus},
			expectedKeys:  []string{},
			expectedIndexedPodsByResourceClaimTemplate: []string{testNamespace + "/" + templateName},
		},
	}
	for name, test := range tests {
		run := func(tCtx ktesting.TContext, features controllerFeatures) {
			fakeKubeClient := createTestClient(test.initialObjects...)
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			podGroupInformer := informerFactory.Scheduling().V1alpha3().PodGroups()
			claimInformer := informerFactory.Resource().V1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1().ResourceClaimTemplates()
			setupMetrics()

			ec, err := newControllerWithFeatures(tCtx.Logger(), fakeKubeClient, podInformer, podGroupInformer, claimInformer, templateInformer, features)
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
				err := fakeKubeClient.Tracker().Delete(gvr(object), object.GetNamespace(), object.GetName(), metav1.DeleteOptions{})
				tCtx.ExpectNoError(err)
			}

			actualKeys := drainQueue()
			tCtx.Expect(actualKeys).To(gomega.ConsistOf(test.expectedKeys), "Workqueue does not contain expected elements")

			em := newNumMetrics(claimInformer.Lister())
			for labels, val := range test.expectedMetrics {
				em = em.withUpdates(labels, val)
			}
			em.verify(tCtx)

			actualIndexedPodsByResourceClaimTemplate := ec.podIndexer.ListIndexFuncValues(podResourceClaimTemplateIndex)
			tCtx.Expect(actualIndexedPodsByResourceClaimTemplate).To(gomega.ConsistOf(test.expectedIndexedPodsByResourceClaimTemplate), "expected Pods were not indexed by ResourceClaimTemplate")
		}
		if len(test.featureCombinations) == 0 {
			test.featureCombinations = allPossibleFeatures
		}
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			for _, features := range test.featureCombinations {
				tCtx.SyncTest(features.String(), func(tCtx ktesting.TContext) {
					run(tCtx, features)
				})
			}
		})
	}
}

func TestWorkqueue(t *testing.T) { testWorkqueue(ktesting.Init(t)) }
func testWorkqueue(tCtx ktesting.TContext) {
	tests := map[string]struct {
		shutdown       bool
		syncHandlerErr error
		expectStopLoop bool
		expectSynced   bool
		expectRequeue  bool
	}{
		"no-error": {
			syncHandlerErr: nil,
			expectSynced:   true,
		},
		"retryable-error": {
			syncHandlerErr: fmt.Errorf("will retry"),
			expectSynced:   true,
			expectRequeue:  true,
		},
		"nonretryable-error": {
			syncHandlerErr: nonRetryableError{fmt.Errorf("will not retry")},
			expectSynced:   true,
			expectRequeue:  false,
		},
		"wrapped-nonretryable-error": {
			syncHandlerErr: fmt.Errorf("%w", nonRetryableError{fmt.Errorf("will not retry")}),
			expectSynced:   true,
			expectRequeue:  false,
		},
		"shutdown": {
			shutdown:       true,
			expectSynced:   false,
			expectStopLoop: true,
		},
	}
	for name, test := range tests {
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			// The default [utilruntime.ErrorHandlers] use [time] but are
			// initialized before the synctest bubble is established. As a
			// result, when the error handler is invoked the default rate
			// limiting begins a sleep for the duration between *fake* now and
			// *actual* now. The workqueue in turn spins through its internal
			// periodic accounting for more than 26 years(!) which eventually
			// resolves, but wastes actual CPU time.
			oldErrorHandlers := utilruntime.ErrorHandlers
			tCtx.Cleanup(func() {
				utilruntime.ErrorHandlers = oldErrorHandlers
			})
			utilruntime.ErrorHandlers = []utilruntime.ErrorHandler{
				func(ctx context.Context, err error, msg string, keysAndValues ...any) {
					klog.FromContext(ctx).Error(err, msg, keysAndValues...)
				},
			}

			rateLimitDelay := 1 * time.Second
			ec := &Controller{
				queue: workqueue.NewTypedRateLimitingQueueWithConfig(
					&staticRateLimiter[string]{delay: rateLimitDelay, requeues: map[string]int{}},
					workqueue.TypedRateLimitingQueueConfig[string]{Name: "resource_claim"},
				),
			}
			tCtx.Cleanup(ec.queue.ShutDown)

			synced := false
			ec.syncHandlerFunc = func(ctx context.Context, s string) error {
				synced = true
				return test.syncHandlerErr
			}

			if test.shutdown {
				ec.queue.ShutDown()
			}

			var key string
			ec.queue.Add(key)

			actual := ec.processNextWorkItem(tCtx)
			tCtx.Expect(actual).To(gomega.Equal(!test.expectStopLoop))

			if test.expectSynced {
				tCtx.Expect(synced).To(gomega.BeTrueBecause("sync handler should have run but did not"))
			} else {
				tCtx.Expect(synced).To(gomega.BeFalseBecause("sync handler should not have run but did"))
			}
			time.Sleep(rateLimitDelay)
			// The delayed Add races with checking the length of the queue. Wait
			// for the Add to finish before checking the length.
			tCtx.Wait()
			if test.expectRequeue {
				tCtx.Expect(ec.queue.Len()).To(gomega.Equal(1), "key should have been queued")
			} else {
				tCtx.Expect(ec.queue.Len()).To(gomega.Equal(0), "no key should have been queued")
			}
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
									AdminAccess: new(false),
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
									AdminAccess: new(true),
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
									AdminAccess: new(false),
								},
							},
							{
								Exactly: &resourceapi.ExactDeviceRequest{
									AdminAccess: new(true),
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
									AdminAccess: new(false),
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

func makeClaim(name, namespace string, owner *metav1.OwnerReference) *resourceapi.ResourceClaim {
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	if owner != nil {
		claim.OwnerReferences = []metav1.OwnerReference{*owner}
	}

	return claim
}

func makeTemplatedClaim(podClaimName, generateName, namespace string, createCounter int, owner *metav1.OwnerReference, adminAccess *bool) *resourceapi.ResourceClaim {
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

func makeTemplate(name, namespace string, adminAccess *bool) *resourceapi.ResourceClaimTemplate {
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
	metrics map[controllermetrics.NumResourceClaimLabels]float64
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

	metricName := "dynamic_resource_allocation_resource_claims"

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

			em.metrics[controllermetrics.NumResourceClaimLabels{
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

type expectedMetrics struct {
	numCreated          int
	numCreatedWithAdmin int
	numFailures         int
	numFailureWithAdmin int
}

func expectMetrics(tCtx ktesting.TContext, em expectedMetrics) {
	tCtx.Helper()

	// Check created claims
	actualCreated, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("success", "false"))
	handleErr(tCtx, err, "ResourceClaimCreateSuccesses")
	if actualCreated != float64(em.numCreated) {
		tCtx.Errorf("Expected claims to be created %d, got %v", em.numCreated, actualCreated)
	}

	// Check created claims with admin access
	actualCreatedWithAdmin, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("success", "true"))
	handleErr(tCtx, err, "ResourceClaimCreateSuccessesWithAdminAccess")
	if actualCreatedWithAdmin != float64(em.numCreatedWithAdmin) {
		tCtx.Errorf("Expected claims with admin access to be created %d, got %v", em.numCreatedWithAdmin, actualCreatedWithAdmin)
	}

	// Check failed claims
	actualFailed, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("failure", "false"))
	handleErr(tCtx, err, "ResourceClaimCreateFailures")
	if actualFailed != float64(em.numFailures) {
		tCtx.Errorf("Expected claims to have failed %d, got %v", em.numFailures, actualFailed)
	}

	// Check failed claims with admin access
	actualFailedWithAdmin, err := testutil.GetCounterMetricValue(resourceclaimmetrics.ResourceClaimCreate.WithLabelValues("failure", "true"))
	handleErr(tCtx, err, "ResourceClaimCreateFailuresWithAdminAccess")
	if actualFailedWithAdmin != float64(em.numFailureWithAdmin) {
		tCtx.Errorf("Expected claims with admin access to have failed %d, got %v", em.numFailureWithAdmin, actualFailedWithAdmin)
	}
}
func handleErr(tCtx ktesting.TContext, err error, metricName string) {
	if err != nil {
		tCtx.Errorf("Failed to get %s value, err: %v", metricName, err)
	}
}
func setupMetrics() {
	// Enable test mode to prevent global custom collector registration
	controllermetrics.SetTestMode(true)

	// Reset counter metrics for each test (they are registered by the controller itself)
	resourceclaimmetrics.ResourceClaimCreate.Reset()
}

func newNumMetrics(lister resourcelisters.ResourceClaimLister) numMetrics {
	metrics := make(map[controllermetrics.NumResourceClaimLabels]float64)
	for _, allocated := range []string{"false", "true"} {
		for _, adminAccess := range []string{"false", "true"} {
			for _, source := range []string{"", "extended_resource", "resource_claim_template"} {
				metrics[controllermetrics.NumResourceClaimLabels{
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

func (em numMetrics) withUpdates(rcLabels controllermetrics.NumResourceClaimLabels, n float64) numMetrics {
	em.metrics[rcLabels] += n
	return numMetrics{
		metrics: em.metrics,
		lister:  em.lister,
	}
}

// staticRateLimiter delays any item by a fixed amount of time. It allows a test
// to know exactly how long to wait before rate-limited items are added to the
// queue.
type staticRateLimiter[T comparable] struct {
	delay time.Duration

	requeuesLock sync.Mutex
	requeues     map[T]int
}

// Forget implements [workqueue.TypedRateLimiter].
func (s *staticRateLimiter[T]) Forget(item T) {
	s.requeuesLock.Lock()
	defer s.requeuesLock.Unlock()
	delete(s.requeues, item)
}

// NumRequeues implements [workqueue.TypedRateLimiter].
func (s *staticRateLimiter[T]) NumRequeues(item T) int {
	s.requeuesLock.Lock()
	defer s.requeuesLock.Unlock()
	return s.requeues[item]
}

// When implements [workqueue.TypedRateLimiter].
func (s *staticRateLimiter[T]) When(item T) time.Duration {
	s.requeuesLock.Lock()
	defer s.requeuesLock.Unlock()
	s.requeues[item]++
	return s.delay
}
