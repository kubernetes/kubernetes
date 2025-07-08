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
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	resourcelisters "k8s.io/client-go/listers/resource/v1beta1"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	resourceclaimmetrics "k8s.io/kubernetes/pkg/controller/resourceclaim/metrics"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	testPodName          = "test-pod"
	testNamespace        = "my-namespace"
	testPodUID           = types.UID("uidpod1")
	otherNamespace       = "not-my-namespace"
	podResourceClaimName = "acme-resource"
	templateName         = "my-template"
	className            = "my-resource-class"
	nodeName             = "worker"

	testPod             = makePod(testPodName, testNamespace, testPodUID)
	testPodWithResource = makePod(testPodName, testNamespace, testPodUID, *makePodResourceClaim(podResourceClaimName, templateName))

	otherTestPod = makePod(testPodName+"-II", testNamespace, testPodUID+"-II")

	testClaim              = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, makeOwnerReference(testPodWithResource, true))
	testClaimAllocated     = allocateClaim(testClaim)
	testClaimReserved      = reserveClaim(testClaimAllocated, testPodWithResource)
	testClaimReservedTwice = reserveClaim(testClaimReserved, otherTestPod)
	testClaimKey           = claimKeyPrefix + testClaim.Namespace + "/" + testClaim.Name

	generatedTestClaim          = makeGeneratedClaim(podResourceClaimName, testPodName+"-"+podResourceClaimName+"-", testNamespace, className, 1, makeOwnerReference(testPodWithResource, true), nil)
	generatedTestClaimAllocated = allocateClaim(generatedTestClaim)
	generatedTestClaimReserved  = reserveClaim(generatedTestClaimAllocated, testPodWithResource)

	generatedTestClaimWithAdmin          = makeGeneratedClaim(podResourceClaimName, testPodName+"-"+podResourceClaimName+"-", testNamespace, className, 1, makeOwnerReference(testPodWithResource, true), ptr.To(true))
	generatedTestClaimWithAdminAllocated = allocateClaim(generatedTestClaimWithAdmin)

	conflictingClaim        = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, nil)
	otherNamespaceClaim     = makeClaim(testPodName+"-"+podResourceClaimName, otherNamespace, className, nil)
	template                = makeTemplate(templateName, testNamespace, className, nil)
	templateWithAdminAccess = makeTemplate(templateName, testNamespace, className, ptr.To(true))

	testPodWithNodeName = func() *v1.Pod {
		pod := testPodWithResource.DeepCopy()
		pod.Spec.NodeName = nodeName
		pod.Status.ResourceClaimStatuses = append(pod.Status.ResourceClaimStatuses, v1.PodResourceClaimStatus{
			Name:              pod.Spec.ResourceClaims[0].Name,
			ResourceClaimName: &generatedTestClaim.Name,
		})
		return pod
	}()
	adminAccessFeatureOffError = "admin access is requested, but the feature is disabled"
)

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name                   string
		key                    string
		adminAccessEnabled     bool
		prioritizedListEnabled bool
		claims                 []*resourceapi.ResourceClaim
		claimsInCache          []*resourceapi.ResourceClaim
		pods                   []*v1.Pod
		podsLater              []*v1.Pod
		templates              []*resourceapi.ResourceClaimTemplate
		expectedClaims         []resourceapi.ResourceClaim
		expectedStatuses       map[string][]v1.PodResourceClaimStatus
		expectedError          string
		expectedMetrics        expectedMetrics
	}{
		{
			name:           "create",
			pods:           []*v1.Pod{testPodWithResource},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			expectedClaims: []resourceapi.ResourceClaim{*generatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
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
			expectedClaims: []resourceapi.ResourceClaim{*generatedTestClaimWithAdmin},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaimWithAdmin.Name},
				},
			},
			adminAccessEnabled: true,
			expectedMetrics:    expectedMetrics{0, 1, 0, 0},
		},
		{
			name: "nop",
			pods: []*v1.Pod{func() *v1.Pod {
				pod := testPodWithResource.DeepCopy()
				pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
				}
				return pod
			}()},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			claims:         []*resourceapi.ResourceClaim{generatedTestClaim},
			expectedClaims: []resourceapi.ResourceClaim{*generatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "recreate",
			pods: []*v1.Pod{func() *v1.Pod {
				pod := testPodWithResource.DeepCopy()
				pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
				}
				return pod
			}()},
			templates:      []*resourceapi.ResourceClaimTemplate{template},
			key:            podKey(testPodWithResource),
			expectedClaims: []resourceapi.ResourceClaim{*generatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
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
			name:           "find-existing-claim-by-label",
			pods:           []*v1.Pod{testPodWithResource},
			key:            podKey(testPodWithResource),
			claims:         []*resourceapi.ResourceClaim{generatedTestClaim},
			expectedClaims: []resourceapi.ResourceClaim{*generatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name:          "find-created-claim-in-cache",
			pods:          []*v1.Pod{testPodWithResource},
			key:           podKey(testPodWithResource),
			claimsInCache: []*resourceapi.ResourceClaim{generatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
		{
			name: "no-such-pod",
			key:  podKey(testPodWithResource),
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
			expectedClaims: []resourceapi.ResourceClaim{*otherNamespaceClaim, *generatedTestClaim},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithResource.Name: {
					{Name: testPodWithResource.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
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
			name:            "create-conflict",
			pods:            []*v1.Pod{testPodWithResource},
			templates:       []*resourceapi.ResourceClaimTemplate{template},
			key:             podKey(testPodWithResource),
			expectedMetrics: expectedMetrics{1, 0, 1, 0},
			expectedError:   "create ResourceClaim : Operation cannot be fulfilled on resourceclaims.resource.k8s.io \"fake name\": fake conflict",
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
			claims:         []*resourceapi.ResourceClaim{generatedTestClaimAllocated},
			expectedClaims: []resourceapi.ResourceClaim{*generatedTestClaimReserved},
			expectedStatuses: map[string][]v1.PodResourceClaimStatus{
				testPodWithNodeName.Name: {
					{Name: testPodWithNodeName.Spec.ResourceClaims[0].Name, ResourceClaimName: &generatedTestClaim.Name},
				},
			},
			expectedMetrics: expectedMetrics{0, 0, 0, 0},
		},
	}

	for _, tc := range tests {
		// Run sequentially because of global logging and global metrics.
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			tCtx = ktesting.WithCancel(tCtx)

			var objects []runtime.Object
			for _, pod := range tc.pods {
				objects = append(objects, pod)
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
			claimInformer := informerFactory.Resource().V1beta1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1beta1().ResourceClaimTemplates()
			setupMetrics()

			features := Features{
				AdminAccess:     tc.adminAccessEnabled,
				PrioritizedList: tc.prioritizedListEnabled,
			}
			ec, err := NewController(tCtx.Logger(), features, fakeKubeClient, podInformer, claimInformer, templateInformer)
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

			claims, err := fakeKubeClient.ResourceV1beta1().ResourceClaims("").List(tCtx, metav1.ListOptions{})
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

			expectMetrics(t, tc.expectedMetrics)
		})
	}
}

func TestResourceClaimEventHandler(t *testing.T) {
	tCtx := ktesting.Init(t)
	tCtx = ktesting.WithCancel(tCtx)

	fakeKubeClient := createTestClient()
	informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
	podInformer := informerFactory.Core().V1().Pods()
	claimInformer := informerFactory.Resource().V1beta1().ResourceClaims()
	templateInformer := informerFactory.Resource().V1beta1().ResourceClaimTemplates()
	setupMetrics()
	claimClient := fakeKubeClient.ResourceV1beta1().ResourceClaims(testNamespace)

	ec, err := NewController(tCtx.Logger(), Features{}, fakeKubeClient, podInformer, claimInformer, templateInformer)
	tCtx.ExpectNoError(err, "creating ephemeral controller")

	informerFactory.Start(tCtx.Done())
	stopInformers := func() {
		tCtx.Cancel("stopping informers")
		informerFactory.Shutdown()
	}
	defer stopInformers()

	em := newNumMetrics(claimInformer.Lister(), 0, 0, 0, 0)

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
	em = em.withUpdates(1, 0, 0, 0)
	ktesting.Step(tCtx, "create claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	modifiedClaim := testClaim.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	ktesting.Step(tCtx, "modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	_, err = claimClient.Update(tCtx, testClaimAllocated, metav1.UpdateOptions{})
	em = em.withUpdates(-1, 0, 1, 0)
	ktesting.Step(tCtx, "allocate claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	modifiedClaim = testClaimAllocated.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar2"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	ktesting.Step(tCtx, "modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	otherClaimAllocated := testClaimAllocated.DeepCopy()
	otherClaimAllocated.Name += "2"
	_, err = claimClient.Create(tCtx, otherClaimAllocated, metav1.CreateOptions{})
	em = em.withUpdates(0, 0, 1, 0)
	ktesting.Step(tCtx, "create allocated claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey + "2"})
	})

	_, err = claimClient.Update(tCtx, testClaim, metav1.UpdateOptions{})
	em = em.withUpdates(1, 0, -1, 0)
	ktesting.Step(tCtx, "deallocate claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{testClaimKey})
	})

	err = claimClient.Delete(tCtx, testClaim.Name, metav1.DeleteOptions{})
	em = em.withUpdates(-1, 0, 0, 0)
	ktesting.Step(tCtx, "delete deallocated claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{})
	})

	err = claimClient.Delete(tCtx, otherClaimAllocated.Name, metav1.DeleteOptions{})
	em = em.withUpdates(0, 0, -1, 0)
	ktesting.Step(tCtx, "delete allocated claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
		expectQueue(tCtx, []string{})
	})

	_, err = claimClient.Create(tCtx, generatedTestClaimWithAdmin, metav1.CreateOptions{})
	em = em.withUpdates(0, 1, 0, 0)
	ktesting.Step(tCtx, "create claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	modifiedClaim = generatedTestClaimWithAdmin.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	ktesting.Step(tCtx, "modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
	})

	_, err = claimClient.Update(tCtx, generatedTestClaimWithAdminAllocated, metav1.UpdateOptions{})
	em = em.withUpdates(0, -1, 0, 1)
	ktesting.Step(tCtx, "allocate claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	modifiedClaim = generatedTestClaimWithAdminAllocated.DeepCopy()
	modifiedClaim.Labels = map[string]string{"foo": "bar2"}
	_, err = claimClient.Update(tCtx, modifiedClaim, metav1.UpdateOptions{})
	ktesting.Step(tCtx, "modify claim", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Consistently(tCtx)
	})

	otherClaimAllocated = generatedTestClaimWithAdminAllocated.DeepCopy()
	otherClaimAllocated.Name += "2"
	_, err = claimClient.Create(tCtx, otherClaimAllocated, metav1.CreateOptions{})
	em = em.withUpdates(0, 0, 0, 1)
	ktesting.Step(tCtx, "create allocated claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	_, err = claimClient.Update(tCtx, generatedTestClaimWithAdmin, metav1.UpdateOptions{})
	em = em.withUpdates(0, 1, 0, -1)
	ktesting.Step(tCtx, "deallocate claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	err = claimClient.Delete(tCtx, generatedTestClaimWithAdmin.Name, metav1.DeleteOptions{})
	em = em.withUpdates(0, -1, 0, 0)
	ktesting.Step(tCtx, "delete deallocated claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	err = claimClient.Delete(tCtx, otherClaimAllocated.Name, metav1.DeleteOptions{})
	em = em.withUpdates(0, 0, 0, -1)
	ktesting.Step(tCtx, "delete allocated claim with admin access", func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(err)
		em.Eventually(tCtx)
	})

	em.Consistently(tCtx)
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
								AdminAccess: ptr.To(false),
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
								AdminAccess: ptr.To(true),
							},
						},
					},
				},
			},
			want: "true",
		},
		{
			name: "multiple requests, one with admin access true",
			claim: &resourceapi.ResourceClaim{
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{
							{
								AdminAccess: ptr.To(false),
							},
							{
								AdminAccess: ptr.To(true),
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
								AdminAccess: nil,
							},
							{
								AdminAccess: ptr.To(false),
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

func makeGeneratedClaim(podClaimName, generateName, namespace, classname string, createCounter int, owner *metav1.OwnerReference, adminAccess *bool) *resourceapi.ResourceClaim {
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:         fmt.Sprintf("%s-%d", generateName, createCounter),
			GenerateName: generateName,
			Namespace:    namespace,
			Annotations:  map[string]string{"resource.kubernetes.io/pod-claim-name": podClaimName},
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
						Name:            "req-0",
						DeviceClassName: "class",
						AdminAccess:     adminAccess,
					},
				},
			},
		}
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

func reserveClaim(claim *resourceapi.ResourceClaim, pod *v1.Pod) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	claim.Status.ReservedFor = append(claim.Status.ReservedFor,
		resourceapi.ResourceClaimConsumerReference{
			Resource: "pods",
			Name:     pod.Name,
			UID:      pod.UID,
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
							Name:            "req-0",
							DeviceClassName: "class",
							AdminAccess:     adminAccess,
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

func claimKey(claim *resourceapi.ResourceClaim) string {
	return claimKeyPrefix + claim.Namespace + "/" + claim.Name
}

func makeOwnerReference(pod *v1.Pod, isController bool) *metav1.OwnerReference {
	isTrue := true
	return &metav1.OwnerReference{
		APIVersion:         "v1",
		Kind:               "Pod",
		Name:               pod.Name,
		UID:                pod.UID,
		Controller:         &isController,
		BlockOwnerDeletion: &isTrue,
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
	notAllocated                float64
	notAllocatedWithAdminAccess float64
	allocated                   float64
	allocatedWithAdminAccess    float64
	lister                      resourcelisters.ResourceClaimLister
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
			value := metric.GetGauge().GetValue()

			switch {
			case allocated == "false" && adminAccess == "false":
				em.notAllocated = value
			case allocated == "false" && adminAccess == "true":
				em.notAllocatedWithAdminAccess = value
			case allocated == "true" && adminAccess == "false":
				em.allocated = value
			case allocated == "true" && adminAccess == "true":
				em.allocatedWithAdminAccess = value
			}
		}
	}

	return em, nil
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

func newNumMetrics(lister resourcelisters.ResourceClaimLister, notAllocated, notAllocatedWithAdmin, allocated, allocatedWithAdmin float64) numMetrics {
	return numMetrics{
		notAllocated:                notAllocated,
		notAllocatedWithAdminAccess: notAllocatedWithAdmin,
		allocated:                   allocated,
		allocatedWithAdminAccess:    allocatedWithAdmin,
		lister:                      lister,
	}
}

func (em numMetrics) withUpdates(notAllocatedDelta, notAllocatedWithAdminDelta, allocatedDelta, allocatedWithAdminDelta float64) numMetrics {
	return numMetrics{
		notAllocated:                em.notAllocated + notAllocatedDelta,
		notAllocatedWithAdminAccess: em.notAllocatedWithAdminAccess + notAllocatedWithAdminDelta,
		allocated:                   em.allocated + allocatedDelta,
		allocatedWithAdminAccess:    em.allocatedWithAdminAccess + allocatedWithAdminDelta,
		lister:                      em.lister,
	}
}
