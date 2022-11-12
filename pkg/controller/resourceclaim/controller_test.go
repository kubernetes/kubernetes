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
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	ephemeralvolumemetrics "k8s.io/kubernetes/pkg/controller/resourceclaim/metrics"
)

var (
	testPodName          = "test-pod"
	testNamespace        = "my-namespace"
	testPodUID           = types.UID("uidpod1")
	otherNamespace       = "not-my-namespace"
	podResourceClaimName = "acme-resource"
	templateName         = "my-template"
	className            = "my-resource-class"

	testPod             = makePod(testPodName, testNamespace, testPodUID)
	testPodWithResource = makePod(testPodName, testNamespace, testPodUID, *makePodResourceClaim(podResourceClaimName, templateName))
	otherTestPod        = makePod(testPodName+"-II", testNamespace, testPodUID+"-II")
	testClaim           = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, makeOwnerReference(testPodWithResource, true))
	testClaimReserved   = func() *resourcev1alpha1.ResourceClaim {
		claim := testClaim.DeepCopy()
		claim.Status.ReservedFor = append(claim.Status.ReservedFor,
			resourcev1alpha1.ResourceClaimConsumerReference{
				Resource: "pods",
				Name:     testPodWithResource.Name,
				UID:      testPodWithResource.UID,
			},
		)
		return claim
	}()
	testClaimReservedTwice = func() *resourcev1alpha1.ResourceClaim {
		claim := testClaimReserved.DeepCopy()
		claim.Status.ReservedFor = append(claim.Status.ReservedFor,
			resourcev1alpha1.ResourceClaimConsumerReference{
				Resource: "pods",
				Name:     otherTestPod.Name,
				UID:      otherTestPod.UID,
			},
		)
		return claim
	}()
	conflictingClaim    = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, className, nil)
	otherNamespaceClaim = makeClaim(testPodName+"-"+podResourceClaimName, otherNamespace, className, nil)
	template            = makeTemplate(templateName, testNamespace, className)
)

func init() {
	klog.InitFlags(nil)
}

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name            string
		key             string
		claims          []*resourcev1alpha1.ResourceClaim
		pods            []*v1.Pod
		podsLater       []*v1.Pod
		templates       []*resourcev1alpha1.ResourceClaimTemplate
		expectedClaims  []resourcev1alpha1.ResourceClaim
		expectedError   bool
		expectedMetrics expectedMetrics
	}{
		{
			name:            "create",
			pods:            []*v1.Pod{testPodWithResource},
			templates:       []*resourcev1alpha1.ResourceClaimTemplate{template},
			key:             podKey(testPodWithResource),
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*testClaim},
			expectedMetrics: expectedMetrics{1, 0},
		},
		{
			name:          "missing-template",
			pods:          []*v1.Pod{testPodWithResource},
			templates:     nil,
			key:           podKey(testPodWithResource),
			expectedError: true,
		},
		{
			name:            "nop",
			pods:            []*v1.Pod{testPodWithResource},
			key:             podKey(testPodWithResource),
			claims:          []*resourcev1alpha1.ResourceClaim{testClaim},
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*testClaim},
			expectedMetrics: expectedMetrics{0, 0},
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
			name:            "create-with-other-claim",
			pods:            []*v1.Pod{testPodWithResource},
			templates:       []*resourcev1alpha1.ResourceClaimTemplate{template},
			key:             podKey(testPodWithResource),
			claims:          []*resourcev1alpha1.ResourceClaim{otherNamespaceClaim},
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*otherNamespaceClaim, *testClaim},
			expectedMetrics: expectedMetrics{1, 0},
		},
		{
			name:           "wrong-claim-owner",
			pods:           []*v1.Pod{testPodWithResource},
			key:            podKey(testPodWithResource),
			claims:         []*resourcev1alpha1.ResourceClaim{conflictingClaim},
			expectedClaims: []resourcev1alpha1.ResourceClaim{*conflictingClaim},
			expectedError:  true,
		},
		{
			name:            "create-conflict",
			pods:            []*v1.Pod{testPodWithResource},
			templates:       []*resourcev1alpha1.ResourceClaimTemplate{template},
			key:             podKey(testPodWithResource),
			expectedMetrics: expectedMetrics{1, 1},
			expectedError:   true,
		},
		{
			name:            "stay-reserved-seen",
			pods:            []*v1.Pod{testPodWithResource},
			key:             claimKey(testClaimReserved),
			claims:          []*resourcev1alpha1.ResourceClaim{testClaimReserved},
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*testClaimReserved},
			expectedMetrics: expectedMetrics{0, 0},
		},
		{
			name:            "stay-reserved-not-seen",
			podsLater:       []*v1.Pod{testPodWithResource},
			key:             claimKey(testClaimReserved),
			claims:          []*resourcev1alpha1.ResourceClaim{testClaimReserved},
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*testClaimReserved},
			expectedMetrics: expectedMetrics{0, 0},
		},
		{
			name:            "clear-reserved",
			pods:            []*v1.Pod{},
			key:             claimKey(testClaimReserved),
			claims:          []*resourcev1alpha1.ResourceClaim{testClaimReserved},
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*testClaim},
			expectedMetrics: expectedMetrics{0, 0},
		},
		{
			name:            "remove-reserved",
			pods:            []*v1.Pod{testPod},
			key:             claimKey(testClaimReservedTwice),
			claims:          []*resourcev1alpha1.ResourceClaim{testClaimReservedTwice},
			expectedClaims:  []resourcev1alpha1.ResourceClaim{*testClaimReserved},
			expectedMetrics: expectedMetrics{0, 0},
		},
	}

	for _, tc := range tests {
		// Run sequentially because of global logging and global metrics.
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

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
			setupMetrics()
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			claimInformer := informerFactory.Resource().V1alpha1().ResourceClaims()
			templateInformer := informerFactory.Resource().V1alpha1().ResourceClaimTemplates()

			ec, err := NewController(fakeKubeClient, podInformer, claimInformer, templateInformer)
			if err != nil {
				t.Fatalf("error creating ephemeral controller : %v", err)
			}

			// Ensure informers are up-to-date.
			go informerFactory.Start(ctx.Done())
			stopInformers := func() {
				cancel()
				informerFactory.Shutdown()
			}
			defer stopInformers()
			informerFactory.WaitForCacheSync(ctx.Done())
			cache.WaitForCacheSync(ctx.Done(), podInformer.Informer().HasSynced, claimInformer.Informer().HasSynced, templateInformer.Informer().HasSynced)

			// Simulate race: stop informers, add more pods that the controller doesn't know about.
			stopInformers()
			for _, pod := range tc.podsLater {
				_, err := fakeKubeClient.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("unexpected error while creating pod: %v", err)
				}
			}

			err = ec.syncHandler(context.TODO(), tc.key)
			if err != nil && !tc.expectedError {
				t.Fatalf("unexpected error while running handler: %v", err)
			}
			if err == nil && tc.expectedError {
				t.Fatalf("unexpected success")
			}

			claims, err := fakeKubeClient.ResourceV1alpha1().ResourceClaims("").List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("unexpected error while listing claims: %v", err)
			}
			assert.Equal(t, normalizeClaims(tc.expectedClaims), normalizeClaims(claims.Items))
			expectMetrics(t, tc.expectedMetrics)
		})
	}
}

func makeClaim(name, namespace, classname string, owner *metav1.OwnerReference) *resourcev1alpha1.ResourceClaim {
	claim := &resourcev1alpha1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: resourcev1alpha1.ResourceClaimSpec{
			ResourceClassName: classname,
		},
	}
	if owner != nil {
		claim.OwnerReferences = []metav1.OwnerReference{*owner}
	}

	return claim
}

func makePodResourceClaim(name, templateName string) *v1.PodResourceClaim {
	return &v1.PodResourceClaim{
		Name: name,
		Source: v1.ClaimSource{
			ResourceClaimTemplateName: &templateName,
		},
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

func makeTemplate(name, namespace, classname string) *resourcev1alpha1.ResourceClaimTemplate {
	template := &resourcev1alpha1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: resourcev1alpha1.ResourceClaimTemplateSpec{
			Spec: resourcev1alpha1.ResourceClaimSpec{
				ResourceClassName: classname,
			},
		},
	}
	return template
}

func podKey(pod *v1.Pod) string {
	return podKeyPrefix + pod.Namespace + "/" + pod.Name
}

func claimKey(claim *resourcev1alpha1.ResourceClaim) string {
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

func normalizeClaims(claims []resourcev1alpha1.ResourceClaim) []resourcev1alpha1.ResourceClaim {
	sort.Slice(claims, func(i, j int) bool {
		return claims[i].Namespace < claims[j].Namespace ||
			claims[i].Name < claims[j].Name
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
	return fakeClient
}

// Metrics helpers

type expectedMetrics struct {
	numCreated  int
	numFailures int
}

func expectMetrics(t *testing.T, em expectedMetrics) {
	t.Helper()

	actualCreated, err := testutil.GetCounterMetricValue(ephemeralvolumemetrics.ResourceClaimCreateAttempts)
	handleErr(t, err, "ResourceClaimCreate")
	if actualCreated != float64(em.numCreated) {
		t.Errorf("Expected claims to be created %d, got %v", em.numCreated, actualCreated)
	}
	actualConflicts, err := testutil.GetCounterMetricValue(ephemeralvolumemetrics.ResourceClaimCreateFailures)
	handleErr(t, err, "ResourceClaimCreate/Conflict")
	if actualConflicts != float64(em.numFailures) {
		t.Errorf("Expected claims to have conflicts %d, got %v", em.numFailures, actualConflicts)
	}
}

func handleErr(t *testing.T, err error, metricName string) {
	if err != nil {
		t.Errorf("Failed to get %s value, err: %v", metricName, err)
	}
}

func setupMetrics() {
	ephemeralvolumemetrics.RegisterMetrics()
	ephemeralvolumemetrics.ResourceClaimCreateAttempts.Reset()
	ephemeralvolumemetrics.ResourceClaimCreateFailures.Reset()
}
