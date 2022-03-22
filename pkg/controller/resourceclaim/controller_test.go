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

	corev1 "k8s.io/api/core/v1"
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

	"github.com/stretchr/testify/assert"
)

var (
	testPodName          = "test-pod"
	testNamespace        = "my-namespace"
	testPodUID           = types.UID("uidpod1")
	otherNamespace       = "not-my-namespace"
	podResourceClaimName = "acme-resource"

	testPod             = makePod(testPodName, testNamespace, testPodUID)
	testPodWithResource = makePod(testPodName, testNamespace, testPodUID, *makePodResourceClaim(podResourceClaimName))
	testClaim           = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, makeOwnerReference(testPodWithResource, true))
	conflictingClaim    = makeClaim(testPodName+"-"+podResourceClaimName, testNamespace, nil)
	otherNamespaceClaim = makeClaim(testPodName+"-"+podResourceClaimName, otherNamespace, nil)
)

func init() {
	klog.InitFlags(nil)
}

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name            string
		key             string
		claims          []*corev1.ResourceClaim
		pods            []*corev1.Pod
		expectedClaims  []corev1.ResourceClaim
		expectedError   bool
		expectedMetrics expectedMetrics
	}{
		{
			name:            "create",
			pods:            []*corev1.Pod{testPodWithResource},
			key:             podKey(testPodWithResource),
			expectedClaims:  []corev1.ResourceClaim{*testClaim},
			expectedMetrics: expectedMetrics{1, 0},
		},
		{
			name: "no-such-pod",
			key:  podKey(testPodWithResource),
		},
		{
			name: "pod-deleted",
			pods: func() []*corev1.Pod {
				deleted := metav1.Now()
				pods := []*corev1.Pod{testPodWithResource.DeepCopy()}
				pods[0].DeletionTimestamp = &deleted
				return pods
			}(),
			key: podKey(testPodWithResource),
		},
		{
			name: "no-volumes",
			pods: []*corev1.Pod{testPod},
			key:  podKey(testPod),
		},
		{
			name:            "create-with-other-claim",
			pods:            []*corev1.Pod{testPodWithResource},
			key:             podKey(testPodWithResource),
			claims:          []*corev1.ResourceClaim{otherNamespaceClaim},
			expectedClaims:  []corev1.ResourceClaim{*otherNamespaceClaim, *testClaim},
			expectedMetrics: expectedMetrics{1, 0},
		},
		{
			name:           "wrong-claim-owner",
			pods:           []*corev1.Pod{testPodWithResource},
			key:            podKey(testPodWithResource),
			claims:         []*corev1.ResourceClaim{conflictingClaim},
			expectedClaims: []corev1.ResourceClaim{*conflictingClaim},
			expectedError:  true,
		},
		{
			name:            "create-conflict",
			pods:            []*corev1.Pod{testPodWithResource},
			key:             podKey(testPodWithResource),
			expectedMetrics: expectedMetrics{1, 1},
			expectedError:   true,
		},

		// TODO: tests for ReservedFor cleanup
	}

	for _, tc := range tests {
		// Run sequentially because of global logging and global metrics.
		t.Run(tc.name, func(t *testing.T) {
			// There is no good way to shut down the informers. They spawn
			// various goroutines and some of them (in particular shared informer)
			// become very unhappy ("close on closed channel") when using a context
			// that gets cancelled. Therefore we just keep everything running.
			ctx := context.Background()

			var objects []runtime.Object
			for _, pod := range tc.pods {
				objects = append(objects, pod)
			}
			for _, claim := range tc.claims {
				objects = append(objects, claim)
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
			claimInformer := informerFactory.Core().V1().ResourceClaims()

			c, err := NewController(fakeKubeClient, podInformer, claimInformer)
			if err != nil {
				t.Fatalf("error creating ephemeral controller : %v", err)
			}
			ec, _ := c.(*resourceClaimController)

			// Ensure informers are up-to-date.
			go informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			cache.WaitForCacheSync(ctx.Done(), podInformer.Informer().HasSynced, claimInformer.Informer().HasSynced)

			err = ec.syncHandler(context.TODO(), tc.key)
			if err != nil && !tc.expectedError {
				t.Fatalf("unexpected error while running handler: %v", err)
			}
			if err == nil && tc.expectedError {
				t.Fatalf("unexpected success")
			}

			claims, err := fakeKubeClient.CoreV1().ResourceClaims("").List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("unexpected error while listing claims: %v", err)
			}
			assert.Equal(t, sortClaims(tc.expectedClaims), sortClaims(claims.Items))
			expectMetrics(t, tc.expectedMetrics)
		})
	}
}

func makeClaim(name, namespace string, owner *metav1.OwnerReference) *corev1.ResourceClaim {
	claim := &corev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
	}
	if owner != nil {
		claim.OwnerReferences = []metav1.OwnerReference{*owner}
	}

	return claim
}

func makePodResourceClaim(name string) *corev1.PodResourceClaim {
	return &corev1.PodResourceClaim{
		Name: name,
		Claim: corev1.ClaimSource{
			Template: &corev1.ResourceClaimTemplate{},
		},
	}
}

func makePod(name, namespace string, uid types.UID, podClaims ...corev1.PodResourceClaim) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, UID: uid},
		Spec: corev1.PodSpec{
			ResourceClaims: podClaims,
		},
	}

	return pod
}

func podKey(pod *corev1.Pod) string {
	return podKeyPrefix + pod.Namespace + "/" + pod.Name
}

func makeOwnerReference(pod *corev1.Pod, isController bool) *metav1.OwnerReference {
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

func sortClaims(claims []corev1.ResourceClaim) []corev1.ResourceClaim {
	sort.Slice(claims, func(i, j int) bool {
		return claims[i].Namespace < claims[j].Namespace ||
			claims[i].Name < claims[j].Name
	})
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
