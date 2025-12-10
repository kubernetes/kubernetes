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

package ephemeral

import (
	"context"
	"errors"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
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
	ephemeralvolumemetrics "k8s.io/kubernetes/pkg/controller/volume/ephemeral/metrics"
)

var (
	testPodName         = "test-pod"
	testNamespace       = "my-namespace"
	testPodUID          = types.UID("uidpod1")
	otherNamespace      = "not-my-namespace"
	ephemeralVolumeName = "ephemeral-volume"

	testPod               = makePod(testPodName, testNamespace, testPodUID)
	testPodWithEphemeral  = makePod(testPodName, testNamespace, testPodUID, *makeEphemeralVolume(ephemeralVolumeName))
	testPodEphemeralClaim = makePVC(testPodName+"-"+ephemeralVolumeName, testNamespace, makeOwnerReference(testPodWithEphemeral, true))
	conflictingClaim      = makePVC(testPodName+"-"+ephemeralVolumeName, testNamespace, nil)
	otherNamespaceClaim   = makePVC(testPodName+"-"+ephemeralVolumeName, otherNamespace, nil)
)

func init() {
	klog.InitFlags(nil)
}

func TestSyncHandler(t *testing.T) {
	tests := []struct {
		name            string
		podKey          string
		pvcs            []*v1.PersistentVolumeClaim
		pods            []*v1.Pod
		expectedPVCs    []v1.PersistentVolumeClaim
		expectedError   bool
		expectedMetrics expectedMetrics
	}{
		{
			name:            "create",
			pods:            []*v1.Pod{testPodWithEphemeral},
			podKey:          podKey(testPodWithEphemeral),
			expectedPVCs:    []v1.PersistentVolumeClaim{*testPodEphemeralClaim},
			expectedMetrics: expectedMetrics{1, 0},
		},
		{
			name:   "no-such-pod",
			podKey: podKey(testPodWithEphemeral),
		},
		{
			name: "pod-deleted",
			pods: func() []*v1.Pod {
				deleted := metav1.Now()
				pods := []*v1.Pod{testPodWithEphemeral.DeepCopy()}
				pods[0].DeletionTimestamp = &deleted
				return pods
			}(),
			podKey: podKey(testPodWithEphemeral),
		},
		{
			name:   "no-volumes",
			pods:   []*v1.Pod{testPod},
			podKey: podKey(testPod),
		},
		{
			name:            "create-with-other-PVC",
			pods:            []*v1.Pod{testPodWithEphemeral},
			podKey:          podKey(testPodWithEphemeral),
			pvcs:            []*v1.PersistentVolumeClaim{otherNamespaceClaim},
			expectedPVCs:    []v1.PersistentVolumeClaim{*otherNamespaceClaim, *testPodEphemeralClaim},
			expectedMetrics: expectedMetrics{1, 0},
		},
		{
			name:          "wrong-PVC-owner",
			pods:          []*v1.Pod{testPodWithEphemeral},
			podKey:        podKey(testPodWithEphemeral),
			pvcs:          []*v1.PersistentVolumeClaim{conflictingClaim},
			expectedPVCs:  []v1.PersistentVolumeClaim{*conflictingClaim},
			expectedError: true,
		},
		{
			name:            "create-conflict",
			pods:            []*v1.Pod{testPodWithEphemeral},
			podKey:          podKey(testPodWithEphemeral),
			expectedMetrics: expectedMetrics{1, 1},
			expectedError:   true,
		},
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
			for _, pvc := range tc.pvcs {
				objects = append(objects, pvc)
			}

			fakeKubeClient := createTestClient(objects...)
			if tc.expectedMetrics.numFailures > 0 {
				fakeKubeClient.PrependReactor("create", "persistentvolumeclaims", func(action k8stesting.Action) (handled bool, ret runtime.Object, err error) {
					return true, nil, apierrors.NewConflict(action.GetResource().GroupResource(), "fake name", errors.New("fake conflict"))
				})
			}
			setupMetrics()
			informerFactory := informers.NewSharedInformerFactory(fakeKubeClient, controller.NoResyncPeriodFunc())
			podInformer := informerFactory.Core().V1().Pods()
			pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()

			c, err := NewController(ctx, fakeKubeClient, podInformer, pvcInformer)
			if err != nil {
				t.Fatalf("error creating ephemeral controller : %v", err)
			}
			ec, _ := c.(*ephemeralController)

			// Ensure informers are up-to-date.
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())
			cache.WaitForCacheSync(ctx.Done(), podInformer.Informer().HasSynced, pvcInformer.Informer().HasSynced)

			err = ec.syncHandler(context.TODO(), tc.podKey)
			if err != nil && !tc.expectedError {
				t.Fatalf("unexpected error while running handler: %v", err)
			}
			if err == nil && tc.expectedError {
				t.Fatalf("unexpected success")
			}

			pvcs, err := fakeKubeClient.CoreV1().PersistentVolumeClaims("").List(ctx, metav1.ListOptions{})
			if err != nil {
				t.Fatalf("unexpected error while listing PVCs: %v", err)
			}
			assert.Equal(t, sortPVCs(tc.expectedPVCs), sortPVCs(pvcs.Items))
			expectMetrics(t, tc.expectedMetrics)
		})
	}
}

func makePVC(name, namespace string, owner *metav1.OwnerReference) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       v1.PersistentVolumeClaimSpec{},
	}
	if owner != nil {
		pvc.OwnerReferences = []metav1.OwnerReference{*owner}
	}

	return pvc
}

func makeEphemeralVolume(name string) *v1.Volume {
	return &v1.Volume{
		Name: name,
		VolumeSource: v1.VolumeSource{
			Ephemeral: &v1.EphemeralVolumeSource{
				VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{},
			},
		},
	}
}

func makePod(name, namespace string, uid types.UID, volumes ...v1.Volume) *v1.Pod {
	pvc := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, UID: uid},
		Spec: v1.PodSpec{
			Volumes: volumes,
		},
	}

	return pvc
}

func podKey(pod *v1.Pod) string {
	key, _ := cache.DeletionHandlingMetaNamespaceKeyFunc(testPodWithEphemeral)
	return key
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

func sortPVCs(pvcs []v1.PersistentVolumeClaim) []v1.PersistentVolumeClaim {
	sort.Slice(pvcs, func(i, j int) bool {
		return pvcs[i].Namespace < pvcs[j].Namespace ||
			pvcs[i].Name < pvcs[j].Name
	})
	return pvcs
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

	actualCreated, err := testutil.GetCounterMetricValue(ephemeralvolumemetrics.EphemeralVolumeCreateAttempts)
	handleErr(t, err, "ephemeralVolumeCreate")
	if actualCreated != float64(em.numCreated) {
		t.Errorf("Expected PVCs to be created %d, got %v", em.numCreated, actualCreated)
	}
	actualConflicts, err := testutil.GetCounterMetricValue(ephemeralvolumemetrics.EphemeralVolumeCreateFailures)
	handleErr(t, err, "ephemeralVolumeCreate/Conflict")
	if actualConflicts != float64(em.numFailures) {
		t.Errorf("Expected PVCs to have conflicts %d, got %v", em.numFailures, actualConflicts)
	}
}

func handleErr(t *testing.T, err error, metricName string) {
	if err != nil {
		t.Errorf("Failed to get %s value, err: %v", metricName, err)
	}
}

func setupMetrics() {
	ephemeralvolumemetrics.RegisterMetrics()
	ephemeralvolumemetrics.EphemeralVolumeCreateAttempts.Reset()
	ephemeralvolumemetrics.EphemeralVolumeCreateFailures.Reset()
}
