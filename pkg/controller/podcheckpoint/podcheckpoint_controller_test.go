/*
Copyright The Kubernetes Authors.

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

package podcheckpoint

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
)

func TestSyncHandlerRestoreLock(t *testing.T) {
	// pod builds a Pod that restores from checkpoint "cp" in the given phase.
	pod := func(name string, phase v1.PodPhase) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
			Spec:       v1.PodSpec{RestoreFrom: &v1.CheckpointReference{Name: "cp"}},
			Status:     v1.PodStatus{Phase: phase},
		}
	}
	// checkpoint builds the "cp" PodCheckpoint, optionally carrying the finalizer.
	checkpoint := func(withFinalizer bool) *nodev1alpha1.PodCheckpoint {
		checkpoint := &nodev1alpha1.PodCheckpoint{
			ObjectMeta: metav1.ObjectMeta{Name: "cp", Namespace: "ns"},
		}
		if withFinalizer {
			checkpoint.Finalizers = []string{RestoreLockFinalizer}
		}
		return checkpoint
	}

	tests := []struct {
		name          string
		withFinalizer bool
		pods          []*v1.Pod
		deleting      bool
		livePods      []*v1.Pod
		wantFinalizer bool
	}{
		{name: "creation installs protection", wantFinalizer: true},
		{name: "inactive checkpoint stays protected", withFinalizer: true, wantFinalizer: true},
		{name: "deleting unused checkpoint releases protection", withFinalizer: true, deleting: true},
		{name: "cached pending Pod retains protection", withFinalizer: true, deleting: true, pods: []*v1.Pod{pod("restore", v1.PodPending)}, wantFinalizer: true},
		{name: "live Pod missing from cache retains protection", withFinalizer: true, deleting: true, livePods: []*v1.Pod{pod("restore", v1.PodPending)}, wantFinalizer: true},
		{name: "newly admitted Pod retains protection", withFinalizer: true, deleting: true, livePods: []*v1.Pod{pod("restore", "")}, wantFinalizer: true},
		{name: "terminal Pod releases protection", withFinalizer: true, deleting: true, livePods: []*v1.Pod{pod("restore", v1.PodFailed)}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()

			podIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			for _, p := range tc.pods {
				require.NoError(t, podIndexer.Add(p))
			}

			pc := checkpoint(tc.withFinalizer)
			if tc.deleting {
				pc.DeletionTimestamp = &metav1.Time{Time: time.Now()}
			}
			objects := []runtime.Object{pc}
			for _, pod := range tc.livePods {
				objects = append(objects, pod)
			}
			client := fake.NewSimpleClientset(objects...)

			c := &Controller{kubeClient: client, podLister: corelisters.NewPodLister(podIndexer)}

			require.NoError(t, c.syncHandler(ctx, "ns/cp"))

			obj, err := client.NodeV1alpha1().PodCheckpoints("ns").Get(ctx, "cp", metav1.GetOptions{})
			require.NoError(t, err)
			require.Equal(t, tc.wantFinalizer, slices.Contains(obj.Finalizers, RestoreLockFinalizer))
		})
	}
}
