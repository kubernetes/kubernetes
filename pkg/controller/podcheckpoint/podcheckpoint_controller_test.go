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

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	dynamicfake "k8s.io/client-go/dynamic/fake"
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
	checkpoint := func(withFinalizer bool) *unstructured.Unstructured {
		meta := map[string]interface{}{"name": "cp", "namespace": "ns"}
		if withFinalizer {
			meta["finalizers"] = []interface{}{RestoreLockFinalizer}
		}
		return &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": "checkpoint.k8s.io/v1alpha1",
			"kind":       "PodCheckpoint",
			"metadata":   meta,
		}}
	}

	tests := []struct {
		name          string
		withFinalizer bool
		pods          []*v1.Pod
		wantFinalizer bool
	}{
		{name: "active restore adds the finalizer", withFinalizer: false, pods: []*v1.Pod{pod("p", v1.PodPending)}, wantFinalizer: true},
		{name: "completed restore removes the finalizer", withFinalizer: true, pods: []*v1.Pod{pod("p", v1.PodRunning)}, wantFinalizer: false},
		{name: "active restore keeps the finalizer", withFinalizer: true, pods: []*v1.Pod{pod("p", v1.PodPending)}, wantFinalizer: true},
		{name: "no referencing pods leaves it unset", withFinalizer: false, pods: nil, wantFinalizer: false},
		{name: "unrelated running pod does not lock", withFinalizer: false, pods: []*v1.Pod{pod("p", v1.PodRunning)}, wantFinalizer: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()

			podIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
			for _, p := range tc.pods {
				require.NoError(t, podIndexer.Add(p))
			}

			scheme := apiruntime.NewScheme()
			gvrToListKind := map[schema.GroupVersionResource]string{podCheckpointGVR: "PodCheckpointList"}
			dc := dynamicfake.NewSimpleDynamicClientWithCustomListKinds(scheme, gvrToListKind, checkpoint(tc.withFinalizer))

			c := &Controller{dynamicClient: dc, podLister: corelisters.NewPodLister(podIndexer)}

			require.NoError(t, c.syncHandler(ctx, "ns/cp"))

			obj, err := dc.Resource(podCheckpointGVR).Namespace("ns").Get(ctx, "cp", metav1.GetOptions{})
			require.NoError(t, err)
			finalizers, _, err := unstructured.NestedStringSlice(obj.Object, "metadata", "finalizers")
			require.NoError(t, err)
			require.Equal(t, tc.wantFinalizer, slices.Contains(finalizers, RestoreLockFinalizer))
		})
	}
}
