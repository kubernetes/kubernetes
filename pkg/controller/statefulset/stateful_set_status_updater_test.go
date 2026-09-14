/*
Copyright 2017 The Kubernetes Authors.

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

package statefulset

import (
	"context"
	"errors"
	"testing"

	apps "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	appslisters "k8s.io/client-go/listers/apps/v1"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller/util/consistency"
)

func TestStatefulSetStatusUpdater(t *testing.T) {
	tests := []struct {
		name       string
		status     apps.StatefulSetStatus
		reactorErr error
		wantErr    bool
		checkFn    func(*testing.T, apps.StatefulSetStatus)
	}{
		{
			name:   "updates status",
			status: apps.StatefulSetStatus{ObservedGeneration: 1, Replicas: 2},
			checkFn: func(t *testing.T, status apps.StatefulSetStatus) {
				if status.Replicas != 2 {
					t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", status.Replicas)
				}
			},
		},
		{
			name:   "updates observed generation",
			status: apps.StatefulSetStatus{ObservedGeneration: 3, Replicas: 2},
			checkFn: func(t *testing.T, status apps.StatefulSetStatus) {
				if status.ObservedGeneration != 3 {
					t.Errorf("expected observedGeneration to be synced with generation for statefulset %d", status.ObservedGeneration)
				}
			},
		},
		{
			name:   "updates available replicas",
			status: apps.StatefulSetStatus{ObservedGeneration: 1, Replicas: 2, AvailableReplicas: 3},
			checkFn: func(t *testing.T, status apps.StatefulSetStatus) {
				if status.AvailableReplicas != 3 {
					t.Errorf("UpdateStatefulSetStatus mutated the sets available replicas %d", status.AvailableReplicas)
				}
			},
		},
		{
			name:       "update replicas server failure",
			status:     apps.StatefulSetStatus{ObservedGeneration: 3, Replicas: 2},
			reactorErr: apierrors.NewInternalError(errors.New("API server down")),
			wantErr:    true,
		},
		{
			name:       "update replicas conflict persists",
			status:     apps.StatefulSetStatus{ObservedGeneration: 3, Replicas: 2},
			reactorErr: apierrors.NewConflict(schema.GroupResource{Group: "apps", Resource: "statefulset"}, "foo", errors.New("object already exists")),
			wantErr:    true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			set := newStatefulSet(3)
			fakeClient := &fake.Clientset{}
			indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			indexer.Add(set) // nolint:errcheck
			updater := NewRealStatefulSetStatusUpdater(fakeClient, appslisters.NewStatefulSetLister(indexer), consistency.NewNoopConsistencyStore())
			fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
				update := action.(core.UpdateAction)
				return true, update.GetObject(), tc.reactorErr
			})
			status := tc.status
			if err := updater.UpdateStatefulSetStatus(ctx, set, &status); (err != nil) != tc.wantErr {
				t.Errorf("UpdateStatefulSetStatus() error = %v, wantErr %v", err, tc.wantErr)
			}
			if tc.checkFn != nil {
				tc.checkFn(t, status)
			}
		})
	}
}

func TestStatefulSetStatusUpdaterUpdateReplicasConflict(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	set := newStatefulSet(3)
	status := apps.StatefulSetStatus{ObservedGeneration: 3, Replicas: 2}
	conflict := false
	fakeClient := &fake.Clientset{}
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	indexer.Add(set) // nolint:errcheck
	setLister := appslisters.NewStatefulSetLister(indexer)
	updater := NewRealStatefulSetStatusUpdater(fakeClient, setLister, consistency.NewNoopConsistencyStore())
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		if !conflict {
			conflict = true
			return true, update.GetObject(), apierrors.NewConflict(action.GetResource().GroupResource(), set.Name, errors.New("object already exists"))
		}
		return true, update.GetObject(), nil

	})
	if err := updater.UpdateStatefulSetStatus(ctx, set, &status); err != nil {
		t.Errorf("UpdateStatefulSetStatus returned an error: %s", err)
	}
	if set.Status.Replicas != 2 {
		t.Errorf("UpdateStatefulSetStatus mutated the sets replicas %d", set.Status.Replicas)
	}
}

func TestStatefulSetStatusUpdaterConsistencyStore(t *testing.T) {
	set := newStatefulSet(3)
	status := apps.StatefulSetStatus{ObservedGeneration: 1, Replicas: 2}
	fakeClient := &fake.Clientset{}

	rvGetter := &fakeRVGetter{rv: "1"}
	consistencyStore := consistency.NewConsistencyStore(map[schema.GroupResource]consistency.LastSyncRVGetter{
		{Group: "apps", Resource: "statefulsets"}: rvGetter,
	})

	updater := NewRealStatefulSetStatusUpdater(fakeClient, nil, consistencyStore)
	fakeClient.AddReactor("update", "statefulsets", func(action core.Action) (bool, runtime.Object, error) {
		update := action.(core.UpdateAction)
		update.GetObject().(*apps.StatefulSet).ResourceVersion = "2"
		return true, update.GetObject(), nil
	})
	if err := updater.UpdateStatefulSetStatus(context.TODO(), set, &status); err != nil {
		t.Fatalf("Error returned on successful status update: %s", err)
	}

	if err := consistencyStore.EnsureReady(types.NamespacedName{Namespace: set.Namespace, Name: set.Name}); err == nil {
		t.Error("expected consistency store to return an error, got nil")
	}

	rvGetter.rv = "2"

	if err := consistencyStore.EnsureReady(types.NamespacedName{Namespace: set.Namespace, Name: set.Name}); err != nil {
		t.Errorf("expected consistency store to be ready, got an error: %v", err)
	}
}
