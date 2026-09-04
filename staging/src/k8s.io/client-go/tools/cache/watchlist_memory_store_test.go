/*
Copyright 2026 The Kubernetes Authors.

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

package cache

import (
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestWatchListMemoryOptimizedStore(t *testing.T) {
	makePodFunc := func(name, rv string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       "ns",
				ResourceVersion: rv,
			},
		}
	}

	scenarios := []struct {
		name           string
		op             string
		existingObject *v1.Pod
		incomingObject *v1.Pod
		expectReuse    bool
	}{
		{
			name:           "add reuses cached (existing) object when rv matches",
			op:             "add",
			existingObject: makePodFunc("p1", "10"),
			incomingObject: makePodFunc("p1", "10"),
			expectReuse:    true,
		},
		{
			name:           "update reuses cached (existing) object when rv matches",
			op:             "update",
			existingObject: makePodFunc("p1", "10"),
			incomingObject: makePodFunc("p1", "10"),
			expectReuse:    true,
		},
		{
			name:           "does not reuse when resourceVersion differs",
			op:             "add",
			existingObject: makePodFunc("p1", "10"),
			incomingObject: makePodFunc("p1", "11"),
			expectReuse:    false,
		},
		{
			name:           "does not reuse when cache is empty",
			op:             "add",
			incomingObject: makePodFunc("p1", "10"),
			expectReuse:    false,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			keyFunc := DeletionHandlingMetaNamespaceKeyFunc
			temporaryStore := NewStore(keyFunc)
			clientStore := NewStore(keyFunc)

			if scenario.existingObject != nil {
				require.NoError(t, clientStore.Add(scenario.existingObject))
			}

			store := newWatchListMemoryOptimizedStore(temporaryStore, clientStore, keyFunc)
			switch scenario.op {
			case "add":
				require.NoError(t, store.Add(scenario.incomingObject))
			case "update":
				require.NoError(t, store.Update(scenario.incomingObject))
			default:
				require.Failf(t, "unknown op", "op: %s", scenario.op)
			}

			key, err := keyFunc(scenario.incomingObject)
			require.NoError(t, err)

			got, exists, err := temporaryStore.GetByKey(key)
			require.NoError(t, err)
			require.True(t, exists)

			if scenario.expectReuse {
				require.Same(t, scenario.existingObject, got)
			} else {
				require.Same(t, scenario.incomingObject, got)
			}
		})
	}
}

func TestWatchListMemoryOptimizedStoreNilClientStore(t *testing.T) {
	keyFunc := DeletionHandlingMetaNamespaceKeyFunc
	temporaryStore := NewStore(keyFunc)

	store := newWatchListMemoryOptimizedStore(temporaryStore, nil, keyFunc)
	require.Same(t, temporaryStore, store)
}
