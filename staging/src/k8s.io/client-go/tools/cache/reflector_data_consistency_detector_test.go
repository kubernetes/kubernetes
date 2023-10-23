/*
Copyright 2023 The Kubernetes Authors.

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
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

func TestWatchListConsistency(t *testing.T) {
	scenarios := []struct {
		name string

		podList      *v1.PodList
		storeContent []*v1.Pod

		expectedRequestOptions []metav1.ListOptions
		expectedListRequests   int
		expectPanic            bool
	}{
		{
			name: "watchlist consistency check won't panic when data is consistent",
			podList: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
			},
			storeContent:         []*v1.Pod{makePod("p1", "1"), makePod("p2", "2")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
				},
			},
		},

		{
			name: "watchlist consistency check won't panic when there is no data",
			podList: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
			},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
				},
			},
		},

		{
			name: "watchlist consistency panics when data is inconsistent",
			podList: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3")},
			},
			storeContent:         []*v1.Pod{makePod("p1", "1"), makePod("p2", "2")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
				},
			},
			expectPanic: true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			listWatcher, store, _, stopCh := testData()
			for _, obj := range scenario.storeContent {
				require.NoError(t, store.Add(obj))
			}
			listWatcher.customListResponse = scenario.podList

			if scenario.expectPanic {
				require.Panics(t, func() { checkWatchListConsistency(stopCh, "", scenario.podList.ResourceVersion, listWatcher, store) })
			} else {
				checkWatchListConsistency(stopCh, "", scenario.podList.ResourceVersion, listWatcher, store)
			}

			verifyListCounter(t, listWatcher, scenario.expectedListRequests)
			verifyRequestOptions(t, listWatcher, scenario.expectedRequestOptions)
		})
	}
}

func TestDriveWatchLisConsistencyIfRequired(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	checkWatchListConsistencyIfRequested(stopCh, "", "", nil, nil)
}

func TestWatchListConsistencyRetry(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	stopCh := make(chan struct{})
	defer close(stopCh)

	stopListErrorAfter := 5
	errLister := &errorLister{stopErrorAfter: stopListErrorAfter}

	checkWatchListConsistency(stopCh, "", "", errLister, store)
	require.Equal(t, errLister.listCounter, errLister.stopErrorAfter)
}

type errorLister struct {
	listCounter    int
	stopErrorAfter int
}

func (lw *errorLister) List(_ metav1.ListOptions) (runtime.Object, error) {
	lw.listCounter++
	if lw.listCounter == lw.stopErrorAfter {
		return &v1.PodList{}, nil
	}
	return nil, fmt.Errorf("nasty error")
}

func (lw *errorLister) Watch(_ metav1.ListOptions) (watch.Interface, error) {
	panic("not implemented")
}
