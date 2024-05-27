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

package consistencydetector

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func TestDataConsistencyChecker(t *testing.T) {
	scenarios := []struct {
		name string

		listResponse   *v1.PodList
		retrievedItems []*v1.Pod
		requestOptions metav1.ListOptions

		expectedRequestOptions []metav1.ListOptions
		expectedListRequests   int
		expectPanic            bool
	}{
		{
			name: "data consistency check won't panic when data is consistent",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
			},
			requestOptions:       metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))},
			retrievedItems:       []*v1.Pod{makePod("p1", "1"), makePod("p2", "2")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					TimeoutSeconds:       ptr.To(int64(39)),
				},
			},
		},

		{
			name: "data consistency check won't panic when there is no data",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
			},
			requestOptions:       metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					TimeoutSeconds:       ptr.To(int64(39)),
				},
			},
		},

		{
			name: "data consistency panics when data is inconsistent",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3")},
			},
			requestOptions:       metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))},
			retrievedItems:       []*v1.Pod{makePod("p1", "1"), makePod("p2", "2")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
					TimeoutSeconds:       ptr.To(int64(39)),
				},
			},
			expectPanic: true,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			ctx := context.TODO()
			fakeLister := &listWrapper{response: scenario.listResponse}
			retrievedItemsFunc := func() []*v1.Pod {
				return scenario.retrievedItems
			}

			if scenario.expectPanic {
				require.Panics(t, func() {
					checkDataConsistency(ctx, "", scenario.listResponse.ResourceVersion, fakeLister.List, scenario.requestOptions, retrievedItemsFunc)
				})
			} else {
				checkDataConsistency(ctx, "", scenario.listResponse.ResourceVersion, fakeLister.List, scenario.requestOptions, retrievedItemsFunc)
			}

			require.Equal(t, fakeLister.counter, scenario.expectedListRequests)
			require.Equal(t, fakeLister.requestOptions, scenario.expectedRequestOptions)
		})
	}
}

func TestDriveWatchLisConsistencyIfRequired(t *testing.T) {
	ctx := context.TODO()
	checkWatchListDataConsistencyIfRequested[runtime.Object, runtime.Object](ctx, "", "", nil, nil)
}

func TestDataConsistencyCheckerRetry(t *testing.T) {
	ctx := context.TODO()
	retrievedItemsFunc := func() []*v1.Pod {
		return nil
	}
	stopListErrorAfter := 5
	fakeErrLister := &errorLister{stopErrorAfter: stopListErrorAfter}

	checkDataConsistency(ctx, "", "", fakeErrLister.List, metav1.ListOptions{}, retrievedItemsFunc)
	require.Equal(t, fakeErrLister.listCounter, fakeErrLister.stopErrorAfter)
}

type errorLister struct {
	listCounter    int
	stopErrorAfter int
}

func (lw *errorLister) List(_ context.Context, _ metav1.ListOptions) (runtime.Object, error) {
	lw.listCounter++
	if lw.listCounter == lw.stopErrorAfter {
		return &v1.PodList{}, nil
	}
	return nil, fmt.Errorf("nasty error")
}

type listWrapper struct {
	counter        int
	requestOptions []metav1.ListOptions
	response       *v1.PodList
}

func (lw *listWrapper) List(_ context.Context, opts metav1.ListOptions) (*v1.PodList, error) {
	lw.counter++
	lw.requestOptions = append(lw.requestOptions, opts)
	return lw.response, nil
}

func makePod(name, rv string) *v1.Pod {
	return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, ResourceVersion: rv, UID: types.UID(name)}}
}
