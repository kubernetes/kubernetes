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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

func TestDataConsistencyChecker(t *testing.T) {
	scenarios := []struct {
		name string

		lastSyncedResourceVersion string
		listResponse              runtime.Object
		retrievedItems            []runtime.Object
		requestOptions            metav1.ListOptions

		expectedRequestOptions []metav1.ListOptions
		expectedListRequests   int
		expectPanic            bool
	}{
		{
			name:                      "data consistency check won't panic when data is consistent",
			lastSyncedResourceVersion: "2",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
			},
			requestOptions:       metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))},
			retrievedItems:       []runtime.Object{makePod("p1", "1"), makePod("p2", "2")},
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
			name:                      "data consistency check works with unstructured data (dynamic client)",
			lastSyncedResourceVersion: "2",
			listResponse: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "vTest",
					"kind":       "rTestList",
				},
				Items: []unstructured.Unstructured{
					*makeUnstructuredObject("vTest", "rTest", "item1"),
					*makeUnstructuredObject("vTest", "rTest", "item2"),
				},
			},
			requestOptions: metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))},
			retrievedItems: []runtime.Object{
				makeUnstructuredObject("vTest", "rTest", "item1"),
				makeUnstructuredObject("vTest", "rTest", "item2"),
			},
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
			name:                      "legacy, the limit is removed from the list options when it wasn't honored by the watch cache",
			lastSyncedResourceVersion: "2",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3")},
			},
			requestOptions:       metav1.ListOptions{ResourceVersion: "0", Limit: 2},
			retrievedItems:       []runtime.Object{makePod("p1", "1"), makePod("p2", "2"), makePod("p3", "3")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
				},
			},
		},

		{
			name:                      "the limit is NOT removed from the list options for non-legacy request",
			lastSyncedResourceVersion: "2",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3")},
			},
			requestOptions:       metav1.ListOptions{ResourceVersion: "2", Limit: 2},
			retrievedItems:       []runtime.Object{makePod("p1", "1"), makePod("p2", "2"), makePod("p3", "3")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					Limit:                2,
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
				},
			},
		},

		{
			name:                      "legacy, the limit is NOT removed from the list options when the watch cache is disabled",
			lastSyncedResourceVersion: "2",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3")},
			},
			requestOptions:       metav1.ListOptions{ResourceVersion: "0", Limit: 5},
			retrievedItems:       []runtime.Object{makePod("p1", "1"), makePod("p2", "2"), makePod("p3", "3")},
			expectedListRequests: 1,
			expectedRequestOptions: []metav1.ListOptions{
				{
					Limit:                5,
					ResourceVersion:      "2",
					ResourceVersionMatch: metav1.ResourceVersionMatchExact,
				},
			},
		},

		{
			name:                      "data consistency check won't panic when there is no data",
			lastSyncedResourceVersion: "2",
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
			name:                 "data consistency check won't be performed when Continuation was set",
			requestOptions:       metav1.ListOptions{Continue: "fake continuation token"},
			expectedListRequests: 0,
		},

		{
			name:                      "data consistency check won't be performed when ResourceVersion was set to 0",
			lastSyncedResourceVersion: "0",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "0"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
			},
			requestOptions:       metav1.ListOptions{},
			expectedListRequests: 0,
		},

		{
			name:                      "data consistency panics when data is inconsistent",
			lastSyncedResourceVersion: "2",
			listResponse: &v1.PodList{
				ListMeta: metav1.ListMeta{ResourceVersion: "2"},
				Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2"), *makePod("p3", "3")},
			},
			requestOptions:       metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))},
			retrievedItems:       []runtime.Object{makePod("p1", "1"), makePod("p2", "2")},
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
			if scenario.listResponse == nil {
				scenario.listResponse = &v1.PodList{}
			}
			fakeLister := &listWrapper{response: scenario.listResponse}
			retrievedItemsFunc := func() []runtime.Object {
				return scenario.retrievedItems
			}

			if scenario.expectPanic {
				require.Panics(t, func() {
					CheckDataConsistency(ctx, "", scenario.lastSyncedResourceVersion, fakeLister.List, scenario.requestOptions, retrievedItemsFunc)
				})
			} else {
				CheckDataConsistency(ctx, "", scenario.lastSyncedResourceVersion, fakeLister.List, scenario.requestOptions, retrievedItemsFunc)
			}

			require.Equal(t, scenario.expectedListRequests, fakeLister.counter)
			require.Equal(t, scenario.expectedRequestOptions, fakeLister.requestOptions)
		})
	}
}

func TestDataConsistencyCheckerRetry(t *testing.T) {
	ctx := context.TODO()
	retrievedItemsFunc := func() []*v1.Pod {
		return nil
	}
	stopListErrorAfter := 5
	fakeErrLister := &errorLister{stopErrorAfter: stopListErrorAfter}

	CheckDataConsistency(ctx, "", "", fakeErrLister.List, metav1.ListOptions{}, retrievedItemsFunc)
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
	response       runtime.Object
}

func (lw *listWrapper) List(_ context.Context, opts metav1.ListOptions) (runtime.Object, error) {
	lw.counter++
	lw.requestOptions = append(lw.requestOptions, opts)
	return lw.response, nil
}

func makePod(name, rv string) *v1.Pod {
	return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, ResourceVersion: rv, UID: types.UID(name)}}
}

func makeUnstructuredObject(version, kind, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": version,
			"kind":       kind,
			"metadata": map[string]interface{}{
				"name": name,
			},
		},
	}
}
