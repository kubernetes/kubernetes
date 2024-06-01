/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

var (
	emptyListFunc = func(_ context.Context, opts metav1.ListOptions) (*v1.PodList, error) {
		return nil, nil
	}
	emptyListOptions = metav1.ListOptions{}
)

func TestDriveCheckListFromCacheDataConsistencyIfRequested(t *testing.T) {
	ctx := context.TODO()

	CheckListFromCacheDataConsistencyIfRequested(ctx, "", emptyListFunc, emptyListOptions, nil)
}

func TestCheckListFromCacheDataConsistencyIfRequestedInternalPanics(t *testing.T) {
	ctx := context.TODO()
	pod := makePod("p1", "1")

	wrappedTarget := func() {
		checkListFromCacheDataConsistencyIfRequestedInternal(ctx, "", emptyListFunc, emptyListOptions, pod)
	}

	require.PanicsWithError(t, "object does not implement the List interfaces", wrappedTarget)
}

func TestCheckListFromCacheDataConsistencyIfRequestedInternalHappyPath(t *testing.T) {
	ctx := context.TODO()
	listOptions := metav1.ListOptions{TimeoutSeconds: ptr.To(int64(39))}
	expectedRequestOptions := metav1.ListOptions{
		ResourceVersion:      "2",
		ResourceVersionMatch: metav1.ResourceVersionMatchExact,
		TimeoutSeconds:       ptr.To(int64(39)),
	}
	listResponse := &v1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "2"},
		Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
	}
	retrievedList := &v1.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "2"},
		Items:    []v1.Pod{*makePod("p1", "1"), *makePod("p2", "2")},
	}
	fakeLister := &listWrapper{response: listResponse}

	checkListFromCacheDataConsistencyIfRequestedInternal(ctx, "", fakeLister.List, listOptions, retrievedList)

	require.Equal(t, 1, fakeLister.counter)
	require.Equal(t, 1, len(fakeLister.requestOptions))
	require.Equal(t, fakeLister.requestOptions[0], expectedRequestOptions)
}
