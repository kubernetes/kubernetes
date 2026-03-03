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

package resourceslice

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
)

func TestOnlyPoolWatcher_FiltersByPoolName(t *testing.T) {
	ctx := context.Background()
	poolName := "pool-a"

	source := watch.NewFake()
	defer source.Stop()

	// Verifies onlyPoolWatcher used in initInformer when ReconcileOnlyPoolName
	wrapped := onlyPoolWatcher(ctx, source, poolName)
	wrapped.(*wrapWatcher).result = make(chan watch.Event, 3)
	defer wrapped.Stop()

	// Feed events for different pools; only pool-a should appear in the wrapped watch.
	source.Add(resourceSliceForPool("pool-a", "slice-a"))
	source.Add(resourceSliceForPool("pool-b", "slice-b"))
	source.Add(resourceSliceForPool("pool-a", "slice-a2"))

	var received []watch.Event
	for i := 0; i < 2; i++ {
		event, ok := <-wrapped.ResultChan()
		require.True(t, ok, "expected event %d", i+1)
		received = append(received, event)
	}

	require.Len(t, received, 2, "onlyPoolWatcher should pass only events for the given pool name")
	assert.Equal(t, watch.Added, received[0].Type)
	assert.Equal(t, "slice-a", received[0].Object.(*resourceapi.ResourceSlice).Name)
	assert.Equal(t, poolName, received[0].Object.(*resourceapi.ResourceSlice).Spec.Pool.Name)
	assert.Equal(t, watch.Added, received[1].Type)
	assert.Equal(t, "slice-a2", received[1].Object.(*resourceapi.ResourceSlice).Name)
}

func TestWrapWatcher_NilMatchPassesAll(t *testing.T) {
	ctx := context.Background()

	source := watch.NewFake()
	defer source.Stop()

	// When the match is nil, all events pass through
	wrapped := newWrapWatcher(ctx, source, nil)
	wrapped.result = make(chan watch.Event, 2)
	defer wrapped.Stop()

	source.Add(resourceSliceForPool("pool-a", "slice-a"))
	source.Add(resourceSliceForPool("pool-b", "slice-b"))

	resultChan := wrapped.ResultChan()
	received := []watch.Event{<-resultChan, <-resultChan}

	require.Len(t, received, 2)
	assert.Equal(t, "slice-a", received[0].Object.(*resourceapi.ResourceSlice).Name)
	assert.Equal(t, "slice-b", received[1].Object.(*resourceapi.ResourceSlice).Name)
}

func TestWrapWatcher_StopClosesResultChan(t *testing.T) {
	ctx := context.Background()
	source := watch.NewFake()
	wrapped := newWrapWatcher(ctx, source, func(watch.Event) bool { return true })
	source.Add(resourceSliceForPool("pool-a", "slice-a"))
	<-wrapped.ResultChan() // consume one event
	wrapped.Stop()
	_, ok := <-wrapped.ResultChan()
	assert.False(t, ok, "ResultChan should be closed after Stop()")
}

func resourceSliceForPool(poolName, name string) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: "test-driver",
			Pool:   resourceapi.ResourcePool{Name: poolName},
		},
	}
}
