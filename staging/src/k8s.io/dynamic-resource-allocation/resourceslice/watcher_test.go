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

package resourceslice

import (
	"testing"
	"testing/synctest"

	"github.com/stretchr/testify/require"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
)

func TestFilterSliceWatchByPoolName_FiltersByPoolName(t *testing.T) {
	synctest.Test(t, testFilterSliceWatchByPoolNameFiltersByPoolName)
}

func testFilterSliceWatchByPoolNameFiltersByPoolName(t *testing.T) {
	poolName := "pool-a"

	source := watch.NewFakeWithOptions(watch.FakeOptions{ChannelSize: 3})
	defer source.Stop()

	wrapped := filterSliceWatchByPoolName(t.Context(), source, poolName)
	defer wrapped.Stop()

	source.Add(resourceSliceForPool("pool-a", "slice-a"))
	source.Add(resourceSliceForPool("pool-b", "slice-b"))
	source.Add(resourceSliceForPool("pool-a", "slice-a2"))

	event1 := <-wrapped.ResultChan()
	require.Equal(t, watch.Added, event1.Type)
	require.Equal(t, "slice-a", event1.Object.(*resourceapi.ResourceSlice).Name)
	require.Equal(t, poolName, event1.Object.(*resourceapi.ResourceSlice).Spec.Pool.Name)

	event2 := <-wrapped.ResultChan()
	require.Equal(t, watch.Added, event2.Type)
	require.Equal(t, "slice-a2", event2.Object.(*resourceapi.ResourceSlice).Name)
}

func TestWrapWatcher_NilMatchPassesAll(t *testing.T) {
	synctest.Test(t, testWrapWatcherNilMatchPassesAll)
}

func testWrapWatcherNilMatchPassesAll(t *testing.T) {
	source := watch.NewFakeWithOptions(watch.FakeOptions{ChannelSize: 2})
	defer source.Stop()

	wrapped := newWrapWatcher(t.Context(), source, nil)
	defer wrapped.Stop()

	source.Add(resourceSliceForPool("pool-a", "slice-a"))
	source.Add(resourceSliceForPool("pool-b", "slice-b"))

	event1 := <-wrapped.ResultChan()
	require.Equal(t, "slice-a", event1.Object.(*resourceapi.ResourceSlice).Name)

	event2 := <-wrapped.ResultChan()
	require.Equal(t, "slice-b", event2.Object.(*resourceapi.ResourceSlice).Name)
}

func TestWrapWatcher_StopClosesResultChan(t *testing.T) {
	synctest.Test(t, testWrapWatcherStopClosesResultChan)
}

func testWrapWatcherStopClosesResultChan(t *testing.T) {
	source := watch.NewFakeWithOptions(watch.FakeOptions{ChannelSize: 1})
	wrapped := newWrapWatcher(t.Context(), source, nil)
	defer wrapped.Stop()

	source.Add(resourceSliceForPool("pool-a", "slice-a"))

	_, ok := <-wrapped.ResultChan()
	require.True(t, ok)

	wrapped.Stop()

	_, ok = <-wrapped.ResultChan()
	require.False(t, ok)
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
