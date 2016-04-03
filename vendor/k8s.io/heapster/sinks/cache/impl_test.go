// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cache

import (
	"fmt"
	"sync"
	"testing"
	"time"

	cadvisor "github.com/google/cadvisor/info/v1"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	source_api "k8s.io/heapster/sources/api"
)

func TestFuzz(t *testing.T) {
	cache := NewCache(time.Hour, time.Second)
	var (
		pods       []source_api.Pod
		containers []source_api.Container
	)
	f := fuzz.New().NumElements(2, 10).NilChance(0)
	f.Fuzz(&pods)
	f.Fuzz(&containers)
	assert := assert.New(t)
	assert.NoError(cache.StorePods(pods))
	assert.NoError(cache.StoreContainers(containers))
	time.Sleep(5 * time.Second)
	zeroTime := time.Time{}
	assert.NotEmpty(cache.GetPods(zeroTime, zeroTime))
}

func TestGC(t *testing.T) {
	var mutex sync.Mutex
	var podEvictedCount int
	var containerEvictedCount int

	cache := NewCache(time.Millisecond, time.Second)
	cache.AddCacheListener(CacheListener{
		PodEvicted: func(namespace string, name string) {
			mutex.Lock()
			defer mutex.Unlock()
			podEvictedCount++
		},
		FreeContainerEvicted: func(hostname string, name string) {
			mutex.Lock()
			defer mutex.Unlock()
			containerEvictedCount++
		},
	})

	var (
		pods       []source_api.Pod
		containers []source_api.Container
	)
	f := fuzz.New().NumElements(2, 10).NilChance(0)
	f.Fuzz(&pods)
	f.Fuzz(&containers)
	for i := range pods {
		pods[i].ID = fmt.Sprintf("ID-%d", i)
		pods[i].Name = fmt.Sprintf("%d-%s", i, pods[i].Name)
	}
	for i := range containers {
		containers[i].Hostname = fmt.Sprintf("Node-%d", i%5)
		containers[i].Name = fmt.Sprintf("%d-%s", i, containers[i].Name)
	}

	assert := assert.New(t)
	assert.NoError(cache.StorePods(pods))
	assert.NoError(cache.StoreContainers(containers))
	zeroTime := time.Time{}
	assert.NotEmpty(cache.GetFreeContainers(zeroTime, zeroTime))
	assert.NotEmpty(cache.GetPods(zeroTime, zeroTime))
	// Expect all data to be deleted after 2 seconds.
	time.Sleep(10 * time.Second)
	assert.Empty(cache.GetFreeContainers(zeroTime, zeroTime))
	assert.Empty(cache.GetPods(zeroTime, zeroTime))

	mutex.Lock()
	defer mutex.Unlock()
	assert.Equal(len(pods), podEvictedCount)
	assert.Equal(len(containers), containerEvictedCount)
}

func getContainer(name string) source_api.Container {
	f := fuzz.New().NumElements(2, 2).NilChance(0)
	now := time.Now()
	containerSpec := source_api.ContainerSpec{
		ContainerSpec: cadvisor.ContainerSpec{
			CreationTime:  now,
			HasCpu:        true,
			HasMemory:     true,
			HasNetwork:    true,
			HasFilesystem: true,
			HasDiskIo:     true,
		},
	}
	containerStats := make([]*source_api.ContainerStats, 1)
	f.Fuzz(&containerStats)
	for idx := range containerStats {
		containerStats[idx].Timestamp = now
	}
	return source_api.Container{
		Name:  name,
		Spec:  containerSpec,
		Stats: containerStats,
		Image: "gcr.io/" + name,
	}
}

func TestRealCacheData(t *testing.T) {
	containers := []source_api.Container{
		getContainer("container1"),
	}
	pods := []source_api.Pod{
		{
			PodMetadata: source_api.PodMetadata{
				Name:         "pod1",
				ID:           "123",
				Namespace:    "test",
				NamespaceUID: "test-uid",
				Hostname:     "1.2.3.4",
				Status:       "Running",
			},
			Containers: containers,
		},
		{
			PodMetadata: source_api.PodMetadata{
				Name:         "pod2",
				ID:           "1234",
				Namespace:    "test",
				NamespaceUID: "test-uid",
				Hostname:     "1.2.3.5",
				Status:       "Running",
			},
			Containers: containers,
		},
	}
	cache := NewCache(time.Hour, time.Hour)
	assert := assert.New(t)
	assert.NoError(cache.StorePods(pods))
	assert.NoError(cache.StoreContainers(containers))
	actualPods := cache.GetPods(time.Time{}, time.Time{})
	actualContainer := cache.GetNodes(time.Time{}, time.Time{})
	actualContainer = append(actualContainer, cache.GetFreeContainers(time.Time{}, time.Time{})...)
	actualPodsMap := map[string]*PodElement{}
	for _, pod := range actualPods {
		actualPodsMap[pod.Name] = pod
	}
	for _, expectedPod := range pods {
		pod, exists := actualPodsMap[expectedPod.Name]
		require.True(t, exists)
		if pod == nil {
			continue
		}
		require.NotEmpty(t, pod.Containers)
		assert.NotEmpty(pod.Containers[0].Metrics)
		assert.NotEqual(pod.LastUpdate, time.Time{})
	}
	actualContainerMap := map[string]*ContainerElement{}
	for _, cont := range actualContainer {
		actualContainerMap[cont.Name] = cont
	}
	for _, expectedContainer := range containers {
		ce, exists := actualContainerMap[expectedContainer.Name]
		assert.True(exists, "container %q does not exist", expectedContainer.Name)
		if ce == nil {
			continue
		}
		assert.Equal(expectedContainer.Image, ce.Image)
		assert.NotNil(ce.Metrics)
		assert.NotEmpty(ce.Metrics)
		assert.NotEqual(ce.LastUpdate, time.Time{})
	}
}

func TestEvents(t *testing.T) {
	cache := NewCache(time.Hour, time.Hour)
	assert := assert.New(t)
	eventA := &Event{
		Message: "test message 1",
		Source:  "test",
		Metadata: Metadata{
			UID:        "123",
			LastUpdate: time.Now(),
		},
	}
	assert.NoError(cache.StoreEvents([]*Event{eventA}))
	// This duplicate event must be ignored.
	assert.NoError(cache.StoreEvents([]*Event{eventA}))
	eventB := &Event{
		Message: "test message 2",
		Source:  "test",
		Metadata: Metadata{
			UID:        "124",
			LastUpdate: time.Now(),
		},
	}
	assert.NoError(cache.StoreEvents([]*Event{eventB}))
	zeroTime := time.Time{}
	events := cache.GetEvents(zeroTime, zeroTime)
	assert.Len(events, 2)
	assert.Equal(events[0], eventB)
	assert.Equal(events[1], eventA)
}

func TestEventsByTime(t *testing.T) {
	cache := NewCache(time.Hour, time.Hour)
	assert := assert.New(t)
	timeA, _ := time.Parse("2006-01-02 15:04:05", "2016-01-15 10:00:00")
	eventA := &Event{
		Message: "test message 1",
		Source:  "test",
		Metadata: Metadata{
			UID:        "123",
			LastUpdate: timeA,
		},
	}

	assert.NoError(cache.StoreEvents([]*Event{eventA}))

	timeB, _ := time.Parse("2006-01-02 15:04:05", "2016-01-15 10:00:01")
	eventB := &Event{
		Message: "test message 2",
		Source:  "test",
		Metadata: Metadata{
			UID:        "124",
			LastUpdate: timeB,
		},
	}

	zeroTime := time.Time{}
	assert.NoError(cache.StoreEvents([]*Event{eventB}))
	events := cache.GetEvents(zeroTime, zeroTime)
	assert.Len(events, 2)

	events = cache.GetEvents(timeA, zeroTime)
	assert.Len(events, 2)

	events = cache.GetEvents(timeB, zeroTime)
	assert.Len(events, 1)

	events = cache.GetEvents(timeB.Add(time.Nanosecond), zeroTime)
	assert.Len(events, 0)
}
