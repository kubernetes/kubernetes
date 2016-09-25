/*
Copyright 2015 The Kubernetes Authors.

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

package container

import (
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/types"
)

func newTestCache() *cache {
	c := NewCache()
	return c.(*cache)
}

func TestCacheNotInitialized(t *testing.T) {
	cache := newTestCache()
	// If the global timestamp is not set, always return nil.
	d := cache.getIfNewerThan(types.UID("1234"), time.Time{})
	assert.True(t, d == nil, "should return nil since cache is not initialized")
}

func getTestPodIDAndStatus(numContainers int) (types.UID, *PodStatus) {
	id := types.UID(strconv.FormatInt(time.Now().UnixNano(), 10))
	name := fmt.Sprintf("cache-foo-%s", string(id))
	namespace := "ns"
	var status *PodStatus
	if numContainers > 0 {
		status = &PodStatus{ID: id, Name: name, Namespace: namespace}
	} else {
		status = &PodStatus{ID: id}
	}
	for i := 0; i < numContainers; i++ {
		status.ContainerStatuses = append(status.ContainerStatuses, &ContainerStatus{Name: string(i)})
	}
	return id, status
}

func TestGetIfNewerThanWhenPodExists(t *testing.T) {
	cache := newTestCache()
	timestamp := time.Now()

	cases := []struct {
		cacheTime time.Time
		modified  time.Time
		expected  bool
	}{
		{
			// Both the global cache timestamp and the modified time are newer
			// than the timestamp.
			cacheTime: timestamp.Add(time.Second),
			modified:  timestamp,
			expected:  true,
		},
		{
			// Global cache timestamp is newer, but the pod entry modified
			// time is older than the given timestamp. This means that the
			// entry is up-to-date even though it hasn't changed for a while.
			cacheTime: timestamp.Add(time.Second),
			modified:  timestamp.Add(-time.Second * 10),
			expected:  true,
		},
		{
			// Global cache timestamp is older, but the pod entry modified
			// time is newer than the given timestamp. This means that the
			// entry is up-to-date but the rest of the cache are still being
			// updated.
			cacheTime: timestamp.Add(-time.Second),
			modified:  timestamp.Add(time.Second * 3),
			expected:  true,
		},
		{
			// Both the global cache timestamp and the modified time are older
			// than the given timestamp.
			cacheTime: timestamp.Add(-time.Second),
			modified:  timestamp.Add(-time.Second),
			expected:  false,
		},
	}
	for i, c := range cases {
		podID, status := getTestPodIDAndStatus(2)
		cache.UpdateTime(c.cacheTime)
		cache.Set(podID, status, nil, c.modified)
		d := cache.getIfNewerThan(podID, timestamp)
		assert.Equal(t, c.expected, d != nil, "test[%d]", i)
	}
}

func TestGetPodNewerThanWhenPodDoesNotExist(t *testing.T) {
	cache := newTestCache()
	cacheTime := time.Now()
	cache.UpdateTime(cacheTime)
	podID := types.UID("1234")

	cases := []struct {
		timestamp time.Time
		expected  bool
	}{
		{
			timestamp: cacheTime.Add(-time.Second),
			expected:  true,
		},
		{
			timestamp: cacheTime.Add(time.Second),
			expected:  false,
		},
	}
	for i, c := range cases {
		d := cache.getIfNewerThan(podID, c.timestamp)
		assert.Equal(t, c.expected, d != nil, "test[%d]", i)
	}
}

func TestCacheSetAndGet(t *testing.T) {
	cache := NewCache()
	cases := []struct {
		numContainers int
		error         error
	}{
		{numContainers: 3, error: nil},
		{numContainers: 2, error: fmt.Errorf("unable to get status")},
		{numContainers: 0, error: nil},
	}
	for i, c := range cases {
		podID, status := getTestPodIDAndStatus(c.numContainers)
		cache.Set(podID, status, c.error, time.Time{})
		// Read back the status and error stored in cache and make sure they
		// match the original ones.
		actualStatus, actualErr := cache.Get(podID)
		assert.Equal(t, status, actualStatus, "test[%d]", i)
		assert.Equal(t, c.error, actualErr, "test[%d]", i)
	}
}

func TestCacheGetPodDoesNotExist(t *testing.T) {
	cache := NewCache()
	podID, status := getTestPodIDAndStatus(0)
	// If the pod does not exist in cache, cache should return an status
	// object with id filled.
	actualStatus, actualErr := cache.Get(podID)
	assert.Equal(t, status, actualStatus)
	assert.Equal(t, nil, actualErr)
}

func TestDelete(t *testing.T) {
	cache := &cache{pods: map[types.UID]*data{}}
	// Write a new pod status into the cache.
	podID, status := getTestPodIDAndStatus(3)
	cache.Set(podID, status, nil, time.Time{})
	actualStatus, actualErr := cache.Get(podID)
	assert.Equal(t, status, actualStatus)
	assert.Equal(t, nil, actualErr)
	// Delete the pod from cache, and verify that we get an empty status.
	cache.Delete(podID)
	expectedStatus := &PodStatus{ID: podID}
	actualStatus, actualErr = cache.Get(podID)
	assert.Equal(t, expectedStatus, actualStatus)
	assert.Equal(t, nil, actualErr)
}

func verifyNotification(t *testing.T, ch chan *data, expectNotification bool) {
	if expectNotification {
		assert.True(t, len(ch) > 0, "Did not receive notification")
	} else {
		assert.True(t, len(ch) < 1, "Should not have triggered the notification")
	}
	// Drain the channel.
	for i := 0; i < len(ch); i++ {
		<-ch
	}
}

func TestRegisterNotification(t *testing.T) {
	cache := newTestCache()
	cacheTime := time.Now()
	cache.UpdateTime(cacheTime)

	podID, status := getTestPodIDAndStatus(1)
	ch := cache.subscribe(podID, cacheTime.Add(time.Second))
	verifyNotification(t, ch, false)
	cache.Set(podID, status, nil, cacheTime.Add(time.Second))
	// The Set operation should've triggered the notification.
	verifyNotification(t, ch, true)

	podID, _ = getTestPodIDAndStatus(1)

	ch = cache.subscribe(podID, cacheTime.Add(time.Second))
	verifyNotification(t, ch, false)
	cache.UpdateTime(cacheTime.Add(time.Second * 2))
	// The advance of cache timestamp should've triggered the notification.
	verifyNotification(t, ch, true)
}
