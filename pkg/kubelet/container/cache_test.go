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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/clock"
	clocktesting "k8s.io/utils/clock/testing"
)

func newTestCache() *cache {
	c := NewCache()
	return c.(*cache)
}

func newTestCacheWithClock(testClock clock.Clock) *cache {
	config := DefaultCacheConfig()
	config.Clock = testClock
	c := NewCacheWithConfig(config)
	return c.(*cache)
}

func TestCacheNotInitialized(t *testing.T) {
	cache := newTestCache()
	// If the global timestamp is not set, always return nil.
	d := cache.getIfNewerThan(types.UID("1234"), time.Time{})
	assert.Nil(t, d, "should return nil since cache is not initialized")
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
		status.ContainerStatuses = append(status.ContainerStatuses, &Status{Name: strconv.Itoa(i)})
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
	assert.NoError(t, actualErr)
}

func TestDelete(t *testing.T) {
	cache := &cache{pods: map[types.UID]*data{}}
	// Write a new pod status into the cache.
	podID, status := getTestPodIDAndStatus(3)
	cache.Set(podID, status, nil, time.Time{})
	actualStatus, actualErr := cache.Get(podID)
	assert.Equal(t, status, actualStatus)
	assert.NoError(t, actualErr)
	// Delete the pod from cache, and verify that we get an empty status.
	cache.Delete(podID)
	expectedStatus := &PodStatus{ID: podID}
	actualStatus, actualErr = cache.Get(podID)
	assert.Equal(t, expectedStatus, actualStatus)
	assert.NoError(t, actualErr)
}

func verifyNotification(t *testing.T, ch chan *data, expectNotification bool) {
	if expectNotification {
		assert.NotEmpty(t, ch, "Did not receive notification")
	} else {
		assert.Empty(t, ch, "Should not have triggered the notification")
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

func TestTimeShiftHandling(t *testing.T) {
	// Create a test clock to simulate time shifts
	testClock := clocktesting.NewFakeClock(time.Now())
	cache := newTestCacheWithClock(testClock)
	podID, status := getTestPodIDAndStatus(1)
	
	// Set up initial cache state
	baseTime := testClock.Now()
	cache.Set(podID, status, nil, baseTime)
	cache.UpdateTime(baseTime)
	
	// Simulate time shift: clock goes backwards by 40 seconds
	// This simulates the scenario described in issue #134153
	testClock.Step(-40 * time.Second)
	timeShiftedTime := baseTime.Add(-40 * time.Second)
	
	// Test getIfNewerThan with time shift
	// Should return the cached data even though the time went backwards
	d := cache.getIfNewerThan(podID, timeShiftedTime)
	assert.NotNil(t, d, "should return cached data despite time shift")
	assert.Equal(t, status, d.status, "should return the correct cached status")
	
	// Test notification with time shift
	ch := cache.subscribe(podID, timeShiftedTime)
	// Should receive notification immediately due to time shift detection
	verifyNotification(t, ch, true)
}

func TestTimeShiftThreshold(t *testing.T) {
	// Create a test clock to simulate time shifts
	testClock := clocktesting.NewFakeClock(time.Now())
	cache := newTestCacheWithClock(testClock)
	podID, status := getTestPodIDAndStatus(1)
	
	// Set up initial cache state
	baseTime := testClock.Now()
	cache.Set(podID, status, nil, baseTime)
	cache.UpdateTime(baseTime)
	
	// Test with normal time progression (should not trigger time shift detection)
	// Use a time that's 5 seconds later than baseTime, which is normal progression
	normalTime := baseTime.Add(5 * time.Second)
	d := cache.getIfNewerThan(podID, normalTime)
	assert.Nil(t, d, "should return nil for normal time progression")
	
	// Test with small time shift (should not trigger time shift detection)
	// The key insight: we need to simulate the scenario where the current time
	// (testClock.Now()) is much earlier than the minTime parameter
	// This happens when the clock goes backwards after minTime was set
	testClock.Step(-10 * time.Second) // Clock goes back 10 seconds
	// But we're asking for data newer than a time that's 10 seconds in the future
	// relative to the current clock time
	futureTime := testClock.Now().Add(10 * time.Second)
	d = cache.getIfNewerThan(podID, futureTime)
	assert.Nil(t, d, "should return nil for small time difference")
	
	// Test with large time shift (should trigger time shift detection)
	// Clock goes back 40 seconds total, and we ask for data newer than
	// a time that's 40 seconds in the future relative to current clock
	testClock.Step(-30 * time.Second) // Total: -40 seconds from original
	futureTime = testClock.Now().Add(40 * time.Second)
	d = cache.getIfNewerThan(podID, futureTime)
	assert.NotNil(t, d, "should return cached data for large time difference (time shift)")
}

func TestTimeShiftEdgeCases(t *testing.T) {
	// Create a test clock to simulate time shifts
	testClock := clocktesting.NewFakeClock(time.Now())
	cache := newTestCacheWithClock(testClock)
	podID, status := getTestPodIDAndStatus(1)
	
	// Set up initial cache state
	baseTime := testClock.Now()
	cache.Set(podID, status, nil, baseTime)
	cache.UpdateTime(baseTime)
	
	// Test exactly at the threshold boundary (30 seconds)
	// Clock goes back 30 seconds, and we ask for data newer than
	// a time that's 30 seconds in the future relative to current clock
	testClock.Step(-30 * time.Second)
	exactThresholdTime := testClock.Now().Add(30 * time.Second)
	d := cache.getIfNewerThan(podID, exactThresholdTime)
	assert.Nil(t, d, "should return nil for exactly threshold time difference")
	
	// Test just over the threshold (30.1 seconds)
	// Clock goes back an additional 100ms, and we ask for data newer than
	// a time that's 30.1 seconds in the future relative to current clock
	testClock.Step(-100 * time.Millisecond) // Total: -30.1 seconds
	overThresholdTime := testClock.Now().Add(30*time.Second + 100*time.Millisecond)
	d = cache.getIfNewerThan(podID, overThresholdTime)
	assert.NotNil(t, d, "should return cached data for time difference just over threshold")
}

func TestConfigurableThreshold(t *testing.T) {
	// Create a test clock
	testClock := clocktesting.NewFakeClock(time.Now())
	
	// Create cache with custom threshold
	config := DefaultCacheConfig()
	config.Clock = testClock
	config.TimeShiftThreshold = 10 * time.Second // Custom threshold
	cache := NewCacheWithConfig(config).(*cache)
	
	podID, status := getTestPodIDAndStatus(1)
	baseTime := testClock.Now()
	cache.Set(podID, status, nil, baseTime)
	cache.UpdateTime(baseTime)
	
	// Test with time shift just under custom threshold (should not trigger)
	// Clock goes back 5 seconds, and we ask for data newer than
	// a time that's 5 seconds in the future relative to current clock
	testClock.Step(-5 * time.Second)
	underThresholdTime := testClock.Now().Add(5 * time.Second)
	d := cache.getIfNewerThan(podID, underThresholdTime)
	assert.Nil(t, d, "should return nil for time difference under custom threshold")
	
	// Test with time shift just over custom threshold (should trigger)
	// Clock goes back an additional 6 seconds (total: -11 seconds), and we ask for data newer than
	// a time that's 11 seconds in the future relative to current clock
	testClock.Step(-6 * time.Second) // Total: -11 seconds
	overThresholdTime := testClock.Now().Add(11 * time.Second)
	d = cache.getIfNewerThan(podID, overThresholdTime)
	assert.NotNil(t, d, "should return cached data for time difference over custom threshold")
}
