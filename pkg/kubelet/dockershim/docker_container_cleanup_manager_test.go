/*
Copyright 2019 The Kubernetes Authors.

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

package dockershim

import (
	"strconv"
	"testing"
	"time"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestContainerCleanupManager(t *testing.T) {
	t.Run("when there's no cleanup info stored", func(t *testing.T) {
		defer setCurrentNanoTimestamps(t, time.Second)()
		defer setCleanupManagerDurations(t, time.Millisecond)()

		cleanupService := &dummyCleanupService{}
		cleanupManager := newDockerContainerCleanupManager(cleanupService)

		runCleanupManagerForNTicks(t, cleanupManager, 1)
		require.Equal(t, 0, len(cleanupService.args))
	})

	t.Run("when there is a few cleanup infos stored, but none that is stale", func(t *testing.T) {
		defer setCurrentNanoTimestamps(t, time.Second, 2*time.Second, 3*time.Second)()
		defer setCleanupManagerDurations(t, time.Millisecond)()

		cleanupService := &dummyCleanupService{}
		cleanupManager := newDockerContainerCleanupManager(cleanupService)

		insertIntoCleanupManager(cleanupManager, 1)
		insertIntoCleanupManager(cleanupManager, 2)

		runCleanupManagerForNTicks(t, cleanupManager, 1)
		require.Equal(t, 0, len(cleanupService.args))
	})

	t.Run("when there is a single stale cleanup info", func(t *testing.T) {
		defer setCurrentNanoTimestamps(t,
			time.Second,           // time when inserting
			time.Hour+time.Second, // time at the 1st tick
		)()
		defer setCleanupManagerDurations(t, time.Millisecond)()

		cleanupService := &dummyCleanupService{}
		cleanupManager := newDockerContainerCleanupManager(cleanupService)

		containerID, cleanupInfo := insertIntoCleanupManager(cleanupManager, 1)

		runCleanupManagerForNTicks(t, cleanupManager, 1)
		require.Equal(t, 1, len(cleanupService.args))

		assert.Equal(t, containerID, cleanupService.args[0].id)
		assert.True(t, cleanupInfo == cleanupService.args[0].cleanupInfo)
	})

	t.Run("when there are some stale cleanup infos, and some non-stale ones", func(t *testing.T) {
		defer setCurrentNanoTimestamps(t,
			time.Second,             // time when inserting the first info
			2*time.Second,           // time when inserting the second info
			3*time.Second,           // time when inserting the 3rd info,
			4*time.Second,           // time when inserting the 4th info,
			time.Hour+2*time.Second, // time at the 1st tick
		)()
		defer setCleanupManagerDurations(t, time.Millisecond)()

		cleanupService := &dummyCleanupService{}
		cleanupManager := newDockerContainerCleanupManager(cleanupService)

		containerIDs := make([]string, 4)
		cleanupInfos := make([]*containerCleanupInfo, 4)
		for i := 1; i <= 4; i++ {
			containerID, cleanupInfo := insertIntoCleanupManager(cleanupManager, i)
			containerIDs[i-1] = containerID
			cleanupInfos[i-1] = cleanupInfo
		}

		runCleanupManagerForNTicks(t, cleanupManager, 1)
		require.Equal(t, 2, len(cleanupService.args))

		for i := 0; i < 2; i++ {
			assert.Equal(t, containerIDs[i], cleanupService.args[i].id)
			assert.True(t, cleanupInfos[i] == cleanupService.args[i].cleanupInfo)
		}
	})

	t.Run("more complicated run, with several ticks", func(t *testing.T) {
		defer setCurrentNanoTimestamps(t,
			time.Second,             // time when inserting the first info
			2*time.Second,           // time when inserting the second info
			3*time.Second,           // time when inserting the 3rd info,
			4*time.Second,           // time when inserting the 4th info,
			time.Hour,               // time at the 1st tick
			time.Hour+time.Second,   // time at the 2nd tick
			time.Hour+3*time.Second, // time at the 3rd tick
			2*time.Hour,             // time at the 4th tick
		)()
		defer setCleanupManagerDurations(t, time.Millisecond)()

		cleanupService := &dummyCleanupService{}
		cleanupManager := newDockerContainerCleanupManager(cleanupService)

		containerIDs := make([]string, 4)
		cleanupInfos := make([]*containerCleanupInfo, 4)
		for i := 1; i <= 4; i++ {
			containerID, cleanupInfo := insertIntoCleanupManager(cleanupManager, i)
			containerIDs[i-1] = containerID
			cleanupInfos[i-1] = cleanupInfo
		}

		containerIDTicks := runCleanupManagerForNTicks(t, cleanupManager, 4)

		// all containers should have been cleaned up
		require.Equal(t, 4, len(cleanupService.args))
		for i := 0; i < 4; i++ {
			assert.Equal(t, containerIDs[i], cleanupService.args[i].id)
			assert.True(t, cleanupInfos[i] == cleanupService.args[i].cleanupInfo)
		}

		// the first tick shouldn't have cleaned anything
		assert.Equal(t, 0, len(containerIDTicks[0]))
		// the 2nd tick should have cleaned up the 1st container
		assert.Equal(t, containerIDs[0:1], containerIDTicks[1])
		// the 3rd tick containers 2 and 3
		assert.Equal(t, containerIDs[1:3], containerIDTicks[2])
		// and the final tick container 4
		assert.Equal(t, containerIDs[3:], containerIDTicks[3])
	})

	t.Run("if several cleanup infos are inserted for the same container, it keeps track of the last insertion time and info", func(t *testing.T) {
		defer setCurrentNanoTimestamps(t,
			time.Second,             // time when inserting the first info
			time.Hour+time.Second,   // time when re-inserting the info for the same container
			2*time.Hour,             // time at the 1st tick
			2*time.Hour+time.Second, // time at the 2nd tick
		)()
		defer setCleanupManagerDurations(t, time.Millisecond)()

		cleanupService := &dummyCleanupService{}
		cleanupManager := newDockerContainerCleanupManager(cleanupService)

		containerID, originalCleanupInfo := insertIntoCleanupManager(cleanupManager, 1)
		_, newCleanupInfo := insertIntoCleanupManager(cleanupManager, 1)

		containerIDTicks := runCleanupManagerForNTicks(t, cleanupManager, 2)

		// the container should have been cleaned up
		require.Equal(t, 1, len(cleanupService.args))
		assert.Equal(t, containerID, cleanupService.args[0].id)
		assert.True(t, newCleanupInfo == cleanupService.args[0].cleanupInfo)
		if unsafe.Sizeof(*originalCleanupInfo) > 0 {
			// the golang compiler only ever allocates one instance of empty structs
			// so if this struct is empty, the test below will fail
			// note that it still runs on platforms where that struct is not empty, e.g. windows
			assert.False(t, originalCleanupInfo == cleanupService.args[0].cleanupInfo)
		}

		// the first tick shouldn't have cleaned anything
		assert.Equal(t, 0, len(containerIDTicks[0]))
		// the 2nd tick should have cleaned up our container
		assert.Equal(t, []string{containerID}, containerIDTicks[1])
	})
}

func insertIntoCleanupManager(cleanupManager *containerCleanupManager, i int) (string, *containerCleanupInfo) {
	containerID := "container" + strconv.Itoa(i)
	cleanupInfo := new(containerCleanupInfo)
	cleanupManager.insert(containerID, cleanupInfo)
	return containerID, cleanupInfo
}

type containerCleanupArg struct {
	id          string
	cleanupInfo *containerCleanupInfo
}

type dummyCleanupService struct {
	args []containerCleanupArg
}

func (cleanupService *dummyCleanupService) performContainerCleanup(containerID string, cleanupInfo *containerCleanupInfo) {
	cleanupService.args = append(cleanupService.args, containerCleanupArg{containerID, cleanupInfo})
}

// runCleanupManagerForNTicks creates a new create cleanup manager, and waits for it to
// run for n ticks before returning - times out if nothing happens in 5 seconds.
// Returns the container IDs that got cleaned up at each tick.
func runCleanupManagerForNTicks(t *testing.T, cleanupManager *containerCleanupManager, n int) [][]string {
	stopChan := make(chan struct{})
	doneChan := make(chan struct{})
	tickChan := make(chan []string)
	go func() {
		cleanupManager.start(stopChan, tickChan)
		close(doneChan)
	}()

	stopChanClosed := false
	closeStopChan := func() {
		if !stopChanClosed {
			close(stopChan)
			stopChanClosed = true
		}
	}

	containerIDTicks := make([][]string, n)
	tick := 0

	for {
		select {
		case containerIDs := <-tickChan:
			if tick < n {
				containerIDTicks[tick] = containerIDs
				tick++
			} else if len(containerIDs) != 0 {
				// cleane ud up containers past what we expected
				t.Errorf("Received unexpected cleaned up container IDs past the %d-th tick: %v", n, containerIDs)
			}

			if tick >= n {
				closeStopChan()
			}
		case <-doneChan:
			if stopChanClosed {
				// we're done
				return containerIDTicks
			} else {
				// this shouldn't happen
				closeStopChan()
				t.Fatal("Cleanup manager shut down before being asked to do so")
			}
		case <-time.After(5 * time.Second):
			// best effort to clean up
			closeStopChan()
			t.Fatal("Timed out waiting for cleanup manager to stop")
		}
	}
}

// setCurrentNanoTimestamps replaces the package-level currentNanoTimestampFunc variable
// with a function that will return the provided timestamps in order, then repeat the
// last timestamp of the list. It returns a function to be called to revert the change
// when done with testing.
func setCurrentNanoTimestamps(t *testing.T, timestamps ...time.Duration) func() {
	previousNanoTimestampFunc := currentNanoTimestampFunc

	// sanity check
	if len(timestamps) == 0 {
		t.Fatal("need to give at least one timestamp")
	}
	nextIndex := 0
	currentNanoTimestampFunc = func() int64 {
		value := timestamps[nextIndex]

		if nextIndex < len(timestamps)-1 {
			nextIndex += 1
		}

		return int64(value)
	}

	return func() {
		currentNanoTimestampFunc = previousNanoTimestampFunc

		// check we exhausted the given list
		if nextIndex != len(timestamps)-1 {
			t.Fatalf("didn't exhaust timestamps from %v", timestamps)
		}
	}
}

// setCleanupManagerDurations replaces the package-level staleContainerCleanupInterval and/or
// staleContainerCleanupInfoAge variables - depending how many arguments it's given.
// It returns a function to be called to revert the change when done with testing.
func setCleanupManagerDurations(t *testing.T, durations ...time.Duration) func() {
	previousStaleContainerCleanupInterval := staleContainerCleanupInterval
	previousStaleContainerCleanupInfoAge := staleContainerCleanupInfoAge

	switch len(durations) {
	case 2:
		staleContainerCleanupInfoAge = durations[1]
		fallthrough
	case 1:
		staleContainerCleanupInterval = durations[0]
	default:
		t.Fatalf("setCleanupManagerIntervals only accepts 1 or 2 duration arguments, got %v", durations)
	}

	return func() {
		staleContainerCleanupInterval = previousStaleContainerCleanupInterval
		staleContainerCleanupInfoAge = previousStaleContainerCleanupInfoAge
	}
}
