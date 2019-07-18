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

/*
This file contains the structure responsible to clean up stale container clean up infos,
i.e. those that have still not been performed more than staleContainerCleanupInfoAge after
the container's creation.
*/

package dockershim

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/orderedmap"
)

type containerCleanupManager struct {
	cleanupService containerCleanupService
	cleanupInfos   *orderedmap.OrderedMap
	mutex          *sync.Mutex
}

type containerCleanupInfoWithTimestamp struct {
	cleanupInfo *containerCleanupInfo
	timestamp   int64
}

// An interface containing the one of dockerService's methods that we need here;
// allows mocking it in tests.
type containerCleanupService interface {
	performContainerCleanup(containerID string, cleanupInfo *containerCleanupInfo)
}

func newDockerContainerCleanupManager(cleanupService containerCleanupService) *containerCleanupManager {
	return &containerCleanupManager{
		cleanupService: cleanupService,
		cleanupInfos:   orderedmap.New(),
		mutex:          &sync.Mutex{},
	}
}

// start makes the manager periodically review the cleanup infos it knows about and clean up the
// ones that become stale.
// This is a blocking call.
// It can be passed up to do channels: the first one will stop the manager when closed;
// The second one will be sent to after each "tick", ie call to cleanupStaleContainerCleanupInfos,
// giving the IDs of containers that got cleaned up, if any.
// These channels are mainly intended for tests, and both can be left nil.
func (cm *containerCleanupManager) start(stopChannel <-chan struct{}, tickChannel chan []string) {
	if stopChannel == nil {
		stopChannel = wait.NeverStop
	}

	wait.Until(func() {
		containerIDs := cm.cleanupStaleContainerCleanupInfos()
		if tickChannel != nil {
			tickChannel <- containerIDs
		}
	}, staleContainerCleanupInterval, stopChannel)
}

// Having those as variables makes them easy to mock in unit tests.
var (
	currentNanoTimestampFunc = func() int64 {
		return time.Now().UnixNano()
	}

	// the period in between 2 calls to cleanupStaleContainers once started
	staleContainerCleanupInterval = 5 * time.Minute

	// cleanup infos older than this much will be considered stale, and cleaned up
	staleContainerCleanupInfoAge = 1 * time.Hour
)

// insert allows keeping track of a new container's cleanup info.
func (cm *containerCleanupManager) insert(containerID string, cleanupInfo *containerCleanupInfo) {
	if cleanupInfo == nil {
		return
	}

	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	if _, present := cm.cleanupInfos.Get(containerID); present {
		// shouldn't happen, as we only insert at container creation time - but to err on the side of
		// caution, let's delete from and re-insert into the ordered map to ensure that the order
		// stays chronological
		klog.Errorf("duplicate cleanup info for container ID %q", containerID)
		cm.cleanupInfos.Delete(containerID)
	}

	cm.cleanupInfos.Set(containerID, &containerCleanupInfoWithTimestamp{
		cleanupInfo: cleanupInfo,
		timestamp:   currentNanoTimestampFunc(),
	})
}

func (cm *containerCleanupManager) performCleanup(containerID string) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	cm.unsafePerformCleanup(containerID)
}

// unsafePerformCleanup is the same as performCleanup, but assumes the manager's lock has already
// been acquired.
func (cm *containerCleanupManager) unsafePerformCleanup(containerID string) {
	if cleanupInfoWithTimestamp, present := cm.cleanupInfos.Delete(containerID); present {
		cleanupInfo := cleanupInfoWithTimestamp.(*containerCleanupInfoWithTimestamp).cleanupInfo
		cm.cleanupService.performContainerCleanup(containerID, cleanupInfo)
	}
}

// cleanupStaleContainerCleanupInfos runs the clean up for clean up infos older than staleContainerCleanupInfoAge
func (cm *containerCleanupManager) cleanupStaleContainerCleanupInfos() []string {
	timestampCutoff := currentNanoTimestampFunc() - int64(staleContainerCleanupInfoAge)
	containerIDsToCleanup := make([]string, 0)

	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	for pair := cm.cleanupInfos.Oldest(); pair != nil; pair = pair.Next() {
		cleanupInfoWithTimestamp := pair.Value.(*containerCleanupInfoWithTimestamp)

		if cleanupInfoWithTimestamp.timestamp > timestampCutoff {
			// this one is not old enough to be cleaned up yet, and all remaining ones are newer than this one, we're done
			break
		}

		containerID := pair.Key.(string)
		containerIDsToCleanup = append(containerIDsToCleanup, containerID)

		klog.Warningf("performing stale clean up for container %q", containerID)
	}

	for _, containerID := range containerIDsToCleanup {
		cm.unsafePerformCleanup(containerID)
	}

	return containerIDsToCleanup
}
