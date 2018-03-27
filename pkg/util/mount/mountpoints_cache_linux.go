// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package mount

import (
	"sync"
	"time"

	"github.com/golang/glog"
	"golang.org/x/sys/unix"
	"k8s.io/apimachinery/pkg/util/wait"
)

// mountPointsCache implements cache of mountpoints on linux platform.
// We use poll() on /proc/mounts to get notified about mount tree changes.
type mountPointsCache struct {
	sync.RWMutex
	// Cache of mountpoints.
	mountpoints []MountPoint
	// Indicates cache is stale or not.
	stale bool
}

func newMountPointsCache() *mountPointsCache {
	cache := &mountPointsCache{
		mountpoints: nil,
		stale:       true,
	}
	return cache
}

// Get gets mountpoints cache if cache is fresh, otherwise calls `list()` to
// list mountpoints. If `list()` succeeds, update cache and mark it fresh,
// otherwise returns error.
func (c *mountPointsCache) Get(list func() ([]MountPoint, error)) ([]MountPoint, error) {
	c.RLock()
	if !c.stale {
		c.RUnlock()
		return c.mountpoints, nil
	}
	c.RUnlock()
	c.Lock()
	defer c.Unlock()
	mountpoints, err := list()
	if err != nil {
		return nil, err
	}
	c.mountpoints = mountpoints
	c.stale = false
	return c.mountpoints, nil
}

// markStale marks cache stale.
func (c *mountPointsCache) markStale() {
	c.Lock()
	defer c.Unlock()
	c.stale = true
}

// poll polls on `/proc/mounts` to get notified when file is changed. If file
// is changed and an error occurs, mark cache stale.
// It returns an error in following cases:
// - could not open /proc/mounts
// - interrupted by signal
// - fds exceeds RLIMIT_NOFILE
// - no space to allocated file descriptors table
func (c *mountPointsCache) poll(stopCh <-chan struct{}) error {
	// According to http://man7.org/linux/man-pages/man5/proc.5.html, since
	// 2.6.15, "/proc/mounts" is pollable.
	// Between Linux 2.6.15 and 2.6.29, file descriptior is marked as having an
	// error condition (POLLERR) for poll(2).
	// After 2.6.30, file descriptor is marked as having an error condition
	// (POLLERR) and a priority event (POLLPRI) for poll(2).
	// For best compatibility, we check POLLERR here.
	fd, err := unix.Open(procMountsPath, unix.O_RDONLY, 0)
	if err != nil {
		return err
	}
	pollfds := []unix.PollFd{
		{
			Fd:      int32(fd),
			Events:  0,
			Revents: 0,
		},
	}
	for {
		select {
		case <-stopCh:
			return nil
		}
		n, err := unix.Poll(pollfds, 50)
		if err != nil {
			c.markStale()
			return err
		}
		if n <= 0 {
			// timed out, poll again
			continue
		}
		if pollfds[0].Revents&unix.POLLERR != 0 {
			c.markStale()
		}
		pollfds[0].Revents = 0
	}
}

func (c *mountPointsCache) Run(stopCh <-chan struct{}) {
	wait.Until(func() {
		// In all error cases, we should retry again.
		err := c.poll(stopCh)
		if err != nil {
			glog.Errorf("mountPointsCache.poll() failed with error: %v, retry again later", err)
		}
	}, time.Second, stopCh)
}
