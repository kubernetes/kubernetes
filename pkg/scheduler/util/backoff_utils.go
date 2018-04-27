/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	ktypes "k8s.io/apimachinery/pkg/types"

	"github.com/golang/glog"
)

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time {
	return time.Now()
}

// BackoffEntry is single threaded.  in particular, it only allows a single action to be waiting on backoff at a time.
// It is expected that all users will only use the public TryWait(...) method
// It is also not safe to copy this object.
type BackoffEntry struct {
	backoff     time.Duration
	lastUpdate  time.Time
	reqInFlight int32
}

// tryLock attempts to acquire a lock via atomic compare and swap.
// returns true if the lock was acquired, false otherwise
func (b *BackoffEntry) tryLock() bool {
	return atomic.CompareAndSwapInt32(&b.reqInFlight, 0, 1)
}

// unlock returns the lock.  panics if the lock isn't held
func (b *BackoffEntry) unlock() {
	if !atomic.CompareAndSwapInt32(&b.reqInFlight, 1, 0) {
		panic(fmt.Sprintf("unexpected state on unlocking: %+v", b))
	}
}

// TryWait tries to acquire the backoff lock, maxDuration is the maximum allowed period to wait for.
func (b *BackoffEntry) TryWait(maxDuration time.Duration) bool {
	if !b.tryLock() {
		return false
	}
	defer b.unlock()
	b.wait(maxDuration)
	return true
}

func (b *BackoffEntry) getBackoff(maxDuration time.Duration) time.Duration {
	duration := b.backoff
	newDuration := time.Duration(duration) * 2
	if newDuration > maxDuration {
		newDuration = maxDuration
	}
	b.backoff = newDuration
	glog.V(4).Infof("Backing off %s", duration.String())
	return duration
}

func (b *BackoffEntry) wait(maxDuration time.Duration) {
	time.Sleep(b.getBackoff(maxDuration))
}

// PodBackoff is used to restart a pod with back-off delay.
type PodBackoff struct {
	perPodBackoff   map[ktypes.NamespacedName]*BackoffEntry
	lock            sync.Mutex
	clock           clock
	defaultDuration time.Duration
	maxDuration     time.Duration
}

// MaxDuration returns the max time duration of the back-off.
func (p *PodBackoff) MaxDuration() time.Duration {
	return p.maxDuration
}

// CreateDefaultPodBackoff creates a default pod back-off object.
func CreateDefaultPodBackoff() *PodBackoff {
	return CreatePodBackoff(1*time.Second, 60*time.Second)
}

// CreatePodBackoff creates a pod back-off object by default duration and max duration.
func CreatePodBackoff(defaultDuration, maxDuration time.Duration) *PodBackoff {
	return CreatePodBackoffWithClock(defaultDuration, maxDuration, realClock{})
}

// CreatePodBackoffWithClock creates a pod back-off object by default duration, max duration and clock.
func CreatePodBackoffWithClock(defaultDuration, maxDuration time.Duration, clock clock) *PodBackoff {
	return &PodBackoff{
		perPodBackoff:   map[ktypes.NamespacedName]*BackoffEntry{},
		clock:           clock,
		defaultDuration: defaultDuration,
		maxDuration:     maxDuration,
	}
}

// GetEntry returns a back-off entry by Pod ID.
func (p *PodBackoff) GetEntry(podID ktypes.NamespacedName) *BackoffEntry {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, ok := p.perPodBackoff[podID]
	if !ok {
		entry = &BackoffEntry{backoff: p.defaultDuration}
		p.perPodBackoff[podID] = entry
	}
	entry.lastUpdate = p.clock.Now()
	return entry
}

// Gc execute garbage collection on the pod back-off.
func (p *PodBackoff) Gc() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	for podID, entry := range p.perPodBackoff {
		if now.Sub(entry.lastUpdate) > p.maxDuration {
			delete(p.perPodBackoff, podID)
		}
	}
}
