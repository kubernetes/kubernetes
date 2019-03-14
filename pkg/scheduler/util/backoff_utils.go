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

	"k8s.io/klog"
)

type clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time {
	return time.Now()
}

// backoffEntry is single threaded.  in particular, it only allows a single action to be waiting on backoff at a time.
// It is also not safe to copy this object.
type backoffEntry struct {
	initialized bool
	podName     ktypes.NamespacedName
	backoff     time.Duration
	lastUpdate  time.Time
	reqInFlight int32
}

// tryLock attempts to acquire a lock via atomic compare and swap.
// returns true if the lock was acquired, false otherwise
func (b *backoffEntry) tryLock() bool {
	return atomic.CompareAndSwapInt32(&b.reqInFlight, 0, 1)
}

// unlock returns the lock.  panics if the lock isn't held
func (b *backoffEntry) unlock() {
	if !atomic.CompareAndSwapInt32(&b.reqInFlight, 1, 0) {
		panic(fmt.Sprintf("unexpected state on unlocking: %+v", b))
	}
}

// backoffTime returns the Time when a backoffEntry completes backoff
func (b *backoffEntry) backoffTime() time.Time {
	return b.lastUpdate.Add(b.backoff)
}

// getBackoff returns the duration until this entry completes backoff
func (b *backoffEntry) getBackoff(maxDuration time.Duration) time.Duration {
	if !b.initialized {
		b.initialized = true
		return b.backoff
	}
	newDuration := b.backoff * 2
	if newDuration > maxDuration {
		newDuration = maxDuration
	}
	b.backoff = newDuration
	klog.V(4).Infof("Backing off %s", newDuration.String())
	return newDuration
}

// PodBackoff is used to restart a pod with back-off delay.
type PodBackoff struct {
	// expiryQ stores backoffEntry orderedy by lastUpdate until they reach maxDuration and are GC'd
	expiryQ         *Heap
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
		expiryQ:         NewHeap(backoffEntryKeyFunc, backoffEntryCompareUpdate),
		clock:           clock,
		defaultDuration: defaultDuration,
		maxDuration:     maxDuration,
	}
}

// getEntry returns the backoffEntry for a given podID
func (p *PodBackoff) getEntry(podID ktypes.NamespacedName) *backoffEntry {
	entry, exists, _ := p.expiryQ.GetByKey(podID.String())
	var be *backoffEntry
	if !exists {
		be = &backoffEntry{
			initialized: false,
			podName:     podID,
			backoff:     p.defaultDuration,
		}
		p.expiryQ.Update(be)
	} else {
		be = entry.(*backoffEntry)
	}
	return be
}

// BackoffPod updates the backoff for a podId and returns the duration until backoff completion
func (p *PodBackoff) BackoffPod(podID ktypes.NamespacedName) time.Duration {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry := p.getEntry(podID)
	entry.lastUpdate = p.clock.Now()
	p.expiryQ.Update(entry)
	return entry.getBackoff(p.maxDuration)
}

// TryBackoffAndWait tries to acquire the backoff lock
func (p *PodBackoff) TryBackoffAndWait(podID ktypes.NamespacedName, stop <-chan struct{}) bool {
	p.lock.Lock()
	entry := p.getEntry(podID)

	if !entry.tryLock() {
		p.lock.Unlock()
		return false
	}
	defer entry.unlock()
	duration := entry.getBackoff(p.maxDuration)
	p.lock.Unlock()
	select {
	case <-time.After(duration):
		return true
	case <-stop:
		return false
	}
}

// Gc execute garbage collection on the pod back-off.
func (p *PodBackoff) Gc() {
	p.lock.Lock()
	defer p.lock.Unlock()
	now := p.clock.Now()
	var be *backoffEntry
	for {
		entry := p.expiryQ.Peek()
		if entry == nil {
			break
		}
		be = entry.(*backoffEntry)
		if now.Sub(be.lastUpdate) > p.maxDuration {
			p.expiryQ.Pop()
		} else {
			break
		}
	}
}

// GetBackoffTime returns the time that podID completes backoff
func (p *PodBackoff) GetBackoffTime(podID ktypes.NamespacedName) (time.Time, bool) {
	p.lock.Lock()
	defer p.lock.Unlock()
	rawBe, exists, _ := p.expiryQ.GetByKey(podID.String())
	if !exists {
		return time.Time{}, false
	}
	be := rawBe.(*backoffEntry)
	return be.lastUpdate.Add(be.backoff), true
}

// ClearPodBackoff removes all tracking information for podID (clears expiry)
func (p *PodBackoff) ClearPodBackoff(podID ktypes.NamespacedName) bool {
	p.lock.Lock()
	defer p.lock.Unlock()
	entry, exists, _ := p.expiryQ.GetByKey(podID.String())
	if exists {
		err := p.expiryQ.Delete(entry)
		return err == nil
	}
	return false
}

// backoffEntryKeyFunc is the keying function used for mapping a backoffEntry to string for heap
func backoffEntryKeyFunc(b interface{}) (string, error) {
	be := b.(*backoffEntry)
	return be.podName.String(), nil
}

// backoffEntryCompareUpdate returns true when b1's backoff time is before b2's
func backoffEntryCompareUpdate(b1, b2 interface{}) bool {
	be1 := b1.(*backoffEntry)
	be2 := b2.(*backoffEntry)
	return be1.lastUpdate.Before(be2.lastUpdate)
}
