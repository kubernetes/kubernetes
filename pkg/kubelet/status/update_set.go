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

package status

import (
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

// Invariants:
// 		A UID in the "set" must also be in the "cache"
// 		A UID in the "channel" must also be in the "set"
// 		Two identical UIDs are never in the "channel" at once
type updateSet struct {
	lock    sync.RWMutex
	cache   map[types.UID]v1.PodStatus
	set     map[types.UID]struct{}
	channel chan types.UID
}

type Update struct {
	UID    types.UID
	Status v1.PodStatus
}

// UpdateSet is a thread-safe representation of a set of updates.  It can be thought of as a combination of
// a key-value store, and a deduplicated channel of keys.  In our case, the key is a UID, and the value is the PodStatus.
type UpdateSet interface {
	// Get returns the status provided by the most recent Set call to the given UID.
	// If no such call has been made, or Delete has been called since the last Set, return false.
	Get(uid types.UID) (v1.PodStatus, bool)

	// Set sets the most recent status for that uid, causing subsequent Get calls to return that status.
	// The uid whose status is set is guaranteed to come out of the Update channel in an update, although
	// the status passed here may be replaced by a more recent one.
	Set(uid types.UID, status v1.PodStatus)

	// Retry guarantees that the provided uid comes out of the Update channel in an Update,
	// along with the most recent status passed to Set.
	// Setting a delay other than NoDelay causes the update to be placed in the Update channel after a delay.
	Retry(uid types.UID, delay time.Duration)

	// Updates provides a channel of Update structs.  The Status of the update is the status most recently provided to Set.
	// An update is placed in the channel after a Set or Retry, but multiple Sets or Retrys before drawing from the channel
	// will only result in one Update.  The returned channel from Updates is meant to be consumed in parallel with Set and Retry calls,
	// and continuously provides updates based on those Set and Retry calls.
	Updates() <-chan Update

	// Delete causes the next call to Get (without a call to Set) for the given UID
	// to return false, and not to return a status.  After a Delete, the deleted UID will not
	// emerge from the channel returned by Updates().
	Delete(uid types.UID)

	// GarbageCollect causes subsequent calls ot Get (without a call to Set)
	// for UIDs not included in the remainingUIDSet to return false.
	GarbageCollect(remainingUIDSet map[types.UID]struct{})
}

func NewUpdateSet() UpdateSet {
	return &updateSet{
		cache:   make(map[types.UID]v1.PodStatus),
		set:     make(map[types.UID]struct{}),
		channel: make(chan types.UID, 1000),
	}
}

func (u *updateSet) Get(uid types.UID) (v1.PodStatus, bool) {
	u.lock.Lock()
	defer u.lock.Unlock()
	status, ok := u.cache[uid]
	return status, ok
}

func (u *updateSet) Set(uid types.UID, status v1.PodStatus) {
	u.lock.Lock()
	defer u.lock.Unlock()
	u.cache[uid] = status
	_, inSet := u.set[uid]
	u.set[uid] = struct{}{}
	if !inSet {
		u.channel <- uid
	}
}

const NoDelay = 0 * time.Second

func (u *updateSet) Retry(uid types.UID, delay time.Duration) {
	u.lock.Lock()
	defer u.lock.Unlock()
	_, inCache := u.cache[uid]
	_, inSet := u.set[uid]
	u.set[uid] = struct{}{}
	if !inSet && inCache {
		if delay == NoDelay {
			u.channel <- uid
		} else {
			go func() {
				time.Sleep(delay)
				u.channel <- uid
			}()
		}
	}
	return
}

func (u *updateSet) Updates() <-chan Update {
	outputChan := make(chan Update)
	go func() {
		for uid := range u.channel {
			u.lock.Lock()
			delete(u.set, uid)
			status, ok := u.cache[uid]
			u.lock.Unlock()
			if ok {
				outputChan <- Update{
					UID:    uid,
					Status: status,
				}
			}
		}
	}()
	return outputChan
}

func (u *updateSet) Delete(uid types.UID) {
	u.lock.Lock()
	defer u.lock.Unlock()
	delete(u.set, uid)
	delete(u.cache, uid)
}

func (u *updateSet) GarbageCollect(remainingUIDSet map[types.UID]struct{}) {
	u.lock.Lock()
	defer u.lock.Unlock()
	for key := range u.cache {
		if _, ok := remainingUIDSet[key]; !ok {
			glog.V(5).Infof("Removing %q from status update set.", key)
			delete(u.set, key)
			delete(u.cache, key)
		}
	}
}
