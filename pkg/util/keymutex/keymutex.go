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

package keymutex

import (
	"fmt"
	"github.com/golang/glog"
	"sync"
)

type tryMutex struct {
	ch chan struct{}
}

func newTryMutex() *tryMutex {
	return &tryMutex{ch: make(chan struct{}, 1)}
}

func (tm *tryMutex) Lock() {
	tm.ch <- struct{}{}
}

func (tm *tryMutex) TryLock() bool {
	select {
	case tm.ch <- struct{}{}:
		return true
	default:
		return false
	}
}

func (tm *tryMutex) Unlock() {
	select {
	case <-tm.ch:
		return
	default:
		panic("unlock of unlocked mutex")
	}
}

// KeyMutex is a thread-safe interface for acquiring locks on arbitrary strings.
type KeyMutex interface {
	// Acquires a lock associated with the specified ID, creates the lock if one doesn't already exist.
	LockKey(id string)

	// Tries to acquire a lock associated with the specified ID, returns immediately.
	// Returns false if the lock exists and is already acquired, otherwise returns true.
	TryLockKey(id string) bool

	// Releases the lock associated with the specified ID.
	// Returns an error if the specified ID doesn't exist.
	UnlockKey(id string) error
}

// Returns a new instance of a key mutex.
func NewKeyMutex() KeyMutex {
	return &keyMutex{
		mutexMap: make(map[string]*tryMutex),
	}
}

type keyMutex struct {
	sync.RWMutex
	mutexMap map[string]*tryMutex
}

// Acquires a lock associated with the specified ID (creates the lock if one doesn't already exist).
func (km *keyMutex) LockKey(id string) {
	glog.V(5).Infof("LockKey(...) called for id %q\r\n", id)
	mutex := km.getOrCreateLock(id)
	mutex.Lock()
	glog.V(5).Infof("LockKey(...) for id %q completed.\r\n", id)
}

// Tries to acquire a lock associated with the specified ID (creates the lock if one doesn't already exist).
// Returns false if the lock exists and is already acquired, otherwise returns true.
func (km *keyMutex) TryLockKey(id string) bool {
	glog.V(5).Infof("TryLockKey(...) called for id %q\r\n", id)

	mutex := km.getOrCreateLock(id)
	if mutex.TryLock() {
		glog.V(5).Infof("TryLockKey(...) for id %q completed.\r\n", id)
		return true
	}

	glog.V(5).Infof("TryLockKey(...) for id %q could not acquire the lock.\r\n", id)
	return false
}

// Releases the lock associated with the specified ID.
// Returns an error if the specified ID doesn't exist.
func (km *keyMutex) UnlockKey(id string) error {
	glog.V(5).Infof("UnlockKey(...) called for id %q\r\n", id)
	km.RLock()
	defer km.RUnlock()
	mutex, exists := km.mutexMap[id]
	if !exists {
		return fmt.Errorf("id %q not found", id)
	}
	glog.V(5).Infof("UnlockKey(...) for id. Mutex found, trying to unlock it. %q\r\n", id)

	mutex.Unlock()
	glog.V(5).Infof("UnlockKey(...) for id %q completed.\r\n", id)
	return nil
}

// Returns lock associated with the specified ID, or creates the lock if one doesn't already exist.
func (km *keyMutex) getOrCreateLock(id string) *tryMutex {
	km.Lock()
	defer km.Unlock()

	if _, exists := km.mutexMap[id]; !exists {
		km.mutexMap[id] = newTryMutex()
	}

	return km.mutexMap[id]
}
