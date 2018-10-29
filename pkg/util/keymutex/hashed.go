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

package keymutex

import (
	"hash/fnv"
	"runtime"
	"sync"

	"github.com/golang/glog"
)

// NewHashed returns a new instance of KeyMutex which hashes arbitrary keys to
// a fixed set of locks. `n` specifies number of locks, if n <= 0, we use
// number of cpus.
// Note that because it uses fixed set of locks, different keys may share same
// lock, so it's possible to wait on same lock.
func NewHashed(n int) KeyMutex {
	if n <= 0 {
		n = runtime.NumCPU()
	}
	return &hashedKeyMutex{
		mutexes: make([]sync.Mutex, n),
	}
}

type hashedKeyMutex struct {
	mutexes []sync.Mutex
}

// Acquires a lock associated with the specified ID.
func (km *hashedKeyMutex) LockKey(id string) {
	glog.V(5).Infof("hashedKeyMutex.LockKey(...) called for id %q\r\n", id)
	km.mutexes[km.hash(id)%len(km.mutexes)].Lock()
	glog.V(5).Infof("hashedKeyMutex.LockKey(...) for id %q completed.\r\n", id)
}

// Releases the lock associated with the specified ID.
func (km *hashedKeyMutex) UnlockKey(id string) error {
	glog.V(5).Infof("hashedKeyMutex.UnlockKey(...) called for id %q\r\n", id)
	km.mutexes[km.hash(id)%len(km.mutexes)].Unlock()
	glog.V(5).Infof("hashedKeyMutex.UnlockKey(...) for id %q completed.\r\n", id)
	return nil
}

func (km *hashedKeyMutex) hash(id string) int {
	h := fnv.New32a()
	h.Write([]byte(id))
	return int(h.Sum32())
}
