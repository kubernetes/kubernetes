// Copyright 2016 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package concurrency

import (
	"sync"

	v3 "github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

// Mutex implements the sync Locker interface with etcd
type Mutex struct {
	client *v3.Client

	pfx   string
	myKey string
	myRev int64
}

func NewMutex(client *v3.Client, pfx string) *Mutex {
	return &Mutex{client, pfx, "", -1}
}

// Lock locks the mutex with a cancellable context. If the context is cancelled
// while trying to acquire the lock, the mutex tries to clean its stale lock entry.
func (m *Mutex) Lock(ctx context.Context) error {
	s, err := NewSession(m.client)
	if err != nil {
		return err
	}
	// put self in lock waiters via myKey; oldest waiter holds lock
	m.myKey, m.myRev, err = NewUniqueKey(ctx, m.client, m.pfx, v3.WithLease(s.Lease()))
	// wait for deletion revisions prior to myKey
	err = waitDeletes(ctx, m.client, m.pfx, v3.WithPrefix(), v3.WithRev(m.myRev-1))
	// release lock key if cancelled
	select {
	case <-ctx.Done():
		m.Unlock()
	default:
	}
	return err
}

func (m *Mutex) Unlock() error {
	if _, err := m.client.Delete(m.client.Ctx(), m.myKey); err != nil {
		return err
	}
	m.myKey = "\x00"
	m.myRev = -1
	return nil
}

func (m *Mutex) IsOwner() v3.Cmp {
	return v3.Compare(v3.CreateRevision(m.myKey), "=", m.myRev)
}

func (m *Mutex) Key() string { return m.myKey }

type lockerMutex struct{ *Mutex }

func (lm *lockerMutex) Lock() {
	if err := lm.Mutex.Lock(lm.client.Ctx()); err != nil {
		panic(err)
	}
}
func (lm *lockerMutex) Unlock() {
	if err := lm.Mutex.Unlock(); err != nil {
		panic(err)
	}
}

// NewLocker creates a sync.Locker backed by an etcd mutex.
func NewLocker(client *v3.Client, pfx string) sync.Locker {
	return &lockerMutex{NewMutex(client, pfx)}
}
