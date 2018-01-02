// Copyright 2016 The etcd Authors
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
	"fmt"
	"sync"

	v3 "github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

// Mutex implements the sync Locker interface with etcd
type Mutex struct {
	s *Session

	pfx   string
	myKey string
	myRev int64
}

func NewMutex(s *Session, pfx string) *Mutex {
	return &Mutex{s, pfx + "/", "", -1}
}

// Lock locks the mutex with a cancellable context. If the context is cancelled
// while trying to acquire the lock, the mutex tries to clean its stale lock entry.
func (m *Mutex) Lock(ctx context.Context) error {
	s := m.s
	client := m.s.Client()

	m.myKey = fmt.Sprintf("%s%x", m.pfx, s.Lease())
	cmp := v3.Compare(v3.CreateRevision(m.myKey), "=", 0)
	// put self in lock waiters via myKey; oldest waiter holds lock
	put := v3.OpPut(m.myKey, "", v3.WithLease(s.Lease()))
	// reuse key in case this session already holds the lock
	get := v3.OpGet(m.myKey)
	resp, err := client.Txn(ctx).If(cmp).Then(put).Else(get).Commit()
	if err != nil {
		return err
	}
	m.myRev = resp.Header.Revision
	if !resp.Succeeded {
		m.myRev = resp.Responses[0].GetResponseRange().Kvs[0].CreateRevision
	}

	// wait for deletion revisions prior to myKey
	err = waitDeletes(ctx, client, m.pfx, m.myRev-1)
	// release lock key if cancelled
	select {
	case <-ctx.Done():
		m.Unlock(client.Ctx())
	default:
	}
	return err
}

func (m *Mutex) Unlock(ctx context.Context) error {
	client := m.s.Client()
	if _, err := client.Delete(ctx, m.myKey); err != nil {
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
	client := lm.s.Client()
	if err := lm.Mutex.Lock(client.Ctx()); err != nil {
		panic(err)
	}
}
func (lm *lockerMutex) Unlock() {
	client := lm.s.Client()
	if err := lm.Mutex.Unlock(client.Ctx()); err != nil {
		panic(err)
	}
}

// NewLocker creates a sync.Locker backed by an etcd mutex.
func NewLocker(s *Session, pfx string) sync.Locker {
	return &lockerMutex{NewMutex(s, pfx)}
}
