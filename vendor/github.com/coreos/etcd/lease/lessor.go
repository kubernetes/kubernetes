// Copyright 2015 CoreOS, Inc.
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

package lease

import (
	"encoding/binary"
	"errors"
	"math"
	"sync"
	"time"

	"github.com/coreos/etcd/lease/leasepb"
	"github.com/coreos/etcd/storage/backend"
)

const (
	// NoLease is a special LeaseID representing the absence of a lease.
	NoLease = LeaseID(0)
)

var (
	minLeaseTTL = int64(5)

	leaseBucketName = []byte("lease")
	// do not use maxInt64 since it can overflow time which will add
	// the offset of unix time (1970yr to seconds).
	forever = time.Unix(math.MaxInt64>>1, 0)

	ErrNotPrimary    = errors.New("not a primary lessor")
	ErrLeaseNotFound = errors.New("lease not found")
	ErrLeaseExists   = errors.New("lease already exists")
)

type LeaseID int64

// RangeDeleter defines an interface with DeleteRange method.
// We define this interface only for lessor to limit the number
// of methods of storage.KV to what lessor actually needs.
//
// Having a minimum interface makes testing easy.
type RangeDeleter interface {
	DeleteRange(key, end []byte) (int64, int64)
}

// A Lessor is the owner of leases. It can grant, revoke, renew and modify leases for lessee.
type Lessor interface {
	// SetRangeDeleter sets the RangeDeleter to the Lessor.
	// Lessor deletes the items in the revoked or expired lease from the
	// the set RangeDeleter.
	SetRangeDeleter(dr RangeDeleter)

	// Grant grants a lease that expires at least after TTL seconds.
	Grant(id LeaseID, ttl int64) (*Lease, error)
	// Revoke revokes a lease with given ID. The item attached to the
	// given lease will be removed. If the ID does not exist, an error
	// will be returned.
	Revoke(id LeaseID) error

	// Attach attaches given leaseItem to the lease with given LeaseID.
	// If the lease does not exist, an error will be returned.
	Attach(id LeaseID, items []LeaseItem) error

	// Detach detaches given leaseItem from the lease with given LeaseID.
	// If the lease does not exist, an error will be returned.
	Detach(id LeaseID, items []LeaseItem) error

	// Promote promotes the lessor to be the primary lessor. Primary lessor manages
	// the expiration and renew of leases.
	// Newly promoted lessor renew the TTL of all lease to extend + previous TTL.
	Promote(extend time.Duration)

	// Demote demotes the lessor from being the primary lessor.
	Demote()

	// Renew renews a lease with given ID. It returns the renewed TTL. If the ID does not exist,
	// an error will be returned.
	Renew(id LeaseID) (int64, error)

	// Lookup gives the lease at a given lease id, if any
	Lookup(id LeaseID) *Lease

	// ExpiredLeasesC returns a chan that is used to receive expired leases.
	ExpiredLeasesC() <-chan []*Lease

	// Recover recovers the lessor state from the given backend and RangeDeleter.
	Recover(b backend.Backend, rd RangeDeleter)

	// Stop stops the lessor for managing leases. The behavior of calling Stop multiple
	// times is undefined.
	Stop()
}

// lessor implements Lessor interface.
// TODO: use clockwork for testability.
type lessor struct {
	mu sync.Mutex

	// primary indicates if this lessor is the primary lessor. The primary
	// lessor manages lease expiration and renew.
	//
	// in etcd, raft leader is the primary. Thus there might be two primary
	// leaders at the same time (raft allows concurrent leader but with different term)
	// for at most a leader election timeout.
	// The old primary leader cannot affect the correctness since its proposal has a
	// smaller term and will not be committed.
	//
	// TODO: raft follower do not forward lease management proposals. There might be a
	// very small window (within second normally which depends on go scheduling) that
	// a raft follow is the primary between the raft leader demotion and lessor demotion.
	// Usually this should not be a problem. Lease should not be that sensitive to timing.
	primary bool

	// TODO: probably this should be a heap with a secondary
	// id index.
	// Now it is O(N) to loop over the leases to find expired ones.
	// We want to make Grant, Revoke, and findExpiredLeases all O(logN) and
	// Renew O(1).
	// findExpiredLeases and Renew should be the most frequent operations.
	leaseMap map[LeaseID]*Lease

	// When a lease expires, the lessor will delete the
	// leased range (or key) by the RangeDeleter.
	rd RangeDeleter

	// backend to persist leases. We only persist lease ID and expiry for now.
	// The leased items can be recovered by iterating all the keys in kv.
	b backend.Backend

	expiredC chan []*Lease
	// stopC is a channel whose closure indicates that the lessor should be stopped.
	stopC chan struct{}
	// doneC is a channel whose closure indicates that the lessor is stopped.
	doneC chan struct{}
}

func NewLessor(b backend.Backend) Lessor {
	return newLessor(b)
}

func newLessor(b backend.Backend) *lessor {
	l := &lessor{
		leaseMap: make(map[LeaseID]*Lease),
		b:        b,
		// expiredC is a small buffered chan to avoid unnecessary blocking.
		expiredC: make(chan []*Lease, 16),
		stopC:    make(chan struct{}),
		doneC:    make(chan struct{}),
	}
	l.initAndRecover()

	go l.runLoop()

	return l
}

func (le *lessor) SetRangeDeleter(rd RangeDeleter) {
	le.mu.Lock()
	defer le.mu.Unlock()

	le.rd = rd
}

// TODO: when lessor is under high load, it should give out lease
// with longer TTL to reduce renew load.
func (le *lessor) Grant(id LeaseID, ttl int64) (*Lease, error) {
	if id == NoLease {
		return nil, ErrLeaseNotFound
	}

	l := &Lease{ID: id, TTL: ttl, itemSet: make(map[LeaseItem]struct{})}

	le.mu.Lock()
	defer le.mu.Unlock()

	if _, ok := le.leaseMap[id]; ok {
		return nil, ErrLeaseExists
	}

	if le.primary {
		l.refresh(0)
	} else {
		l.forever()
	}

	le.leaseMap[id] = l
	l.persistTo(le.b)

	return l, nil
}

func (le *lessor) Revoke(id LeaseID) error {
	le.mu.Lock()

	l := le.leaseMap[id]
	if l == nil {
		le.mu.Unlock()
		return ErrLeaseNotFound
	}
	// unlock before doing external work
	le.mu.Unlock()

	if le.rd != nil {
		for item := range l.itemSet {
			le.rd.DeleteRange([]byte(item.Key), nil)
		}
	}

	le.mu.Lock()
	defer le.mu.Unlock()
	delete(le.leaseMap, l.ID)
	l.removeFrom(le.b)

	return nil
}

// Renew renews an existing lease. If the given lease does not exist or
// has expired, an error will be returned.
func (le *lessor) Renew(id LeaseID) (int64, error) {
	le.mu.Lock()
	defer le.mu.Unlock()

	if !le.primary {
		// forward renew request to primary instead of returning error.
		return -1, ErrNotPrimary
	}

	l := le.leaseMap[id]
	if l == nil {
		return -1, ErrLeaseNotFound
	}

	l.refresh(0)
	return l.TTL, nil
}

func (le *lessor) Lookup(id LeaseID) *Lease {
	le.mu.Lock()
	defer le.mu.Unlock()
	if l, ok := le.leaseMap[id]; ok {
		return l
	}
	return nil
}

func (le *lessor) Promote(extend time.Duration) {
	le.mu.Lock()
	defer le.mu.Unlock()

	le.primary = true

	// refresh the expiries of all leases.
	for _, l := range le.leaseMap {
		l.refresh(extend)
	}
}

func (le *lessor) Demote() {
	le.mu.Lock()
	defer le.mu.Unlock()

	// set the expiries of all leases to forever
	for _, l := range le.leaseMap {
		l.forever()
	}

	le.primary = false
}

// Attach attaches items to the lease with given ID. When the lease
// expires, the attached items will be automatically removed.
// If the given lease does not exist, an error will be returned.
func (le *lessor) Attach(id LeaseID, items []LeaseItem) error {
	le.mu.Lock()
	defer le.mu.Unlock()

	l := le.leaseMap[id]
	if l == nil {
		return ErrLeaseNotFound
	}

	for _, it := range items {
		l.itemSet[it] = struct{}{}
	}
	return nil
}

// Detach detaches items from the lease with given ID.
// If the given lease does not exist, an error will be returned.
func (le *lessor) Detach(id LeaseID, items []LeaseItem) error {
	le.mu.Lock()
	defer le.mu.Unlock()

	l := le.leaseMap[id]
	if l == nil {
		return ErrLeaseNotFound
	}

	for _, it := range items {
		delete(l.itemSet, it)
	}
	return nil
}

func (le *lessor) Recover(b backend.Backend, rd RangeDeleter) {
	le.mu.Lock()
	defer le.mu.Unlock()

	le.b = b
	le.rd = rd
	le.leaseMap = make(map[LeaseID]*Lease)

	le.initAndRecover()
}

func (le *lessor) ExpiredLeasesC() <-chan []*Lease {
	return le.expiredC
}

func (le *lessor) Stop() {
	close(le.stopC)
	<-le.doneC
}

func (le *lessor) runLoop() {
	defer close(le.doneC)

	for {
		var ls []*Lease

		le.mu.Lock()
		if le.primary {
			ls = le.findExpiredLeases()
		}
		le.mu.Unlock()

		if len(ls) != 0 {
			select {
			case <-le.stopC:
				return
			case le.expiredC <- ls:
			default:
				// the receiver of expiredC is probably busy handling
				// other stuff
				// let's try this next time after 500ms
			}
		}

		select {
		case <-time.After(500 * time.Millisecond):
		case <-le.stopC:
			return
		}
	}
}

// findExpiredLeases loops all the leases in the leaseMap and returns the expired
// leases that needed to be revoked.
func (le *lessor) findExpiredLeases() []*Lease {
	leases := make([]*Lease, 0, 16)
	now := time.Now()

	for _, l := range le.leaseMap {
		// TODO: probably should change to <= 100-500 millisecond to
		// make up committing latency.
		if l.expiry.Sub(now) <= 0 {
			leases = append(leases, l)
		}
	}

	return leases
}

// get gets the lease with given id.
// get is a helper function for testing, at least for now.
func (le *lessor) get(id LeaseID) *Lease {
	le.mu.Lock()
	defer le.mu.Unlock()

	return le.leaseMap[id]
}

func (le *lessor) initAndRecover() {
	tx := le.b.BatchTx()
	tx.Lock()

	tx.UnsafeCreateBucket(leaseBucketName)
	_, vs := tx.UnsafeRange(leaseBucketName, int64ToBytes(0), int64ToBytes(math.MaxInt64), 0)
	// TODO: copy vs and do decoding outside tx lock if lock contention becomes an issue.
	for i := range vs {
		var lpb leasepb.Lease
		err := lpb.Unmarshal(vs[i])
		if err != nil {
			tx.Unlock()
			panic("failed to unmarshal lease proto item")
		}
		ID := LeaseID(lpb.ID)
		le.leaseMap[ID] = &Lease{
			ID:  ID,
			TTL: lpb.TTL,
			// itemSet will be filled in when recover key-value pairs
			// set expiry to forever, refresh when promoted
			itemSet: make(map[LeaseItem]struct{}),
			expiry:  forever,
		}
	}
	tx.Unlock()

	le.b.ForceCommit()
}

type Lease struct {
	ID  LeaseID
	TTL int64 // time to live in seconds

	itemSet map[LeaseItem]struct{}
	// expiry time in unixnano
	expiry time.Time
}

func (l Lease) persistTo(b backend.Backend) {
	key := int64ToBytes(int64(l.ID))

	lpb := leasepb.Lease{ID: int64(l.ID), TTL: int64(l.TTL)}
	val, err := lpb.Marshal()
	if err != nil {
		panic("failed to marshal lease proto item")
	}

	b.BatchTx().Lock()
	b.BatchTx().UnsafePut(leaseBucketName, key, val)
	b.BatchTx().Unlock()
}

func (l Lease) removeFrom(b backend.Backend) {
	key := int64ToBytes(int64(l.ID))

	b.BatchTx().Lock()
	b.BatchTx().UnsafeDelete(leaseBucketName, key)
	b.BatchTx().Unlock()
}

// refresh refreshes the expiry of the lease. It extends the expiry at least
// minLeaseTTL second.
func (l *Lease) refresh(extend time.Duration) {
	if l.TTL < minLeaseTTL {
		l.TTL = minLeaseTTL
	}
	l.expiry = time.Now().Add(extend + time.Second*time.Duration(l.TTL))
}

// forever sets the expiry of lease to be forever.
func (l *Lease) forever() {
	if l.TTL < minLeaseTTL {
		l.TTL = minLeaseTTL
	}
	l.expiry = forever
}

type LeaseItem struct {
	Key string
}

func int64ToBytes(n int64) []byte {
	bytes := make([]byte, 8)
	binary.BigEndian.PutUint64(bytes, uint64(n))
	return bytes
}

// FakeLessor is a fake implementation of Lessor interface.
// Used for testing only.
type FakeLessor struct{}

func (fl *FakeLessor) SetRangeDeleter(dr RangeDeleter) {}

func (fl *FakeLessor) Grant(id LeaseID, ttl int64) (*Lease, error) { return nil, nil }

func (fl *FakeLessor) Revoke(id LeaseID) error { return nil }

func (fl *FakeLessor) Attach(id LeaseID, items []LeaseItem) error { return nil }

func (fl *FakeLessor) Detach(id LeaseID, items []LeaseItem) error { return nil }

func (fl *FakeLessor) Promote(extend time.Duration) {}

func (fl *FakeLessor) Demote() {}

func (fl *FakeLessor) Renew(id LeaseID) (int64, error) { return 10, nil }

func (le *FakeLessor) Lookup(id LeaseID) *Lease { return nil }

func (fl *FakeLessor) ExpiredLeasesC() <-chan []*Lease { return nil }

func (fl *FakeLessor) Recover(b backend.Backend, rd RangeDeleter) {}

func (fl *FakeLessor) Stop() {}
