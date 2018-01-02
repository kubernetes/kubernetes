// Copyright 2015 The etcd Authors
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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/coreos/etcd/mvcc/backend"
)

const (
	minLeaseTTL         = int64(5)
	minLeaseTTLDuration = time.Duration(minLeaseTTL) * time.Second
)

// TestLessorGrant ensures Lessor can grant wanted lease.
// The granted lease should have a unique ID with a term
// that is greater than minLeaseTTL.
func TestLessorGrant(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	le := newLessor(be, minLeaseTTL)
	le.Promote(0)

	l, err := le.Grant(1, 1)
	if err != nil {
		t.Fatalf("could not grant lease 1 (%v)", err)
	}
	gl := le.Lookup(l.ID)

	if !reflect.DeepEqual(gl, l) {
		t.Errorf("lease = %v, want %v", gl, l)
	}
	if l.Remaining() < minLeaseTTLDuration-time.Second {
		t.Errorf("term = %v, want at least %v", l.Remaining(), minLeaseTTLDuration-time.Second)
	}

	nl, err := le.Grant(1, 1)
	if err == nil {
		t.Errorf("allocated the same lease")
	}

	nl, err = le.Grant(2, 1)
	if err != nil {
		t.Errorf("could not grant lease 2 (%v)", err)
	}
	if nl.ID == l.ID {
		t.Errorf("new lease.id = %x, want != %x", nl.ID, l.ID)
	}

	be.BatchTx().Lock()
	_, vs := be.BatchTx().UnsafeRange(leaseBucketName, int64ToBytes(int64(l.ID)), nil, 0)
	if len(vs) != 1 {
		t.Errorf("len(vs) = %d, want 1", len(vs))
	}
	be.BatchTx().Unlock()
}

// TestLeaseConcurrentKeys ensures Lease.Keys method calls are guarded
// from concurrent map writes on 'itemSet'.
func TestLeaseConcurrentKeys(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	fd := &fakeDeleter{}

	le := newLessor(be, minLeaseTTL)
	le.SetRangeDeleter(fd)

	// grant a lease with long term (100 seconds) to
	// avoid early termination during the test.
	l, err := le.Grant(1, 100)
	if err != nil {
		t.Fatalf("could not grant lease for 100s ttl (%v)", err)
	}

	itemn := 10
	items := make([]LeaseItem, itemn)
	for i := 0; i < itemn; i++ {
		items[i] = LeaseItem{Key: fmt.Sprintf("foo%d", i)}
	}
	if err = le.Attach(l.ID, items); err != nil {
		t.Fatalf("failed to attach items to the lease: %v", err)
	}

	donec := make(chan struct{})
	go func() {
		le.Detach(l.ID, items)
		close(donec)
	}()

	var wg sync.WaitGroup
	wg.Add(itemn)
	for i := 0; i < itemn; i++ {
		go func() {
			defer wg.Done()
			l.Keys()
		}()
	}

	<-donec
	wg.Wait()
}

// TestLessorRevoke ensures Lessor can revoke a lease.
// The items in the revoked lease should be removed from
// the backend.
// The revoked lease cannot be got from Lessor again.
func TestLessorRevoke(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	fd := &fakeDeleter{}

	le := newLessor(be, minLeaseTTL)
	le.SetRangeDeleter(fd)

	// grant a lease with long term (100 seconds) to
	// avoid early termination during the test.
	l, err := le.Grant(1, 100)
	if err != nil {
		t.Fatalf("could not grant lease for 100s ttl (%v)", err)
	}

	items := []LeaseItem{
		{"foo"},
		{"bar"},
	}

	if err = le.Attach(l.ID, items); err != nil {
		t.Fatalf("failed to attach items to the lease: %v", err)
	}

	if err = le.Revoke(l.ID); err != nil {
		t.Fatal("failed to revoke lease:", err)
	}

	if le.Lookup(l.ID) != nil {
		t.Errorf("got revoked lease %x", l.ID)
	}

	wdeleted := []string{"bar_", "foo_"}
	sort.Sort(sort.StringSlice(fd.deleted))
	if !reflect.DeepEqual(fd.deleted, wdeleted) {
		t.Errorf("deleted= %v, want %v", fd.deleted, wdeleted)
	}

	be.BatchTx().Lock()
	_, vs := be.BatchTx().UnsafeRange(leaseBucketName, int64ToBytes(int64(l.ID)), nil, 0)
	if len(vs) != 0 {
		t.Errorf("len(vs) = %d, want 0", len(vs))
	}
	be.BatchTx().Unlock()
}

// TestLessorRenew ensures Lessor can renew an existing lease.
func TestLessorRenew(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer be.Close()
	defer os.RemoveAll(dir)

	le := newLessor(be, minLeaseTTL)
	le.Promote(0)

	l, err := le.Grant(1, minLeaseTTL)
	if err != nil {
		t.Fatalf("failed to grant lease (%v)", err)
	}

	// manually change the ttl field
	le.mu.Lock()
	l.ttl = 10
	le.mu.Unlock()
	ttl, err := le.Renew(l.ID)
	if err != nil {
		t.Fatalf("failed to renew lease (%v)", err)
	}
	if ttl != l.ttl {
		t.Errorf("ttl = %d, want %d", ttl, l.ttl)
	}

	l = le.Lookup(l.ID)
	if l.Remaining() < 9*time.Second {
		t.Errorf("failed to renew the lease")
	}
}

func TestLessorDetach(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	fd := &fakeDeleter{}

	le := newLessor(be, minLeaseTTL)
	le.SetRangeDeleter(fd)

	// grant a lease with long term (100 seconds) to
	// avoid early termination during the test.
	l, err := le.Grant(1, 100)
	if err != nil {
		t.Fatalf("could not grant lease for 100s ttl (%v)", err)
	}

	items := []LeaseItem{
		{"foo"},
		{"bar"},
	}

	if err := le.Attach(l.ID, items); err != nil {
		t.Fatalf("failed to attach items to the lease: %v", err)
	}

	if err := le.Detach(l.ID, items[0:1]); err != nil {
		t.Fatalf("failed to de-attach items to the lease: %v", err)
	}

	l = le.Lookup(l.ID)
	if len(l.itemSet) != 1 {
		t.Fatalf("len(l.itemSet) = %d, failed to de-attach items", len(l.itemSet))
	}
	if _, ok := l.itemSet[LeaseItem{"bar"}]; !ok {
		t.Fatalf("de-attached wrong item, want %q exists", "bar")
	}
}

// TestLessorRecover ensures Lessor recovers leases from
// persist backend.
func TestLessorRecover(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	le := newLessor(be, minLeaseTTL)
	l1, err1 := le.Grant(1, 10)
	l2, err2 := le.Grant(2, 20)
	if err1 != nil || err2 != nil {
		t.Fatalf("could not grant initial leases (%v, %v)", err1, err2)
	}

	// Create a new lessor with the same backend
	nle := newLessor(be, minLeaseTTL)
	nl1 := nle.Lookup(l1.ID)
	if nl1 == nil || nl1.ttl != l1.ttl {
		t.Errorf("nl1 = %v, want nl1.ttl= %d", nl1.ttl, l1.ttl)
	}

	nl2 := nle.Lookup(l2.ID)
	if nl2 == nil || nl2.ttl != l2.ttl {
		t.Errorf("nl2 = %v, want nl2.ttl= %d", nl2.ttl, l2.ttl)
	}
}

func TestLessorExpire(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	testMinTTL := int64(1)

	le := newLessor(be, testMinTTL)
	defer le.Stop()

	le.Promote(1 * time.Second)
	l, err := le.Grant(1, testMinTTL)
	if err != nil {
		t.Fatalf("failed to create lease: %v", err)
	}

	select {
	case el := <-le.ExpiredLeasesC():
		if el[0].ID != l.ID {
			t.Fatalf("expired id = %x, want %x", el[0].ID, l.ID)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("failed to receive expired lease")
	}

	donec := make(chan struct{})
	go func() {
		// expired lease cannot be renewed
		if _, err := le.Renew(l.ID); err != ErrLeaseNotFound {
			t.Fatalf("unexpected renew")
		}
		donec <- struct{}{}
	}()

	select {
	case <-donec:
		t.Fatalf("renew finished before lease revocation")
	case <-time.After(50 * time.Millisecond):
	}

	// expired lease can be revoked
	if err := le.Revoke(l.ID); err != nil {
		t.Fatalf("failed to revoke expired lease: %v", err)
	}

	select {
	case <-donec:
	case <-time.After(10 * time.Second):
		t.Fatalf("renew has not returned after lease revocation")
	}
}

func TestLessorExpireAndDemote(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	testMinTTL := int64(1)

	le := newLessor(be, testMinTTL)
	defer le.Stop()

	le.Promote(1 * time.Second)
	l, err := le.Grant(1, testMinTTL)
	if err != nil {
		t.Fatalf("failed to create lease: %v", err)
	}

	select {
	case el := <-le.ExpiredLeasesC():
		if el[0].ID != l.ID {
			t.Fatalf("expired id = %x, want %x", el[0].ID, l.ID)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("failed to receive expired lease")
	}

	donec := make(chan struct{})
	go func() {
		// expired lease cannot be renewed
		if _, err := le.Renew(l.ID); err != ErrNotPrimary {
			t.Fatalf("unexpected renew: %v", err)
		}
		donec <- struct{}{}
	}()

	select {
	case <-donec:
		t.Fatalf("renew finished before demotion")
	case <-time.After(50 * time.Millisecond):
	}

	// demote will cause the renew request to fail with ErrNotPrimary
	le.Demote()

	select {
	case <-donec:
	case <-time.After(10 * time.Second):
		t.Fatalf("renew has not returned after lessor demotion")
	}
}

type fakeDeleter struct {
	deleted []string
}

func (fd *fakeDeleter) TxnBegin() int64 {
	return 0
}

func (fd *fakeDeleter) TxnEnd(txnID int64) error {
	return nil
}

func (fd *fakeDeleter) TxnDeleteRange(tid int64, key, end []byte) (int64, int64, error) {
	fd.deleted = append(fd.deleted, string(key)+"_"+string(end))
	return 0, 0, nil
}

func NewTestBackend(t *testing.T) (string, backend.Backend) {
	tmpPath, err := ioutil.TempDir("", "lease")
	if err != nil {
		t.Fatalf("failed to create tmpdir (%v)", err)
	}

	return tmpPath, backend.New(filepath.Join(tmpPath, "be"), time.Second, 10000)
}
