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
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/storage/backend"
)

// TestLessorGrant ensures Lessor can grant wanted lease.
// The granted lease should have a unique ID with a term
// that is greater than minLeaseTTL.
func TestLessorGrant(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	le := newLessor(be)
	le.Promote(0)

	l, err := le.Grant(1, 1)
	if err != nil {
		t.Fatalf("could not grant lease 1 (%v)", err)
	}
	gl := le.get(l.ID)

	if !reflect.DeepEqual(gl, l) {
		t.Errorf("lease = %v, want %v", gl, l)
	}
	if l.expiry.Sub(time.Now()) < time.Duration(minLeaseTTL)*time.Second-time.Second {
		t.Errorf("term = %v, want at least %v", l.expiry.Sub(time.Now()), time.Duration(minLeaseTTL)*time.Second-time.Second)
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

// TestLessorRevoke ensures Lessor can revoke a lease.
// The items in the revoked lease should be removed from
// the backend.
// The revoked lease cannot be got from Lessor again.
func TestLessorRevoke(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	fd := &fakeDeleter{}

	le := newLessor(be)
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

	if err = le.Revoke(l.ID); err != nil {
		t.Fatal("failed to revoke lease:", err)
	}

	if le.get(l.ID) != nil {
		t.Errorf("got revoked lease %x", l.ID)
	}

	wdeleted := []string{"foo_", "bar_"}
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

	le := newLessor(be)
	le.Promote(0)

	l, err := le.Grant(1, 5)
	if err != nil {
		t.Fatalf("failed to grant lease (%v)", err)
	}

	// manually change the ttl field
	l.TTL = 10
	ttl, err := le.Renew(l.ID)
	if err != nil {
		t.Fatalf("failed to renew lease (%v)", err)
	}
	if ttl != l.TTL {
		t.Errorf("ttl = %d, want %d", ttl, l.TTL)
	}

	l = le.get(l.ID)
	if l.expiry.Sub(time.Now()) < 9*time.Second {
		t.Errorf("failed to renew the lease")
	}
}

func TestLessorDetach(t *testing.T) {
	dir, be := NewTestBackend(t)
	defer os.RemoveAll(dir)
	defer be.Close()

	fd := &fakeDeleter{}

	le := newLessor(be)
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

	le := newLessor(be)
	l1, err1 := le.Grant(1, 10)
	l2, err2 := le.Grant(2, 20)
	if err1 != nil || err2 != nil {
		t.Fatalf("could not grant initial leases (%v, %v)", err1, err2)
	}

	// Create a new lessor with the same backend
	nle := newLessor(be)
	nl1 := nle.get(l1.ID)
	if nl1 == nil || nl1.TTL != l1.TTL {
		t.Errorf("nl1 = %v, want nl1.TTL= %d", nl1.TTL, l1.TTL)
	}

	nl2 := nle.get(l2.ID)
	if nl2 == nil || nl2.TTL != l2.TTL {
		t.Errorf("nl2 = %v, want nl2.TTL= %d", nl2.TTL, l2.TTL)
	}
}

type fakeDeleter struct {
	deleted []string
}

func (fd *fakeDeleter) DeleteRange(key, end []byte) (int64, int64) {
	fd.deleted = append(fd.deleted, string(key)+"_"+string(end))
	return 0, 0
}

func NewTestBackend(t *testing.T) (string, backend.Backend) {
	tmpPath, err := ioutil.TempDir("", "lease")
	if err != nil {
		t.Fatalf("failed to create tmpdir (%v)", err)
	}

	return tmpPath, backend.New(path.Join(tmpPath, "be"), time.Second, 10000)
}
