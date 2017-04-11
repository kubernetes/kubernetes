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

package recipe

import (
	"fmt"
	"strings"
	"time"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	"golang.org/x/net/context"
)

// RemoteKV is a key/revision pair created by the client and stored on etcd
type RemoteKV struct {
	kv  v3.KV
	key string
	rev int64
	val string
}

func NewKey(kv v3.KV, key string, leaseID v3.LeaseID) (*RemoteKV, error) {
	return NewKV(kv, key, "", leaseID)
}

func NewKV(kv v3.KV, key, val string, leaseID v3.LeaseID) (*RemoteKV, error) {
	rev, err := putNewKV(kv, key, val, leaseID)
	if err != nil {
		return nil, err
	}
	return &RemoteKV{kv, key, rev, val}, nil
}

func GetRemoteKV(kv v3.KV, key string) (*RemoteKV, error) {
	resp, err := kv.Get(context.TODO(), key)
	if err != nil {
		return nil, err
	}
	rev := resp.Header.Revision
	val := ""
	if len(resp.Kvs) > 0 {
		rev = resp.Kvs[0].ModRevision
		val = string(resp.Kvs[0].Value)
	}
	return &RemoteKV{kv: kv, key: key, rev: rev, val: val}, nil
}

func NewUniqueKey(kv v3.KV, prefix string) (*RemoteKV, error) {
	return NewUniqueKV(kv, prefix, "", 0)
}

func NewUniqueKV(kv v3.KV, prefix string, val string, leaseID v3.LeaseID) (*RemoteKV, error) {
	for {
		newKey := fmt.Sprintf("%s/%v", prefix, time.Now().UnixNano())
		rev, err := putNewKV(kv, newKey, val, 0)
		if err == nil {
			return &RemoteKV{kv, newKey, rev, val}, nil
		}
		if err != ErrKeyExists {
			return nil, err
		}
	}
}

// putNewKV attempts to create the given key, only succeeding if the key did
// not yet exist.
func putNewKV(kv v3.KV, key, val string, leaseID v3.LeaseID) (int64, error) {
	cmp := v3.Compare(v3.Version(key), "=", 0)
	req := v3.OpPut(key, val, v3.WithLease(leaseID))
	txnresp, err := kv.Txn(context.TODO()).If(cmp).Then(req).Commit()
	if err != nil {
		return 0, err
	}
	if !txnresp.Succeeded {
		return 0, ErrKeyExists
	}
	return txnresp.Header.Revision, nil
}

// NewSequentialKV allocates a new sequential key-value pair at <prefix>/nnnnn
func NewSequentialKV(kv v3.KV, prefix, val string) (*RemoteKV, error) {
	return newSequentialKV(kv, prefix, val, 0)
}

// newSequentialKV allocates a new sequential key <prefix>/nnnnn with a given
// value and lease.  Note: a bookkeeping node __<prefix> is also allocated.
func newSequentialKV(kv v3.KV, prefix, val string, leaseID v3.LeaseID) (*RemoteKV, error) {
	resp, err := kv.Get(context.TODO(), prefix, v3.WithLastKey()...)
	if err != nil {
		return nil, err
	}

	// add 1 to last key, if any
	newSeqNum := 0
	if len(resp.Kvs) != 0 {
		fields := strings.Split(string(resp.Kvs[0].Key), "/")
		_, serr := fmt.Sscanf(fields[len(fields)-1], "%d", &newSeqNum)
		if serr != nil {
			return nil, serr
		}
		newSeqNum++
	}
	newKey := fmt.Sprintf("%s/%016d", prefix, newSeqNum)

	// base prefix key must be current (i.e., <=) with the server update;
	// the base key is important to avoid the following:
	// N1: LastKey() == 1, start txn.
	// N2: New Key 2, New Key 3, Delete Key 2
	// N1: txn succeeds allocating key 2 when it shouldn't
	baseKey := "__" + prefix

	// current revision might contain modification so +1
	cmp := v3.Compare(v3.ModRevision(baseKey), "<", resp.Header.Revision+1)
	reqPrefix := v3.OpPut(baseKey, "", v3.WithLease(leaseID))
	reqNewKey := v3.OpPut(newKey, val, v3.WithLease(leaseID))

	txn := kv.Txn(context.TODO())
	txnresp, err := txn.If(cmp).Then(reqPrefix, reqNewKey).Commit()
	if err != nil {
		return nil, err
	}
	if !txnresp.Succeeded {
		return newSequentialKV(kv, prefix, val, leaseID)
	}
	return &RemoteKV{kv, newKey, txnresp.Header.Revision, val}, nil
}

func (rk *RemoteKV) Key() string     { return rk.key }
func (rk *RemoteKV) Revision() int64 { return rk.rev }
func (rk *RemoteKV) Value() string   { return rk.val }

func (rk *RemoteKV) Delete() error {
	if rk.kv == nil {
		return nil
	}
	_, err := rk.kv.Delete(context.TODO(), rk.key)
	rk.kv = nil
	return err
}

func (rk *RemoteKV) Put(val string) error {
	_, err := rk.kv.Put(context.TODO(), rk.key, val)
	return err
}

// EphemeralKV is a new key associated with a session lease
type EphemeralKV struct{ RemoteKV }

// NewEphemeralKV creates a new key/value pair associated with a session lease
func NewEphemeralKV(client *v3.Client, key, val string) (*EphemeralKV, error) {
	s, err := concurrency.NewSession(client)
	if err != nil {
		return nil, err
	}
	k, err := NewKV(client, key, val, s.Lease())
	if err != nil {
		return nil, err
	}
	return &EphemeralKV{*k}, nil
}

// NewUniqueEphemeralKey creates a new unique valueless key associated with a session lease
func NewUniqueEphemeralKey(client *v3.Client, prefix string) (*EphemeralKV, error) {
	return NewUniqueEphemeralKV(client, prefix, "")
}

// NewUniqueEphemeralKV creates a new unique key/value pair associated with a session lease
func NewUniqueEphemeralKV(client *v3.Client, prefix, val string) (ek *EphemeralKV, err error) {
	for {
		newKey := fmt.Sprintf("%s/%v", prefix, time.Now().UnixNano())
		ek, err = NewEphemeralKV(client, newKey, val)
		if err == nil || err != ErrKeyExists {
			break
		}
	}
	return ek, err
}
