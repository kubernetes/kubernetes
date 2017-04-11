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

package integration

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/clientv3/concurrency"
	"golang.org/x/net/context"
)

// TestSTMConflict tests that conflicts are retried.
func TestSTMConflict(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	etcdc := clus.RandClient()
	keys := make([]string, 5)
	for i := 0; i < len(keys); i++ {
		keys[i] = fmt.Sprintf("foo-%d", i)
		if _, err := etcdc.Put(context.TODO(), keys[i], "100"); err != nil {
			t.Fatalf("could not make key (%v)", err)
		}
	}

	errc := make(chan error)
	for i := range keys {
		curEtcdc := clus.RandClient()
		srcKey := keys[i]
		applyf := func(stm concurrency.STM) error {
			src := stm.Get(srcKey)
			// must be different key to avoid double-adding
			dstKey := srcKey
			for dstKey == srcKey {
				dstKey = keys[rand.Intn(len(keys))]
			}
			dst := stm.Get(dstKey)
			srcV, _ := strconv.ParseInt(src, 10, 64)
			dstV, _ := strconv.ParseInt(dst, 10, 64)
			if srcV == 0 {
				// can't rand.Intn on 0, so skip this transaction
				return nil
			}
			xfer := int64(rand.Intn(int(srcV)) / 2)
			stm.Put(srcKey, fmt.Sprintf("%d", srcV-xfer))
			stm.Put(dstKey, fmt.Sprintf("%d", dstV+xfer))
			return nil
		}
		go func() {
			_, err := concurrency.NewSTMRepeatable(context.TODO(), curEtcdc, applyf)
			errc <- err
		}()
	}

	// wait for txns
	for range keys {
		if err := <-errc; err != nil {
			t.Fatalf("apply failed (%v)", err)
		}
	}

	// ensure sum matches initial sum
	sum := 0
	for _, oldkey := range keys {
		rk, err := etcdc.Get(context.TODO(), oldkey)
		if err != nil {
			t.Fatalf("couldn't fetch key %s (%v)", oldkey, err)
		}
		v, _ := strconv.ParseInt(string(rk.Kvs[0].Value), 10, 64)
		sum += int(v)
	}
	if sum != len(keys)*100 {
		t.Fatalf("bad sum. got %d, expected %d", sum, len(keys)*100)
	}
}

// TestSTMPutNewKey confirms a STM put on a new key is visible after commit.
func TestSTMPutNewKey(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	etcdc := clus.RandClient()
	applyf := func(stm concurrency.STM) error {
		stm.Put("foo", "bar")
		return nil
	}
	if _, err := concurrency.NewSTMRepeatable(context.TODO(), etcdc, applyf); err != nil {
		t.Fatalf("error on stm txn (%v)", err)
	}

	resp, err := etcdc.Get(context.TODO(), "foo")
	if err != nil {
		t.Fatalf("error fetching key (%v)", err)
	}
	if string(resp.Kvs[0].Value) != "bar" {
		t.Fatalf("bad value. got %+v, expected 'bar' value", resp)
	}
}

// TestSTMAbort tests that an aborted txn does not modify any keys.
func TestSTMAbort(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	etcdc := clus.RandClient()
	ctx, cancel := context.WithCancel(context.TODO())
	applyf := func(stm concurrency.STM) error {
		stm.Put("foo", "baz")
		cancel()
		stm.Put("foo", "bap")
		return nil
	}
	if _, err := concurrency.NewSTMRepeatable(ctx, etcdc, applyf); err == nil {
		t.Fatalf("no error on stm txn")
	}

	resp, err := etcdc.Get(context.TODO(), "foo")
	if err != nil {
		t.Fatalf("error fetching key (%v)", err)
	}
	if len(resp.Kvs) != 0 {
		t.Fatalf("bad value. got %+v, expected nothing", resp)
	}
}

// TestSTMSerialize tests that serialization is honored when serializable.
func TestSTMSerialize(t *testing.T) {
	clus := NewClusterV3(t, &ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	etcdc := clus.RandClient()

	// set up initial keys
	keys := make([]string, 5)
	for i := 0; i < len(keys); i++ {
		keys[i] = fmt.Sprintf("foo-%d", i)
	}

	// update keys in full batches
	updatec := make(chan struct{})
	go func() {
		defer close(updatec)
		for i := 0; i < 5; i++ {
			s := fmt.Sprintf("%d", i)
			ops := []v3.Op{}
			for _, k := range keys {
				ops = append(ops, v3.OpPut(k, s))
			}
			if _, err := etcdc.Txn(context.TODO()).Then(ops...).Commit(); err != nil {
				t.Fatalf("couldn't put keys (%v)", err)
			}
			updatec <- struct{}{}
		}
	}()

	// read all keys in txn, make sure all values match
	errc := make(chan error)
	for range updatec {
		curEtcdc := clus.RandClient()
		applyf := func(stm concurrency.STM) error {
			vs := []string{}
			for i := range keys {
				vs = append(vs, stm.Get(keys[i]))
			}
			for i := range vs {
				if vs[0] != vs[i] {
					return fmt.Errorf("got vs[%d] = %v, want %v", i, vs[i], vs[0])
				}
			}
			return nil
		}
		go func() {
			_, err := concurrency.NewSTMSerializable(context.TODO(), curEtcdc, applyf)
			errc <- err
		}()
	}

	for i := 0; i < 5; i++ {
		if err := <-errc; err != nil {
			t.Error(err)
		}
	}
}
