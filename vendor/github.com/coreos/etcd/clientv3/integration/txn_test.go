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
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
)

func TestTxnError(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	_, err := kv.Txn(ctx).Then(clientv3.OpPut("foo", "bar1"), clientv3.OpPut("foo", "bar2")).Commit()
	if err != rpctypes.ErrDuplicateKey {
		t.Fatalf("expected %v, got %v", rpctypes.ErrDuplicateKey, err)
	}

	ops := make([]clientv3.Op, v3rpc.MaxOpsPerTxn+10)
	for i := range ops {
		ops[i] = clientv3.OpPut(fmt.Sprintf("foo%d", i), "")
	}
	_, err = kv.Txn(ctx).Then(ops...).Commit()
	if err != rpctypes.ErrTooManyOps {
		t.Fatalf("expected %v, got %v", rpctypes.ErrTooManyOps, err)
	}
}

func TestTxnWriteFail(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.Client(0))

	clus.Members[0].Stop(t)

	txnc, getc := make(chan struct{}), make(chan struct{})
	go func() {
		ctx, cancel := context.WithTimeout(context.TODO(), time.Second)
		defer cancel()
		resp, err := kv.Txn(ctx).Then(clientv3.OpPut("foo", "bar")).Commit()
		if err == nil {
			t.Fatalf("expected error, got response %v", resp)
		}
		close(txnc)
	}()

	go func() {
		defer close(getc)
		select {
		case <-time.After(5 * time.Second):
			t.Fatalf("timed out waiting for txn fail")
		case <-txnc:
		}
		// and ensure the put didn't take
		gresp, gerr := clus.Client(1).Get(context.TODO(), "foo")
		if gerr != nil {
			t.Fatal(gerr)
		}
		if len(gresp.Kvs) != 0 {
			t.Fatalf("expected no keys, got %v", gresp.Kvs)
		}
	}()

	select {
	case <-time.After(2 * clus.Members[1].ServerConfig.ReqTimeout()):
		t.Fatalf("timed out waiting for get")
	case <-getc:
	}

	// reconnect so terminate doesn't complain about double-close
	clus.Members[0].Restart(t)
}

func TestTxnReadRetry(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.Client(0))
	clus.Members[0].Stop(t)
	<-clus.Members[0].StopNotify()

	donec := make(chan struct{})
	go func() {
		ctx := context.TODO()
		_, err := kv.Txn(ctx).Then(clientv3.OpGet("foo")).Commit()
		if err != nil {
			t.Fatalf("expected response, got error %v", err)
		}
		donec <- struct{}{}
	}()
	// wait for txn to fail on disconnect
	time.Sleep(100 * time.Millisecond)

	// restart node; client should resume
	clus.Members[0].Restart(t)
	select {
	case <-donec:
	case <-time.After(2 * clus.Members[1].ServerConfig.ReqTimeout()):
		t.Fatalf("waited too long")
	}
}

func TestTxnSuccess(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.Client(0))
	ctx := context.TODO()

	_, err := kv.Txn(ctx).Then(clientv3.OpPut("foo", "bar")).Commit()
	if err != nil {
		t.Fatal(err)
	}

	resp, err := kv.Get(ctx, "foo")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Kvs) != 1 || string(resp.Kvs[0].Key) != "foo" {
		t.Fatalf("unexpected Get response %v", resp)
	}
}
