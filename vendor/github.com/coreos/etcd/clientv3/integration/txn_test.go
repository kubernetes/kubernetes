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

package integration

import (
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
)

func TestTxnWriteFail(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.Client(0))
	ctx := context.TODO()

	clus.Members[0].Stop(t)
	<-clus.Members[0].StopNotify()

	donec := make(chan struct{})
	go func() {
		resp, err := kv.Txn(ctx).Then(clientv3.OpPut("foo", "bar")).Commit()
		if err == nil {
			t.Fatalf("expected error, got response %v", resp)
		}
		donec <- struct{}{}
	}()

	dialTimeout := 5 * time.Second
	select {
	case <-time.After(2*dialTimeout + time.Second):
		t.Fatalf("timed out waiting for txn to fail")
	case <-donec:
		// don't restart cluster until txn errors out
	}

	go func() {
		// reconnect so terminate doesn't complain about double-close
		clus.Members[0].Restart(t)
		// wait for etcdserver to get established (CI races and get req times out)
		time.Sleep(2 * time.Second)
		donec <- struct{}{}

		// and ensure the put didn't take
		gresp, gerr := kv.Get(ctx, "foo")
		if gerr != nil {
			t.Fatal(gerr)
		}
		if len(gresp.Kvs) != 0 {
			t.Fatalf("expected no keys, got %v", gresp.Kvs)
		}
		donec <- struct{}{}
	}()

	select {
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for restart")
	case <-donec:
	}

	select {
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for get")
	case <-donec:
	}
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
	case <-time.After(5 * time.Second):
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
