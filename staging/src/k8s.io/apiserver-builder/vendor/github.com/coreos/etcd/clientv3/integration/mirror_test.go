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
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3/mirror"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
)

func TestMirrorSync(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	c := clus.Client(0)
	_, err := c.KV.Put(context.TODO(), "foo", "bar")
	if err != nil {
		t.Fatal(err)
	}

	syncer := mirror.NewSyncer(c, "", 0)
	gch, ech := syncer.SyncBase(context.TODO())
	wkvs := []*mvccpb.KeyValue{{Key: []byte("foo"), Value: []byte("bar"), CreateRevision: 2, ModRevision: 2, Version: 1}}

	for g := range gch {
		if !reflect.DeepEqual(g.Kvs, wkvs) {
			t.Fatalf("kv = %v, want %v", g.Kvs, wkvs)
		}
	}

	for e := range ech {
		t.Fatalf("unexpected error %v", e)
	}

	wch := syncer.SyncUpdates(context.TODO())

	_, err = c.KV.Put(context.TODO(), "foo", "bar")
	if err != nil {
		t.Fatal(err)
	}

	select {
	case r := <-wch:
		wkv := &mvccpb.KeyValue{Key: []byte("foo"), Value: []byte("bar"), CreateRevision: 2, ModRevision: 3, Version: 2}
		if !reflect.DeepEqual(r.Events[0].Kv, wkv) {
			t.Fatalf("kv = %v, want %v", r.Events[0].Kv, wkv)
		}
	case <-time.After(time.Second):
		t.Fatal("failed to receive update in one second")
	}
}

func TestMirrorSyncBase(t *testing.T) {
	cluster := integration.NewClusterV3(nil, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(nil)

	cli := cluster.Client(0)
	ctx := context.TODO()

	keyCh := make(chan string)
	var wg sync.WaitGroup

	for i := 0; i < 50; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for key := range keyCh {
				if _, err := cli.Put(ctx, key, "test"); err != nil {
					t.Fatal(err)
				}
			}
		}()
	}

	for i := 0; i < 2000; i++ {
		keyCh <- fmt.Sprintf("test%d", i)
	}

	close(keyCh)
	wg.Wait()

	syncer := mirror.NewSyncer(cli, "test", 0)
	respCh, errCh := syncer.SyncBase(ctx)

	count := 0

	for resp := range respCh {
		count = count + len(resp.Kvs)
		if !resp.More {
			break
		}
	}

	for err := range errCh {
		t.Fatalf("unexpected error %v", err)
	}

	if count != 2000 {
		t.Errorf("unexpected kv count: %d", count)
	}
}
