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
	"bytes"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/storage/storagepb"
	"golang.org/x/net/context"
)

func TestKVPut(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	resp, err := lapi.Create(context.Background(), 10)
	if err != nil {
		t.Fatalf("failed to create lease %v", err)
	}

	tests := []struct {
		key, val string
		leaseID  clientv3.LeaseID
	}{
		{"foo", "bar", clientv3.NoLease},
		{"hello", "world", clientv3.LeaseID(resp.ID)},
	}

	for i, tt := range tests {
		if _, err := kv.Put(ctx, tt.key, tt.val, clientv3.WithLease(tt.leaseID)); err != nil {
			t.Fatalf("#%d: couldn't put %q (%v)", i, tt.key, err)
		}
		resp, err := kv.Get(ctx, tt.key)
		if err != nil {
			t.Fatalf("#%d: couldn't get key (%v)", i, err)
		}
		if len(resp.Kvs) != 1 {
			t.Fatalf("#%d: expected 1 key, got %d", i, len(resp.Kvs))
		}
		if !bytes.Equal([]byte(tt.val), resp.Kvs[0].Value) {
			t.Errorf("#%d: val = %s, want %s", i, tt.val, resp.Kvs[0].Value)
		}
		if tt.leaseID != clientv3.LeaseID(resp.Kvs[0].Lease) {
			t.Errorf("#%d: val = %d, want %d", i, tt.leaseID, resp.Kvs[0].Lease)
		}
	}
}

func TestKVRange(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	keySet := []string{"a", "b", "c", "c", "c", "foo", "foo/abc", "fop"}
	for i, key := range keySet {
		if _, err := kv.Put(ctx, key, ""); err != nil {
			t.Fatalf("#%d: couldn't put %q (%v)", i, key, err)
		}
	}
	resp, err := kv.Get(ctx, keySet[0])
	if err != nil {
		t.Fatalf("couldn't get key (%v)", err)
	}
	wheader := resp.Header

	tests := []struct {
		begin, end string
		rev        int64
		opts       []clientv3.OpOption

		wantSet []*storagepb.KeyValue
	}{
		// range first two
		{
			"a", "c",
			0,
			nil,

			[]*storagepb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
			},
		},
		// range first two with serializable
		{
			"a", "c",
			0,
			[]clientv3.OpOption{clientv3.WithSerializable()},

			[]*storagepb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
			},
		},
		// range all with rev
		{
			"a", "x",
			2,
			nil,

			[]*storagepb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
			},
		},
		// range all with SortByKey, SortAscend
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend)},

			[]*storagepb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
			},
		},
		// range all with SortByCreatedRev, SortDescend
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByCreatedRev, clientv3.SortDescend)},

			[]*storagepb.KeyValue{
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
			},
		},
		// range all with SortByModifiedRev, SortDescend
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByModifiedRev, clientv3.SortDescend)},

			[]*storagepb.KeyValue{
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
			},
		},
		// WithPrefix
		{
			"foo", "",
			0,
			[]clientv3.OpOption{clientv3.WithPrefix()},

			[]*storagepb.KeyValue{
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
			},
		},
		// WithFromKey
		{
			"fo", "",
			0,
			[]clientv3.OpOption{clientv3.WithFromKey()},

			[]*storagepb.KeyValue{
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
			},
		},
	}

	for i, tt := range tests {
		opts := []clientv3.OpOption{clientv3.WithRange(tt.end), clientv3.WithRev(tt.rev)}
		opts = append(opts, tt.opts...)
		resp, err := kv.Get(ctx, tt.begin, opts...)
		if err != nil {
			t.Fatalf("#%d: couldn't range (%v)", i, err)
		}
		if !reflect.DeepEqual(wheader, resp.Header) {
			t.Fatalf("#%d: wheader expected %+v, got %+v", i, wheader, resp.Header)
		}
		if !reflect.DeepEqual(tt.wantSet, resp.Kvs) {
			t.Fatalf("#%d: resp.Kvs expected %+v, got %+v", i, tt.wantSet, resp.Kvs)
		}
	}
}

func TestKVDeleteRange(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	tests := []struct {
		key  string
		opts []clientv3.OpOption

		wkeys []string
	}{
		// [a, c)
		{
			key:  "a",
			opts: []clientv3.OpOption{clientv3.WithRange("c")},

			wkeys: []string{"c", "c/abc", "d"},
		},
		// >= c
		{
			key:  "c",
			opts: []clientv3.OpOption{clientv3.WithFromKey()},

			wkeys: []string{"a", "b"},
		},
		// c*
		{
			key:  "c",
			opts: []clientv3.OpOption{clientv3.WithPrefix()},

			wkeys: []string{"a", "b", "d"},
		},
		// *
		{
			key:  "\x00",
			opts: []clientv3.OpOption{clientv3.WithFromKey()},

			wkeys: []string{},
		},
	}

	for i, tt := range tests {
		keySet := []string{"a", "b", "c", "c/abc", "d"}
		for j, key := range keySet {
			if _, err := kv.Put(ctx, key, ""); err != nil {
				t.Fatalf("#%d: couldn't put %q (%v)", j, key, err)
			}
		}

		_, err := kv.Delete(ctx, tt.key, tt.opts...)
		if err != nil {
			t.Fatalf("#%d: couldn't delete range (%v)", i, err)
		}

		resp, err := kv.Get(ctx, "a", clientv3.WithFromKey())
		if err != nil {
			t.Fatalf("#%d: couldn't get keys (%v)", i, err)
		}
		keys := []string{}
		for _, kv := range resp.Kvs {
			keys = append(keys, string(kv.Key))
		}
		if !reflect.DeepEqual(tt.wkeys, keys) {
			t.Errorf("#%d: resp.Kvs got %v, expected %v", i, keys, tt.wkeys)
		}
	}
}

func TestKVDelete(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	presp, err := kv.Put(ctx, "foo", "")
	if err != nil {
		t.Fatalf("couldn't put 'foo' (%v)", err)
	}
	if presp.Header.Revision != 2 {
		t.Fatalf("presp.Header.Revision got %d, want %d", presp.Header.Revision, 2)
	}
	resp, err := kv.Delete(ctx, "foo")
	if err != nil {
		t.Fatalf("couldn't delete key (%v)", err)
	}
	if resp.Header.Revision != 3 {
		t.Fatalf("resp.Header.Revision got %d, want %d", resp.Header.Revision, 3)
	}
	gresp, err := kv.Get(ctx, "foo")
	if err != nil {
		t.Fatalf("couldn't get key (%v)", err)
	}
	if len(gresp.Kvs) > 0 {
		t.Fatalf("gresp.Kvs got %+v, want none", gresp.Kvs)
	}
}

func TestKVCompact(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	for i := 0; i < 10; i++ {
		if _, err := kv.Put(ctx, "foo", "bar"); err != nil {
			t.Fatalf("couldn't put 'foo' (%v)", err)
		}
	}

	err := kv.Compact(ctx, 7)
	if err != nil {
		t.Fatalf("couldn't compact kv space (%v)", err)
	}
	err = kv.Compact(ctx, 7)
	if err == nil || err != rpctypes.ErrCompacted {
		t.Fatalf("error got %v, want %v", err, rpctypes.ErrFutureRev)
	}

	wc := clientv3.NewWatcher(clus.RandClient())
	defer wc.Close()
	wchan := wc.Watch(ctx, "foo", clientv3.WithRev(3))

	if wr := <-wchan; wr.CompactRevision != 7 {
		t.Fatalf("wchan CompactRevision got %v, want 7", wr.CompactRevision)
	}
	if wr, ok := <-wchan; ok {
		t.Fatalf("wchan got %v, expected closed", wr)
	}

	err = kv.Compact(ctx, 1000)
	if err == nil || err != rpctypes.ErrFutureRev {
		t.Fatalf("error got %v, want %v", err, rpctypes.ErrFutureRev)
	}
}

// TestKVGetRetry ensures get will retry on disconnect.
func TestKVGetRetry(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.Client(0))
	ctx := context.TODO()

	if _, err := kv.Put(ctx, "foo", "bar"); err != nil {
		t.Fatal(err)
	}

	clus.Members[0].Stop(t)
	<-clus.Members[0].StopNotify()

	donec := make(chan struct{})
	go func() {
		// Get will fail, but reconnect will trigger
		gresp, gerr := kv.Get(ctx, "foo")
		if gerr != nil {
			t.Fatal(gerr)
		}
		wkvs := []*storagepb.KeyValue{
			{
				Key:            []byte("foo"),
				Value:          []byte("bar"),
				CreateRevision: 2,
				ModRevision:    2,
				Version:        1,
			},
		}
		if !reflect.DeepEqual(gresp.Kvs, wkvs) {
			t.Fatalf("bad get: got %v, want %v", gresp.Kvs, wkvs)
		}
		donec <- struct{}{}
	}()

	time.Sleep(100 * time.Millisecond)
	clus.Members[0].Restart(t)

	select {
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for get")
	case <-donec:
	}
}

// TestKVPutFailGetRetry ensures a get will retry following a failed put.
func TestKVPutFailGetRetry(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.Client(0))
	ctx := context.TODO()

	clus.Members[0].Stop(t)
	<-clus.Members[0].StopNotify()

	_, err := kv.Put(ctx, "foo", "bar")
	if err == nil {
		t.Fatalf("got success on disconnected put, wanted error")
	}

	donec := make(chan struct{})
	go func() {
		// Get will fail, but reconnect will trigger
		gresp, gerr := kv.Get(ctx, "foo")
		if gerr != nil {
			t.Fatal(gerr)
		}
		if len(gresp.Kvs) != 0 {
			t.Fatalf("bad get kvs: got %+v, want empty", gresp.Kvs)
		}
		donec <- struct{}{}
	}()

	time.Sleep(100 * time.Millisecond)
	clus.Members[0].Restart(t)

	select {
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for get")
	case <-donec:
	}
}

// TestKVGetCancel tests that a context cancel on a Get terminates as expected.
func TestKVGetCancel(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	oldconn := clus.Client(0).ActiveConnection()
	kv := clientv3.NewKV(clus.Client(0))

	ctx, cancel := context.WithCancel(context.TODO())
	cancel()

	resp, err := kv.Get(ctx, "abc")
	if err == nil {
		t.Fatalf("cancel on get response %v, expected context error", resp)
	}
	newconn := clus.Client(0).ActiveConnection()
	if oldconn != newconn {
		t.Fatalf("cancel on get broke client connection")
	}
}
