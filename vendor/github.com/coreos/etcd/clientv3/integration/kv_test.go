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
	"bytes"
	"math/rand"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/coreos/etcd/pkg/testutil"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

func TestKVPutError(t *testing.T) {
	defer testutil.AfterTest(t)

	var (
		maxReqBytes = 1.5 * 1024 * 1024 // hard coded max in v3_server.go
		quota       = int64(int(maxReqBytes) + 8*os.Getpagesize())
	)
	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1, QuotaBackendBytes: quota})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	_, err := kv.Put(ctx, "", "bar")
	if err != rpctypes.ErrEmptyKey {
		t.Fatalf("expected %v, got %v", rpctypes.ErrEmptyKey, err)
	}

	_, err = kv.Put(ctx, "key", strings.Repeat("a", int(maxReqBytes+100)))
	if err != rpctypes.ErrRequestTooLarge {
		t.Fatalf("expected %v, got %v", rpctypes.ErrRequestTooLarge, err)
	}

	_, err = kv.Put(ctx, "foo1", strings.Repeat("a", int(maxReqBytes-50)))
	if err != nil { // below quota
		t.Fatal(err)
	}

	time.Sleep(1 * time.Second) // give enough time for commit

	_, err = kv.Put(ctx, "foo2", strings.Repeat("a", int(maxReqBytes-50)))
	if err != rpctypes.ErrNoSpace { // over quota
		t.Fatalf("expected %v, got %v", rpctypes.ErrNoSpace, err)
	}
}

func TestKVPut(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	lapi := clientv3.NewLease(clus.RandClient())
	defer lapi.Close()

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	resp, err := lapi.Grant(context.Background(), 10)
	if err != nil {
		t.Fatalf("failed to create lease %v", err)
	}

	tests := []struct {
		key, val string
		leaseID  clientv3.LeaseID
	}{
		{"foo", "bar", clientv3.NoLease},
		{"hello", "world", resp.ID},
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

func TestKVPutWithRequireLeader(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	clus.Members[1].Stop(t)
	clus.Members[2].Stop(t)

	// wait for election timeout, then member[0] will not have a leader.
	var (
		electionTicks = 10
		tickDuration  = 10 * time.Millisecond
	)
	time.Sleep(time.Duration(3*electionTicks) * tickDuration)

	kv := clientv3.NewKV(clus.Client(0))
	_, err := kv.Put(clientv3.WithRequireLeader(context.Background()), "foo", "bar")
	if err != rpctypes.ErrNoLeader {
		t.Fatal(err)
	}

	// clients may give timeout errors since the members are stopped; take
	// the clients so that terminating the cluster won't complain
	clus.Client(1).Close()
	clus.Client(2).Close()
	clus.TakeClient(1)
	clus.TakeClient(2)
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

		wantSet []*mvccpb.KeyValue
	}{
		// range first two
		{
			"a", "c",
			0,
			nil,

			[]*mvccpb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
			},
		},
		// range first two with serializable
		{
			"a", "c",
			0,
			[]clientv3.OpOption{clientv3.WithSerializable()},

			[]*mvccpb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
			},
		},
		// range all with rev
		{
			"a", "x",
			2,
			nil,

			[]*mvccpb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
			},
		},
		// range all with countOnly
		{
			"a", "x",
			2,
			[]clientv3.OpOption{clientv3.WithCountOnly()},

			nil,
		},
		// range all with SortByKey, SortAscend
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByKey, clientv3.SortAscend)},

			[]*mvccpb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
			},
		},
		// range all with SortByKey, missing sorting order (ASCEND by default)
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByKey, clientv3.SortNone)},

			[]*mvccpb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
			},
		},
		// range all with SortByCreateRevision, SortDescend
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByCreateRevision, clientv3.SortDescend)},

			[]*mvccpb.KeyValue{
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
			},
		},
		// range all with SortByCreateRevision, missing sorting order (ASCEND by default)
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByCreateRevision, clientv3.SortNone)},

			[]*mvccpb.KeyValue{
				{Key: []byte("a"), Value: nil, CreateRevision: 2, ModRevision: 2, Version: 1},
				{Key: []byte("b"), Value: nil, CreateRevision: 3, ModRevision: 3, Version: 1},
				{Key: []byte("c"), Value: nil, CreateRevision: 4, ModRevision: 6, Version: 3},
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
				{Key: []byte("fop"), Value: nil, CreateRevision: 9, ModRevision: 9, Version: 1},
			},
		},
		// range all with SortByModRevision, SortDescend
		{
			"a", "x",
			0,
			[]clientv3.OpOption{clientv3.WithSort(clientv3.SortByModRevision, clientv3.SortDescend)},

			[]*mvccpb.KeyValue{
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

			[]*mvccpb.KeyValue{
				{Key: []byte("foo"), Value: nil, CreateRevision: 7, ModRevision: 7, Version: 1},
				{Key: []byte("foo/abc"), Value: nil, CreateRevision: 8, ModRevision: 8, Version: 1},
			},
		},
		// WithFromKey
		{
			"fo", "",
			0,
			[]clientv3.OpOption{clientv3.WithFromKey()},

			[]*mvccpb.KeyValue{
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

func TestKVGetErrConnClosed(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	kv := clientv3.NewKV(cli)

	donec := make(chan struct{})
	go func() {
		defer close(donec)
		_, err := kv.Get(context.TODO(), "foo")
		if err != nil && err != grpc.ErrClientConnClosing {
			t.Fatalf("expected %v, got %v", grpc.ErrClientConnClosing, err)
		}
	}()

	if err := cli.Close(); err != nil {
		t.Fatal(err)
	}
	clus.TakeClient(0)

	select {
	case <-time.After(3 * time.Second):
		t.Fatal("kv.Get took too long")
	case <-donec:
	}
}

func TestKVNewAfterClose(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	clus.TakeClient(0)
	if err := cli.Close(); err != nil {
		t.Fatal(err)
	}

	donec := make(chan struct{})
	go func() {
		kv := clientv3.NewKV(cli)
		if _, err := kv.Get(context.TODO(), "foo"); err != grpc.ErrClientConnClosing {
			t.Fatalf("expected %v, got %v", grpc.ErrClientConnClosing, err)
		}
		close(donec)
	}()
	select {
	case <-time.After(3 * time.Second):
		t.Fatal("kv.Get took too long")
	case <-donec:
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

func TestKVCompactError(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	kv := clientv3.NewKV(clus.RandClient())
	ctx := context.TODO()

	for i := 0; i < 5; i++ {
		if _, err := kv.Put(ctx, "foo", "bar"); err != nil {
			t.Fatalf("couldn't put 'foo' (%v)", err)
		}
	}
	_, err := kv.Compact(ctx, 6)
	if err != nil {
		t.Fatalf("couldn't compact 6 (%v)", err)
	}

	_, err = kv.Compact(ctx, 6)
	if err != rpctypes.ErrCompacted {
		t.Fatalf("expected %v, got %v", rpctypes.ErrCompacted, err)
	}

	_, err = kv.Compact(ctx, 100)
	if err != rpctypes.ErrFutureRev {
		t.Fatalf("expected %v, got %v", rpctypes.ErrFutureRev, err)
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

	_, err := kv.Compact(ctx, 7)
	if err != nil {
		t.Fatalf("couldn't compact kv space (%v)", err)
	}
	_, err = kv.Compact(ctx, 7)
	if err == nil || err != rpctypes.ErrCompacted {
		t.Fatalf("error got %v, want %v", err, rpctypes.ErrCompacted)
	}

	wcli := clus.RandClient()
	// new watcher could precede receiving the compaction without quorum first
	wcli.Get(ctx, "quorum-get")

	wc := clientv3.NewWatcher(wcli)
	defer wc.Close()
	wchan := wc.Watch(ctx, "foo", clientv3.WithRev(3))

	if wr := <-wchan; wr.CompactRevision != 7 {
		t.Fatalf("wchan CompactRevision got %v, want 7", wr.CompactRevision)
	}
	if wr, ok := <-wchan; ok {
		t.Fatalf("wchan got %v, expected closed", wr)
	}

	_, err = kv.Compact(ctx, 1000)
	if err == nil || err != rpctypes.ErrFutureRev {
		t.Fatalf("error got %v, want %v", err, rpctypes.ErrFutureRev)
	}
}

// TestKVGetRetry ensures get will retry on disconnect.
func TestKVGetRetry(t *testing.T) {
	defer testutil.AfterTest(t)

	clusterSize := 3
	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: clusterSize})
	defer clus.Terminate(t)

	// because killing leader and following election
	// could give no other endpoints for client reconnection
	fIdx := (clus.WaitLeader(t) + 1) % clusterSize

	kv := clientv3.NewKV(clus.Client(fIdx))
	ctx := context.TODO()

	if _, err := kv.Put(ctx, "foo", "bar"); err != nil {
		t.Fatal(err)
	}

	clus.Members[fIdx].Stop(t)

	donec := make(chan struct{})
	go func() {
		// Get will fail, but reconnect will trigger
		gresp, gerr := kv.Get(ctx, "foo")
		if gerr != nil {
			t.Fatal(gerr)
		}
		wkvs := []*mvccpb.KeyValue{
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
	clus.Members[fIdx].Restart(t)

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
	clus.Members[0].Stop(t)

	ctx, cancel := context.WithTimeout(context.TODO(), time.Second)
	defer cancel()
	_, err := kv.Put(ctx, "foo", "bar")
	if err == nil {
		t.Fatalf("got success on disconnected put, wanted error")
	}

	donec := make(chan struct{})
	go func() {
		// Get will fail, but reconnect will trigger
		gresp, gerr := kv.Get(context.TODO(), "foo")
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

// TestKVGetStoppedServerAndClose ensures closing after a failed Get works.
func TestKVGetStoppedServerAndClose(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	clus.Members[0].Stop(t)
	ctx, cancel := context.WithTimeout(context.TODO(), time.Second)
	// this Get fails and triggers an asynchronous connection retry
	_, err := cli.Get(ctx, "abc")
	cancel()
	if !strings.Contains(err.Error(), "context deadline") {
		t.Fatal(err)
	}
}

// TestKVPutStoppedServerAndClose ensures closing after a failed Put works.
func TestKVPutStoppedServerAndClose(t *testing.T) {
	defer testutil.AfterTest(t)

	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer clus.Terminate(t)

	cli := clus.Client(0)
	clus.Members[0].Stop(t)

	ctx, cancel := context.WithTimeout(context.TODO(), time.Second)
	// get retries on all errors.
	// so here we use it to eat the potential broken pipe error for the next put.
	// grpc client might see a broken pipe error when we issue the get request before
	// grpc finds out the original connection is down due to the member shutdown.
	_, err := cli.Get(ctx, "abc")
	cancel()
	if !strings.Contains(err.Error(), "context deadline") {
		t.Fatal(err)
	}

	// this Put fails and triggers an asynchronous connection retry
	_, err = cli.Put(ctx, "abc", "123")
	cancel()
	if !strings.Contains(err.Error(), "context deadline") {
		t.Fatal(err)
	}
}

// TestKVGetOneEndpointDown ensures a client can connect and get if one endpoint is down
func TestKVPutOneEndpointDown(t *testing.T) {
	defer testutil.AfterTest(t)
	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 3})
	defer clus.Terminate(t)

	// get endpoint list
	eps := make([]string, 3)
	for i := range eps {
		eps[i] = clus.Members[i].GRPCAddr()
	}

	// make a dead node
	clus.Members[rand.Intn(len(eps))].Stop(t)

	// try to connect with dead node in the endpoint list
	cfg := clientv3.Config{Endpoints: eps, DialTimeout: 1 * time.Second}
	cli, err := clientv3.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cli.Close()
	ctx, cancel := context.WithTimeout(context.TODO(), 3*time.Second)
	if _, err := cli.Get(ctx, "abc", clientv3.WithSerializable()); err != nil {
		t.Fatal(err)
	}
	cancel()
}

// TestKVGetResetLoneEndpoint ensures that if an endpoint resets and all other
// endpoints are down, then it will reconnect.
func TestKVGetResetLoneEndpoint(t *testing.T) {
	defer testutil.AfterTest(t)
	clus := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 2})
	defer clus.Terminate(t)

	// get endpoint list
	eps := make([]string, 2)
	for i := range eps {
		eps[i] = clus.Members[i].GRPCAddr()
	}

	cfg := clientv3.Config{Endpoints: eps, DialTimeout: 500 * time.Millisecond}
	cli, err := clientv3.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer cli.Close()

	// disconnect everything
	clus.Members[0].Stop(t)
	clus.Members[1].Stop(t)

	// have Get try to reconnect
	donec := make(chan struct{})
	go func() {
		ctx, cancel := context.WithTimeout(context.TODO(), 5*time.Second)
		if _, err := cli.Get(ctx, "abc", clientv3.WithSerializable()); err != nil {
			t.Fatal(err)
		}
		cancel()
		close(donec)
	}()
	time.Sleep(500 * time.Millisecond)
	clus.Members[0].Restart(t)
	select {
	case <-time.After(10 * time.Second):
		t.Fatalf("timed out waiting for Get")
	case <-donec:
	}
}
