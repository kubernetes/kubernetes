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

package main

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/context" // grpc does a comparison on context.Cancel; can't use "context" package
	"golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/transport"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type keyStresser struct {
	Endpoint string

	keyLargeSize   int
	keySize        int
	keySuffixRange int

	N int

	rateLimiter *rate.Limiter

	wg sync.WaitGroup

	cancel func()
	conn   *grpc.ClientConn
	// atomicModifiedKeys records the number of keys created and deleted by the stresser.
	atomicModifiedKeys int64

	stressTable *stressTable
}

func (s *keyStresser) Stress() error {
	// TODO: add backoff option
	conn, err := grpc.Dial(s.Endpoint, grpc.WithInsecure())
	if err != nil {
		return fmt.Errorf("%v (%s)", err, s.Endpoint)
	}
	ctx, cancel := context.WithCancel(context.Background())

	s.wg.Add(s.N)
	s.conn = conn
	s.cancel = cancel

	kvc := pb.NewKVClient(conn)

	var stressEntries = []stressEntry{
		{weight: 0.7, f: newStressPut(kvc, s.keySuffixRange, s.keySize)},
		{
			weight: 0.7 * float32(s.keySize) / float32(s.keyLargeSize),
			f:      newStressPut(kvc, s.keySuffixRange, s.keyLargeSize),
		},
		{weight: 0.07, f: newStressRange(kvc, s.keySuffixRange)},
		{weight: 0.07, f: newStressRangeInterval(kvc, s.keySuffixRange)},
		{weight: 0.07, f: newStressDelete(kvc, s.keySuffixRange)},
		{weight: 0.07, f: newStressDeleteInterval(kvc, s.keySuffixRange)},
	}
	s.stressTable = createStressTable(stressEntries)

	for i := 0; i < s.N; i++ {
		go s.run(ctx)
	}

	plog.Infof("keyStresser %q is started", s.Endpoint)
	return nil
}

func (s *keyStresser) run(ctx context.Context) {
	defer s.wg.Done()

	for {
		if err := s.rateLimiter.Wait(ctx); err == context.Canceled {
			return
		}

		// TODO: 10-second is enough timeout to cover leader failure
		// and immediate leader election. Find out what other cases this
		// could be timed out.
		sctx, scancel := context.WithTimeout(ctx, 10*time.Second)
		err, modifiedKeys := s.stressTable.choose()(sctx)
		scancel()
		if err == nil {
			atomic.AddInt64(&s.atomicModifiedKeys, modifiedKeys)
			continue
		}

		switch grpc.ErrorDesc(err) {
		case context.DeadlineExceeded.Error():
			// This retries when request is triggered at the same time as
			// leader failure. When we terminate the leader, the request to
			// that leader cannot be processed, and times out. Also requests
			// to followers cannot be forwarded to the old leader, so timing out
			// as well. We want to keep stressing until the cluster elects a
			// new leader and start processing requests again.
		case etcdserver.ErrTimeoutDueToLeaderFail.Error(), etcdserver.ErrTimeout.Error():
			// This retries when request is triggered at the same time as
			// leader failure and follower nodes receive time out errors
			// from losing their leader. Followers should retry to connect
			// to the new leader.
		case etcdserver.ErrStopped.Error():
			// one of the etcd nodes stopped from failure injection
		case transport.ErrConnClosing.Desc:
			// server closed the transport (failure injected node)
		case rpctypes.ErrNotCapable.Error():
			// capability check has not been done (in the beginning)
		case rpctypes.ErrTooManyRequests.Error():
			// hitting the recovering member.
		case context.Canceled.Error():
			// from stresser.Cancel method:
			return
		case grpc.ErrClientConnClosing.Error():
			// from stresser.Cancel method:
			return
		default:
			plog.Errorf("keyStresser %v exited with error (%v)", s.Endpoint, err)
			return
		}
	}
}

func (s *keyStresser) Cancel() {
	s.cancel()
	s.conn.Close()
	s.wg.Wait()
	plog.Infof("keyStresser %q is canceled", s.Endpoint)
}

func (s *keyStresser) ModifiedKeys() int64 {
	return atomic.LoadInt64(&s.atomicModifiedKeys)
}

func (s *keyStresser) Checker() Checker { return nil }

type stressFunc func(ctx context.Context) (err error, modifiedKeys int64)

type stressEntry struct {
	weight float32
	f      stressFunc
}

type stressTable struct {
	entries    []stressEntry
	sumWeights float32
}

func createStressTable(entries []stressEntry) *stressTable {
	st := stressTable{entries: entries}
	for _, entry := range st.entries {
		st.sumWeights += entry.weight
	}
	return &st
}

func (st *stressTable) choose() stressFunc {
	v := rand.Float32() * st.sumWeights
	var sum float32
	var idx int
	for i := range st.entries {
		sum += st.entries[i].weight
		if sum >= v {
			idx = i
			break
		}
	}
	return st.entries[idx].f
}

func newStressPut(kvc pb.KVClient, keySuffixRange, keySize int) stressFunc {
	return func(ctx context.Context) (error, int64) {
		_, err := kvc.Put(ctx, &pb.PutRequest{
			Key:   []byte(fmt.Sprintf("foo%016x", rand.Intn(keySuffixRange))),
			Value: randBytes(keySize),
		}, grpc.FailFast(false))
		return err, 1
	}
}

func newStressRange(kvc pb.KVClient, keySuffixRange int) stressFunc {
	return func(ctx context.Context) (error, int64) {
		_, err := kvc.Range(ctx, &pb.RangeRequest{
			Key: []byte(fmt.Sprintf("foo%016x", rand.Intn(keySuffixRange))),
		}, grpc.FailFast(false))
		return err, 0
	}
}

func newStressRangeInterval(kvc pb.KVClient, keySuffixRange int) stressFunc {
	return func(ctx context.Context) (error, int64) {
		start := rand.Intn(keySuffixRange)
		end := start + 500
		_, err := kvc.Range(ctx, &pb.RangeRequest{
			Key:      []byte(fmt.Sprintf("foo%016x", start)),
			RangeEnd: []byte(fmt.Sprintf("foo%016x", end)),
		}, grpc.FailFast(false))
		return err, 0
	}
}

func newStressDelete(kvc pb.KVClient, keySuffixRange int) stressFunc {
	return func(ctx context.Context) (error, int64) {
		_, err := kvc.DeleteRange(ctx, &pb.DeleteRangeRequest{
			Key: []byte(fmt.Sprintf("foo%016x", rand.Intn(keySuffixRange))),
		}, grpc.FailFast(false))
		return err, 1
	}
}

func newStressDeleteInterval(kvc pb.KVClient, keySuffixRange int) stressFunc {
	return func(ctx context.Context) (error, int64) {
		start := rand.Intn(keySuffixRange)
		end := start + 500
		resp, err := kvc.DeleteRange(ctx, &pb.DeleteRangeRequest{
			Key:      []byte(fmt.Sprintf("foo%016x", start)),
			RangeEnd: []byte(fmt.Sprintf("foo%016x", end)),
		}, grpc.FailFast(false))
		if err == nil {
			return nil, resp.Deleted
		}
		return err, 0
	}
}
