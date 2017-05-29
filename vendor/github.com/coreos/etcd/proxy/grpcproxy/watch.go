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

package grpcproxy

import (
	"sync"

	"golang.org/x/net/context"
	"golang.org/x/time/rate"
	"google.golang.org/grpc/metadata"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type watchProxy struct {
	cw  clientv3.Watcher
	ctx context.Context

	ranges *watchRanges

	// retryLimiter controls the create watch retry rate on lost leaders.
	retryLimiter *rate.Limiter

	// mu protects leaderc updates.
	mu      sync.RWMutex
	leaderc chan struct{}

	// wg waits until all outstanding watch servers quit.
	wg sync.WaitGroup
}

const (
	lostLeaderKey  = "__lostleader" // watched to detect leader loss
	retryPerSecond = 10
)

func NewWatchProxy(c *clientv3.Client) (pb.WatchServer, <-chan struct{}) {
	wp := &watchProxy{
		cw:           c.Watcher,
		ctx:          clientv3.WithRequireLeader(c.Ctx()),
		retryLimiter: rate.NewLimiter(rate.Limit(retryPerSecond), retryPerSecond),
		leaderc:      make(chan struct{}),
	}
	wp.ranges = newWatchRanges(wp)
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		// a new streams without opening any watchers won't catch
		// a lost leader event, so have a special watch to monitor it
		rev := int64((uint64(1) << 63) - 2)
		for wp.ctx.Err() == nil {
			wch := wp.cw.Watch(wp.ctx, lostLeaderKey, clientv3.WithRev(rev))
			for range wch {
			}
			wp.mu.Lock()
			close(wp.leaderc)
			wp.leaderc = make(chan struct{})
			wp.mu.Unlock()
			wp.retryLimiter.Wait(wp.ctx)
		}
		wp.mu.Lock()
		<-wp.ctx.Done()
		wp.mu.Unlock()
		wp.wg.Wait()
		wp.ranges.stop()
	}()
	return wp, ch
}

func (wp *watchProxy) Watch(stream pb.Watch_WatchServer) (err error) {
	wp.mu.Lock()
	select {
	case <-wp.ctx.Done():
		wp.mu.Unlock()
		return
	default:
		wp.wg.Add(1)
	}
	wp.mu.Unlock()

	ctx, cancel := context.WithCancel(stream.Context())
	wps := &watchProxyStream{
		ranges:   wp.ranges,
		watchers: make(map[int64]*watcher),
		stream:   stream,
		watchCh:  make(chan *pb.WatchResponse, 1024),
		ctx:      ctx,
		cancel:   cancel,
	}

	var leaderc <-chan struct{}
	if md, ok := metadata.FromContext(stream.Context()); ok {
		v := md[rpctypes.MetadataRequireLeaderKey]
		if len(v) > 0 && v[0] == rpctypes.MetadataHasLeader {
			leaderc = wp.lostLeaderNotify()
		}
	}

	// post to stopc => terminate server stream; can't use a waitgroup
	// since all goroutines will only terminate after Watch() exits.
	stopc := make(chan struct{}, 3)
	go func() {
		defer func() { stopc <- struct{}{} }()
		wps.recvLoop()
	}()
	go func() {
		defer func() { stopc <- struct{}{} }()
		wps.sendLoop()
	}()
	// tear down watch if leader goes down or entire watch proxy is terminated
	go func() {
		defer func() { stopc <- struct{}{} }()
		select {
		case <-leaderc:
		case <-ctx.Done():
		case <-wp.ctx.Done():
		}
	}()

	<-stopc
	cancel()

	// recv/send may only shutdown after function exits;
	// goroutine notifies proxy that stream is through
	go func() {
		<-stopc
		<-stopc
		wps.close()
		wp.wg.Done()
	}()

	select {
	case <-leaderc:
		return rpctypes.ErrNoLeader
	default:
		return wps.ctx.Err()
	}
}

func (wp *watchProxy) lostLeaderNotify() <-chan struct{} {
	wp.mu.RLock()
	defer wp.mu.RUnlock()
	return wp.leaderc
}

// watchProxyStream forwards etcd watch events to a proxied client stream.
type watchProxyStream struct {
	ranges *watchRanges

	// mu protects watchers and nextWatcherID
	mu sync.Mutex
	// watchers receive events from watch broadcast.
	watchers map[int64]*watcher
	// nextWatcherID is the id to assign the next watcher on this stream.
	nextWatcherID int64

	stream pb.Watch_WatchServer

	// watchCh receives watch responses from the watchers.
	watchCh chan *pb.WatchResponse

	ctx    context.Context
	cancel context.CancelFunc
}

func (wps *watchProxyStream) close() {
	var wg sync.WaitGroup
	wps.cancel()
	wps.mu.Lock()
	wg.Add(len(wps.watchers))
	for _, wpsw := range wps.watchers {
		go func(w *watcher) {
			wps.ranges.delete(w)
			wg.Done()
		}(wpsw)
	}
	wps.watchers = nil
	wps.mu.Unlock()

	wg.Wait()

	close(wps.watchCh)
}

func (wps *watchProxyStream) recvLoop() error {
	for {
		req, err := wps.stream.Recv()
		if err != nil {
			return err
		}
		switch uv := req.RequestUnion.(type) {
		case *pb.WatchRequest_CreateRequest:
			cr := uv.CreateRequest
			w := &watcher{
				wr:  watchRange{string(cr.Key), string(cr.RangeEnd)},
				id:  wps.nextWatcherID,
				wps: wps,

				nextrev:  cr.StartRevision,
				progress: cr.ProgressNotify,
				prevKV:   cr.PrevKv,
				filters:  v3rpc.FiltersFromRequest(cr),
			}
			if !w.wr.valid() {
				w.post(&pb.WatchResponse{WatchId: -1, Created: true, Canceled: true})
				continue
			}
			wps.nextWatcherID++
			w.nextrev = cr.StartRevision
			wps.watchers[w.id] = w
			wps.ranges.add(w)
		case *pb.WatchRequest_CancelRequest:
			wps.delete(uv.CancelRequest.WatchId)
		default:
			panic("not implemented")
		}
	}
}

func (wps *watchProxyStream) sendLoop() {
	for {
		select {
		case wresp, ok := <-wps.watchCh:
			if !ok {
				return
			}
			if err := wps.stream.Send(wresp); err != nil {
				return
			}
		case <-wps.ctx.Done():
			return
		}
	}
}

func (wps *watchProxyStream) delete(id int64) {
	wps.mu.Lock()
	defer wps.mu.Unlock()

	w, ok := wps.watchers[id]
	if !ok {
		return
	}
	wps.ranges.delete(w)
	delete(wps.watchers, id)
	resp := &pb.WatchResponse{
		Header:   &w.lastHeader,
		WatchId:  id,
		Canceled: true,
	}
	wps.watchCh <- resp
}
