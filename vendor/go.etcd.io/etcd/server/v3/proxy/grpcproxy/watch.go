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
	"context"
	"sync"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3rpc"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type watchProxy struct {
	cw  clientv3.Watcher
	ctx context.Context

	leader *leader

	ranges *watchRanges

	// mu protects adding outstanding watch servers through wg.
	mu sync.Mutex

	// wg waits until all outstanding watch servers quit.
	wg sync.WaitGroup

	// kv is used for permission checking
	kv clientv3.KV
	lg *zap.Logger
}

func NewWatchProxy(ctx context.Context, lg *zap.Logger, c *clientv3.Client) (pb.WatchServer, <-chan struct{}) {
	cctx, cancel := context.WithCancel(ctx)
	wp := &watchProxy{
		cw:     c.Watcher,
		ctx:    cctx,
		leader: newLeader(cctx, c.Watcher),

		kv: c.KV, // for permission checking
		lg: lg,
	}
	wp.ranges = newWatchRanges(wp)
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		<-wp.leader.stopNotify()
		wp.mu.Lock()
		select {
		case <-wp.ctx.Done():
		case <-wp.leader.disconnectNotify():
			cancel()
		}
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
		select {
		case <-wp.leader.disconnectNotify():
			return status.Error(codes.Canceled, "the client connection is closing")
		default:
			return wp.ctx.Err()
		}
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
		kv:       wp.kv,
		lg:       wp.lg,
	}

	var lostLeaderC <-chan struct{}
	if md, ok := metadata.FromOutgoingContext(stream.Context()); ok {
		v := md[rpctypes.MetadataRequireLeaderKey]
		if len(v) > 0 && v[0] == rpctypes.MetadataHasLeader {
			lostLeaderC = wp.leader.lostNotify()
			// if leader is known to be lost at creation time, avoid
			// letting events through at all
			select {
			case <-lostLeaderC:
				wp.wg.Done()
				return rpctypes.ErrNoLeader
			default:
			}
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
		case <-lostLeaderC:
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
	case <-lostLeaderC:
		return rpctypes.ErrNoLeader
	case <-wp.leader.disconnectNotify():
		return status.Error(codes.Canceled, "the client connection is closing")
	default:
		return wps.ctx.Err()
	}
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

	// kv is used for permission checking
	kv clientv3.KV
	lg *zap.Logger
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

func (wps *watchProxyStream) checkPermissionForWatch(key, rangeEnd []byte) error {
	if len(key) == 0 {
		// If the length of the key is 0, we need to obtain full range.
		// look at clientv3.WithPrefix()
		key = []byte{0}
		rangeEnd = []byte{0}
	}
	req := &pb.RangeRequest{
		Serializable: true,
		Key:          key,
		RangeEnd:     rangeEnd,
		CountOnly:    true,
		Limit:        1,
	}
	_, err := wps.kv.Do(wps.ctx, RangeRequestToOp(req))
	return err
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

			if err := wps.checkPermissionForWatch(cr.Key, cr.RangeEnd); err != nil {
				wps.watchCh <- &pb.WatchResponse{
					Header:       &pb.ResponseHeader{},
					WatchId:      -1,
					Created:      true,
					Canceled:     true,
					CancelReason: err.Error(),
				}
				continue
			}

			wps.mu.Lock()
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
				wps.mu.Unlock()
				continue
			}
			wps.nextWatcherID++
			w.nextrev = cr.StartRevision
			wps.watchers[w.id] = w
			wps.ranges.add(w)
			wps.mu.Unlock()
			wps.lg.Debug("create watcher", zap.String("key", w.wr.key), zap.String("end", w.wr.end), zap.Int64("watcherId", wps.nextWatcherID))
		case *pb.WatchRequest_CancelRequest:
			wps.delete(uv.CancelRequest.WatchId)
			wps.lg.Debug("cancel watcher", zap.Int64("watcherId", uv.CancelRequest.WatchId))
		default:
			// Panic or Fatalf would allow to network clients to crash the serve remotely.
			wps.lg.Error("not supported request type by gRPC proxy", zap.Stringer("request", req))
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
