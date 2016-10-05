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
	"io"
	"sync"

	"golang.org/x/net/context"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type watchProxy struct {
	cw  clientv3.Watcher
	wgs watchergroups

	mu           sync.Mutex
	nextStreamID int64

	ctx context.Context
}

func NewWatchProxy(c *clientv3.Client) pb.WatchServer {
	wp := &watchProxy{
		cw: c.Watcher,
		wgs: watchergroups{
			cw:        c.Watcher,
			groups:    make(map[watchRange]*watcherGroup),
			idToGroup: make(map[receiverID]*watcherGroup),
			proxyCtx:  c.Ctx(),
		},
		ctx: c.Ctx(),
	}
	go func() {
		<-wp.ctx.Done()
		wp.wgs.stop()
	}()
	return wp
}

func (wp *watchProxy) Watch(stream pb.Watch_WatchServer) (err error) {
	wp.mu.Lock()
	wp.nextStreamID++
	sid := wp.nextStreamID
	wp.mu.Unlock()

	sws := serverWatchStream{
		cw:       wp.cw,
		groups:   &wp.wgs,
		singles:  make(map[int64]*watcherSingle),
		inGroups: make(map[int64]struct{}),

		id:         sid,
		gRPCStream: stream,

		watchCh: make(chan *pb.WatchResponse, 1024),

		proxyCtx: wp.ctx,
	}

	go sws.recvLoop()
	sws.sendLoop()
	return wp.ctx.Err()
}

type serverWatchStream struct {
	id int64
	cw clientv3.Watcher

	mu       sync.Mutex // make sure any access of groups and singles is atomic
	groups   *watchergroups
	singles  map[int64]*watcherSingle
	inGroups map[int64]struct{}

	gRPCStream pb.Watch_WatchServer

	watchCh chan *pb.WatchResponse

	nextWatcherID int64

	proxyCtx context.Context
}

func (sws *serverWatchStream) close() {
	var wg sync.WaitGroup
	sws.mu.Lock()
	wg.Add(len(sws.singles) + len(sws.inGroups))
	for _, ws := range sws.singles {
		// copy the range variable to avoid race
		copyws := ws
		go func() {
			copyws.stop()
			wg.Done()
		}()
	}
	for id := range sws.inGroups {
		// copy the range variable to avoid race
		wid := id
		go func() {
			sws.groups.removeWatcher(receiverID{streamID: sws.id, watcherID: wid})
			wg.Done()
		}()
	}
	sws.inGroups = nil
	sws.mu.Unlock()

	wg.Wait()

	close(sws.watchCh)
}

func (sws *serverWatchStream) recvLoop() error {
	defer sws.close()

	for {
		req, err := sws.gRPCStream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		switch uv := req.RequestUnion.(type) {
		case *pb.WatchRequest_CreateRequest:
			cr := uv.CreateRequest

			watcher := watcher{
				wr: watchRange{
					key: string(cr.Key),
					end: string(cr.RangeEnd),
				},
				id: sws.nextWatcherID,
				ch: sws.watchCh,

				progress: cr.ProgressNotify,
				filters:  v3rpc.FiltersFromRequest(cr),
			}
			if cr.StartRevision != 0 {
				sws.addDedicatedWatcher(watcher, cr.StartRevision)
			} else {
				sws.addCoalescedWatcher(watcher)
			}
			sws.nextWatcherID++

		case *pb.WatchRequest_CancelRequest:
			sws.removeWatcher(uv.CancelRequest.WatchId)
		default:
			panic("not implemented")
		}
	}
}

func (sws *serverWatchStream) sendLoop() {
	for {
		select {
		case wresp, ok := <-sws.watchCh:
			if !ok {
				return
			}
			if err := sws.gRPCStream.Send(wresp); err != nil {
				return
			}
		case <-sws.proxyCtx.Done():
			return
		}
	}
}

func (sws *serverWatchStream) addCoalescedWatcher(w watcher) {
	sws.mu.Lock()
	defer sws.mu.Unlock()

	rid := receiverID{streamID: sws.id, watcherID: w.id}
	sws.groups.addWatcher(rid, w)
	sws.inGroups[w.id] = struct{}{}
}

func (sws *serverWatchStream) addDedicatedWatcher(w watcher, rev int64) {
	sws.mu.Lock()
	defer sws.mu.Unlock()

	ctx, cancel := context.WithCancel(sws.proxyCtx)

	wch := sws.cw.Watch(ctx,
		w.wr.key, clientv3.WithRange(w.wr.end),
		clientv3.WithRev(rev),
		clientv3.WithProgressNotify(),
		clientv3.WithCreatedNotify(),
	)

	ws := newWatcherSingle(wch, cancel, w, sws)
	sws.singles[w.id] = ws
	go ws.run()
}

func (sws *serverWatchStream) maybeCoalesceWatcher(ws watcherSingle) bool {
	sws.mu.Lock()
	defer sws.mu.Unlock()

	rid := receiverID{streamID: sws.id, watcherID: ws.w.id}
	// do not add new watchers when stream is closing
	if sws.inGroups == nil {
		return false
	}
	if sws.groups.maybeJoinWatcherSingle(rid, ws) {
		delete(sws.singles, ws.w.id)
		sws.inGroups[ws.w.id] = struct{}{}
		return true
	}
	return false
}

func (sws *serverWatchStream) removeWatcher(id int64) {
	sws.mu.Lock()
	defer sws.mu.Unlock()

	var (
		rev int64
		ok  bool
	)

	defer func() {
		if !ok {
			return
		}
		resp := &pb.WatchResponse{
			Header: &pb.ResponseHeader{
				// todo: fill in ClusterId
				// todo: fill in MemberId:
				Revision: rev,
				// todo: fill in RaftTerm:
			},
			WatchId:  id,
			Canceled: true,
		}
		sws.watchCh <- resp
	}()

	rev, ok = sws.groups.removeWatcher(receiverID{streamID: sws.id, watcherID: id})
	if ok {
		delete(sws.inGroups, id)
		return
	}

	var ws *watcherSingle
	if ws, ok = sws.singles[id]; ok {
		delete(sws.singles, id)
		ws.stop()
		rev = ws.lastStoreRev
	}
}
