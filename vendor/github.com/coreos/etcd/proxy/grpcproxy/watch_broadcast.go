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

	"github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

// watchBroadcast broadcasts a server watcher to many client watchers.
type watchBroadcast struct {
	// cancel stops the underlying etcd server watcher and closes ch.
	cancel context.CancelFunc
	donec  chan struct{}

	// mu protects rev and receivers.
	mu sync.RWMutex
	// nextrev is the minimum expected next revision of the watcher on ch.
	nextrev int64
	// receivers contains all the client-side watchers to serve.
	receivers map[*watcher]struct{}
	// responses counts the number of responses
	responses int
}

func newWatchBroadcast(wp *watchProxy, w *watcher, update func(*watchBroadcast)) *watchBroadcast {
	cctx, cancel := context.WithCancel(wp.ctx)
	wb := &watchBroadcast{
		cancel:    cancel,
		nextrev:   w.nextrev,
		receivers: make(map[*watcher]struct{}),
		donec:     make(chan struct{}),
	}
	wb.add(w)
	go func() {
		defer close(wb.donec)

		opts := []clientv3.OpOption{
			clientv3.WithRange(w.wr.end),
			clientv3.WithProgressNotify(),
			clientv3.WithRev(wb.nextrev),
			clientv3.WithPrevKV(),
			clientv3.WithCreatedNotify(),
		}

		wch := wp.cw.Watch(cctx, w.wr.key, opts...)

		for wr := range wch {
			wb.bcast(wr)
			update(wb)
		}
	}()
	return wb
}

func (wb *watchBroadcast) bcast(wr clientv3.WatchResponse) {
	wb.mu.Lock()
	defer wb.mu.Unlock()
	// watchers start on the given revision, if any; ignore header rev on create
	if wb.responses > 0 || wb.nextrev == 0 {
		wb.nextrev = wr.Header.Revision + 1
	}
	wb.responses++
	for r := range wb.receivers {
		r.send(wr)
	}
	if len(wb.receivers) > 0 {
		eventsCoalescing.Add(float64(len(wb.receivers) - 1))
	}
}

// add puts a watcher into receiving a broadcast if its revision at least
// meets the broadcast revision. Returns true if added.
func (wb *watchBroadcast) add(w *watcher) bool {
	wb.mu.Lock()
	defer wb.mu.Unlock()
	if wb.nextrev > w.nextrev || (wb.nextrev == 0 && w.nextrev != 0) {
		// wb is too far ahead, w will miss events
		// or wb is being established with a current watcher
		return false
	}
	if wb.responses == 0 {
		// Newly created; create event will be sent by etcd.
		wb.receivers[w] = struct{}{}
		return true
	}
	// already sent by etcd; emulate create event
	ok := w.post(&pb.WatchResponse{
		Header: &pb.ResponseHeader{
			// todo: fill in ClusterId
			// todo: fill in MemberId:
			Revision: w.nextrev,
			// todo: fill in RaftTerm:
		},
		WatchId: w.id,
		Created: true,
	})
	if !ok {
		return false
	}
	wb.receivers[w] = struct{}{}
	watchersCoalescing.Inc()

	return true
}
func (wb *watchBroadcast) delete(w *watcher) {
	wb.mu.Lock()
	defer wb.mu.Unlock()
	if _, ok := wb.receivers[w]; !ok {
		panic("deleting missing watcher from broadcast")
	}
	delete(wb.receivers, w)
	if len(wb.receivers) > 0 {
		// do not dec the only left watcher for coalescing.
		watchersCoalescing.Dec()
	}
}

func (wb *watchBroadcast) size() int {
	wb.mu.RLock()
	defer wb.mu.RUnlock()
	return len(wb.receivers)
}

func (wb *watchBroadcast) empty() bool { return wb.size() == 0 }

func (wb *watchBroadcast) stop() {
	if !wb.empty() {
		// do not dec the only left watcher for coalescing.
		watchersCoalescing.Sub(float64(wb.size() - 1))
	}

	wb.cancel()
	<-wb.donec
}
