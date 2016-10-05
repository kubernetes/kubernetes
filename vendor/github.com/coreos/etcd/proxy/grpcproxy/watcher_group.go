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
)

type watcherGroup struct {
	// ch delievers events received from the etcd server
	ch clientv3.WatchChan
	// cancel is used to cancel the underlying etcd server watcher
	// It should also close the ch.
	cancel context.CancelFunc

	mu        sync.Mutex
	rev       int64 // current revision of the watchergroup
	receivers map[receiverID]watcher

	donec chan struct{}
}

type receiverID struct {
	streamID, watcherID int64
}

func newWatchergroup(wch clientv3.WatchChan, c context.CancelFunc) *watcherGroup {
	return &watcherGroup{
		ch:     wch,
		cancel: c,

		receivers: make(map[receiverID]watcher),
		donec:     make(chan struct{}),
	}
}

func (wg *watcherGroup) run() {
	defer close(wg.donec)
	for wr := range wg.ch {
		wg.broadcast(wr)
	}
}

func (wg *watcherGroup) broadcast(wr clientv3.WatchResponse) {
	wg.mu.Lock()
	defer wg.mu.Unlock()

	wg.rev = wr.Header.Revision
	for _, r := range wg.receivers {
		r.send(wr)
	}
}

// add adds the watcher into the group with given ID.
// The current revision of the watcherGroup is returned.
func (wg *watcherGroup) add(rid receiverID, w watcher) int64 {
	wg.mu.Lock()
	defer wg.mu.Unlock()
	wg.receivers[rid] = w
	return wg.rev
}

func (wg *watcherGroup) delete(rid receiverID) {
	wg.mu.Lock()
	defer wg.mu.Unlock()

	delete(wg.receivers, rid)
}

func (wg *watcherGroup) isEmpty() bool {
	wg.mu.Lock()
	defer wg.mu.Unlock()

	return len(wg.receivers) == 0
}

func (wg *watcherGroup) stop() {
	wg.cancel()
	<-wg.donec
}

func (wg *watcherGroup) revision() int64 {
	wg.mu.Lock()
	defer wg.mu.Unlock()
	return wg.rev
}
