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
	"time"

	"go.etcd.io/etcd/clientv3"
	pb "go.etcd.io/etcd/etcdserver/etcdserverpb"
	"go.etcd.io/etcd/mvcc"
	"go.etcd.io/etcd/mvcc/mvccpb"
)

type watchRange struct {
	key, end string
}

func (wr *watchRange) valid() bool {
	return len(wr.end) == 0 || wr.end > wr.key || (wr.end[0] == 0 && len(wr.end) == 1)
}

type watcher struct {
	// user configuration

	wr       watchRange
	filters  []mvcc.FilterFunc
	progress bool
	prevKV   bool

	// id is the id returned to the client on its watch stream.
	id int64
	// nextrev is the minimum expected next event revision.
	nextrev int64
	// lastHeader has the last header sent over the stream.
	lastHeader pb.ResponseHeader

	// wps is the parent.
	wps *watchProxyStream
}

// send filters out repeated events by discarding revisions older
// than the last one sent over the watch channel.
func (w *watcher) send(wr clientv3.WatchResponse) {
	if wr.IsProgressNotify() && !w.progress {
		return
	}
	if w.nextrev > wr.Header.Revision && len(wr.Events) > 0 {
		return
	}
	if w.nextrev == 0 {
		// current watch; expect updates following this revision
		w.nextrev = wr.Header.Revision + 1
	}

	events := make([]*mvccpb.Event, 0, len(wr.Events))

	var lastRev int64
	for i := range wr.Events {
		ev := (*mvccpb.Event)(wr.Events[i])
		if ev.Kv.ModRevision < w.nextrev {
			continue
		} else {
			// We cannot update w.rev here.
			// txn can have multiple events with the same rev.
			// If w.nextrev updates here, it would skip events in the same txn.
			lastRev = ev.Kv.ModRevision
		}

		filtered := false
		for _, filter := range w.filters {
			if filter(*ev) {
				filtered = true
				break
			}
		}
		if filtered {
			continue
		}

		if !w.prevKV {
			evCopy := *ev
			evCopy.PrevKv = nil
			ev = &evCopy
		}
		events = append(events, ev)
	}

	if lastRev >= w.nextrev {
		w.nextrev = lastRev + 1
	}

	// all events are filtered out?
	if !wr.IsProgressNotify() && !wr.Created && len(events) == 0 && wr.CompactRevision == 0 {
		return
	}

	w.lastHeader = wr.Header
	w.post(&pb.WatchResponse{
		Header:          &wr.Header,
		Created:         wr.Created,
		CompactRevision: wr.CompactRevision,
		Canceled:        wr.Canceled,
		WatchId:         w.id,
		Events:          events,
	})
}

// post puts a watch response on the watcher's proxy stream channel
func (w *watcher) post(wr *pb.WatchResponse) bool {
	select {
	case w.wps.watchCh <- wr:
	case <-time.After(50 * time.Millisecond):
		w.wps.cancel()
		return false
	}
	return true
}
