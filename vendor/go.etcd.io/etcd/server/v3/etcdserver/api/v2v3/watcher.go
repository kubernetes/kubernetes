// Copyright 2017 The etcd Authors
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

package v2v3

import (
	"context"
	"strings"

	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2error"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"
)

func (s *v2v3Store) Watch(prefix string, recursive, stream bool, sinceIndex uint64) (v2store.Watcher, error) {
	ctx, cancel := context.WithCancel(s.ctx)
	wch := s.c.Watch(
		ctx,
		// TODO: very pricey; use a single store-wide watch in future
		s.pfx,
		clientv3.WithPrefix(),
		clientv3.WithRev(int64(sinceIndex)),
		clientv3.WithCreatedNotify(),
		clientv3.WithPrevKV())
	resp, ok := <-wch
	if err := resp.Err(); err != nil || !ok {
		cancel()
		return nil, v2error.NewError(v2error.EcodeRaftInternal, prefix, 0)
	}

	evc, donec := make(chan *v2store.Event), make(chan struct{})
	go func() {
		defer func() {
			close(evc)
			close(donec)
		}()
		for resp := range wch {
			for _, ev := range s.mkV2Events(resp) {
				k := ev.Node.Key
				if recursive {
					if !strings.HasPrefix(k, prefix) {
						continue
					}
					// accept events on hidden keys given in prefix
					k = strings.Replace(k, prefix, "/", 1)
					// ignore hidden keys deeper than prefix
					if strings.Contains(k, "/_") {
						continue
					}
				}
				if !recursive && k != prefix {
					continue
				}
				select {
				case evc <- ev:
				case <-ctx.Done():
					return
				}
				if !stream {
					return
				}
			}
		}
	}()

	return &v2v3Watcher{
		startRev: resp.Header.Revision,
		evc:      evc,
		donec:    donec,
		cancel:   cancel,
	}, nil
}

func (s *v2v3Store) mkV2Events(wr clientv3.WatchResponse) (evs []*v2store.Event) {
	ak := s.mkActionKey()
	for _, rev := range mkRevs(wr) {
		var act, key *clientv3.Event
		for _, ev := range rev {
			if string(ev.Kv.Key) == ak {
				act = ev
			} else if key != nil && len(key.Kv.Key) < len(ev.Kv.Key) {
				// use longest key to ignore intermediate new
				// directories from Create.
				key = ev
			} else if key == nil {
				key = ev
			}
		}
		if act != nil && act.Kv != nil && key != nil {
			v2ev := &v2store.Event{
				Action:    string(act.Kv.Value),
				Node:      s.mkV2Node(key.Kv),
				PrevNode:  s.mkV2Node(key.PrevKv),
				EtcdIndex: mkV2Rev(wr.Header.Revision),
			}
			evs = append(evs, v2ev)
		}
	}
	return evs
}

func mkRevs(wr clientv3.WatchResponse) (revs [][]*clientv3.Event) {
	var curRev []*clientv3.Event
	for _, ev := range wr.Events {
		if curRev != nil && ev.Kv.ModRevision != curRev[0].Kv.ModRevision {
			revs = append(revs, curRev)
			curRev = nil
		}
		curRev = append(curRev, ev)
	}
	if curRev != nil {
		revs = append(revs, curRev)
	}
	return revs
}

type v2v3Watcher struct {
	startRev int64
	evc      chan *v2store.Event
	donec    chan struct{}
	cancel   context.CancelFunc
}

func (w *v2v3Watcher) StartIndex() uint64 { return mkV2Rev(w.startRev) }

func (w *v2v3Watcher) Remove() {
	w.cancel()
	<-w.donec
}

func (w *v2v3Watcher) EventChan() chan *v2store.Event { return w.evc }
