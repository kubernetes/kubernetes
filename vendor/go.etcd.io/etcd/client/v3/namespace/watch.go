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

package namespace

import (
	"context"
	"sync"

	"go.etcd.io/etcd/client/v3"
)

type watcherPrefix struct {
	clientv3.Watcher
	pfx string

	wg       sync.WaitGroup
	stopc    chan struct{}
	stopOnce sync.Once
}

// NewWatcher wraps a Watcher instance so that all Watch requests
// are prefixed with a given string and all Watch responses have
// the prefix removed.
func NewWatcher(w clientv3.Watcher, prefix string) clientv3.Watcher {
	return &watcherPrefix{Watcher: w, pfx: prefix, stopc: make(chan struct{})}
}

func (w *watcherPrefix) Watch(ctx context.Context, key string, opts ...clientv3.OpOption) clientv3.WatchChan {
	// since OpOption is opaque, determine range for prefixing through an OpGet
	op := clientv3.OpGet(key, opts...)
	end := op.RangeBytes()
	pfxBegin, pfxEnd := prefixInterval(w.pfx, []byte(key), end)
	if pfxEnd != nil {
		opts = append(opts, clientv3.WithRange(string(pfxEnd)))
	}

	wch := w.Watcher.Watch(ctx, string(pfxBegin), opts...)

	// translate watch events from prefixed to unprefixed
	pfxWch := make(chan clientv3.WatchResponse)
	w.wg.Add(1)
	go func() {
		defer func() {
			close(pfxWch)
			w.wg.Done()
		}()
		for wr := range wch {
			for i := range wr.Events {
				wr.Events[i].Kv.Key = wr.Events[i].Kv.Key[len(w.pfx):]
				if wr.Events[i].PrevKv != nil {
					wr.Events[i].PrevKv.Key = wr.Events[i].Kv.Key
				}
			}
			select {
			case pfxWch <- wr:
			case <-ctx.Done():
				return
			case <-w.stopc:
				return
			}
		}
	}()
	return pfxWch
}

func (w *watcherPrefix) Close() error {
	err := w.Watcher.Close()
	w.stopOnce.Do(func() { close(w.stopc) })
	w.wg.Wait()
	return err
}
