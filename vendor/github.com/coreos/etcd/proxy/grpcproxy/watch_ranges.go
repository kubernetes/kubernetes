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
)

// watchRanges tracks all open watches for the proxy.
type watchRanges struct {
	wp *watchProxy

	mu     sync.Mutex
	bcasts map[watchRange]*watchBroadcasts
}

func newWatchRanges(wp *watchProxy) *watchRanges {
	return &watchRanges{
		wp:     wp,
		bcasts: make(map[watchRange]*watchBroadcasts),
	}
}

func (wrs *watchRanges) add(w *watcher) {
	wrs.mu.Lock()
	defer wrs.mu.Unlock()

	if wbs := wrs.bcasts[w.wr]; wbs != nil {
		wbs.add(w)
		return
	}
	wbs := newWatchBroadcasts(wrs.wp)
	wrs.bcasts[w.wr] = wbs
	wbs.add(w)
}

func (wrs *watchRanges) delete(w *watcher) {
	wrs.mu.Lock()
	defer wrs.mu.Unlock()
	wbs, ok := wrs.bcasts[w.wr]
	if !ok {
		panic("deleting missing range")
	}
	if wbs.delete(w) == 0 {
		wbs.stop()
		delete(wrs.bcasts, w.wr)
	}
}

func (wrs *watchRanges) stop() {
	wrs.mu.Lock()
	defer wrs.mu.Unlock()
	for _, wb := range wrs.bcasts {
		wb.stop()
	}
	wrs.bcasts = nil
}
