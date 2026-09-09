/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package coverage

import (
	"sync"

	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

type coverageWatcher struct {
	underlying watch.Interface
	key        string
	opts       storage.ListOptions
	tracker    *CoverageTracker
	resultCh   chan watch.Event
	stopOnce   sync.Once
}

func newCoverageWatcher(underlying watch.Interface, key string, opts storage.ListOptions, tracker *CoverageTracker) watch.Interface {
	cw := &coverageWatcher{
		underlying: underlying,
		key:        key,
		opts:       opts,
		tracker:    tracker,
		resultCh:   make(chan watch.Event),
	}

	go func() {
		defer close(cw.resultCh)
		for ev := range underlying.ResultChan() {
			tracker.RecordWatchEvent(ev)
			cw.resultCh <- ev
		}
	}()

	return cw
}

func (w *coverageWatcher) Stop() {
	w.stopOnce.Do(func() {
		w.underlying.Stop()
	})
}

func (w *coverageWatcher) ResultChan() <-chan watch.Event {
	return w.resultCh
}
