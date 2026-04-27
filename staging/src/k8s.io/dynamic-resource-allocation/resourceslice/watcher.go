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

package resourceslice

import (
	"context"

	"k8s.io/apimachinery/pkg/watch"
)

func newWrapWatcher(ctx context.Context, w watch.Interface, match func(event watch.Event) bool) *wrapWatcher {
	ctx, cancel := context.WithCancel(ctx)

	watcher := &wrapWatcher{
		watcher: w,
		match:   match,
		ctx:     ctx,
		cancel:  cancel,
		result:  make(chan watch.Event),
		done:    make(chan struct{}),
	}
	go watcher.receive(ctx)

	return watcher
}

type wrapWatcher struct {
	watcher watch.Interface
	match   func(event watch.Event) bool

	ctx    context.Context
	cancel context.CancelFunc
	result chan watch.Event
	done   chan struct{}
}

func (w *wrapWatcher) receive(ctx context.Context) {
	defer close(w.result)
	defer close(w.done)
	resultChan := w.watcher.ResultChan()
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-resultChan:
			if !ok {
				return
			}
			if w.match == nil || w.match(event) {
				select {
				case <-ctx.Done():
					return
				case w.result <- event:
				}
			}
		}
	}
}

func (w *wrapWatcher) ResultChan() <-chan watch.Event {
	return w.result
}

func (w *wrapWatcher) Stop() {
	select {
	case <-w.ctx.Done():
	default:
		w.watcher.Stop()
		w.cancel()
	}
	<-w.done
}
