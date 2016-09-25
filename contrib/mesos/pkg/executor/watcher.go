/*
Copyright 2015 The Kubernetes Authors.

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

package executor

import (
	"sync"
	"time"

	log "github.com/golang/glog"
)

type (
	// filter registration events, return false to abort further processing of the event
	watchFilter func(pod *PodEvent) (accept bool)

	watchExpiration struct {
		// timeout closes when the handler has expired; it delivers at most one Time.
		timeout <-chan time.Time

		// onEvent is an optional callback that is invoked if/when the expired chan
		// closes
		onEvent func(taskID string)
	}

	watchHandler struct {
		// prevent callbacks from being invoked simultaneously
		sync.Mutex

		// handle registration events, return true to indicate the handler should be
		// de-registered upon completion. If pod is nil then the associated handler
		// has expired.
		onEvent func(pod *PodEvent) (done bool, err error)

		// expiration is an optional configuration that indicates when a handler should
		// be considered to have expired, and what action to take upon such
		expiration watchExpiration
	}

	// watcher observes PodEvent events and conditionally executes handlers that
	// have been associated with the taskID of the PodEvent.
	watcher struct {
		updates  <-chan *PodEvent
		rw       sync.RWMutex
		handlers map[string]*watchHandler
		filters  []watchFilter
		runOnce  chan struct{}
	}
)

func newWatcher(updates <-chan *PodEvent) *watcher {
	return &watcher{
		updates:  updates,
		handlers: make(map[string]*watchHandler),
		runOnce:  make(chan struct{}),
	}
}

func (pw *watcher) run() {
	select {
	case <-pw.runOnce:
		log.Error("run() has already been invoked for this pod-watcher")
		return
	default:
		close(pw.runOnce)
	}
updateLoop:
	for u := range pw.updates {
		log.V(2).Info("filtering " + u.FormatShort())
		for _, f := range pw.filters {
			if !f(u) {
				continue updateLoop
			}
		}
		log.V(1).Info("handling " + u.FormatShort())
		h, ok := func() (h *watchHandler, ok bool) {
			pw.rw.RLock()
			defer pw.rw.RUnlock()
			h, ok = pw.handlers[u.taskID]
			return
		}()
		if ok {
			log.V(1).Info("executing action for " + u.FormatShort())
			done, err := func() (bool, error) {
				h.Lock()
				defer h.Unlock()
				return h.onEvent(u)
			}()
			if err != nil {
				log.Error(err)
			}
			if done {
				// de-register handler upon successful completion of action
				log.V(1).Info("de-registering handler for " + u.FormatShort())
				func() {
					pw.rw.Lock()
					delete(pw.handlers, u.taskID)
					pw.rw.Unlock()
				}()
			}
		}
	}
}

func (pw *watcher) addFilter(f watchFilter) {
	select {
	case <-pw.runOnce:
		log.Errorf("failed to add filter because pod-watcher is already running")
	default:
		pw.filters = append(pw.filters, f)
	}
}

// forTask associates a handler `h` with the given taskID.
func (pw *watcher) forTask(taskID string, h *watchHandler) {
	pw.rw.Lock()
	pw.handlers[taskID] = h
	pw.rw.Unlock()

	if exp := h.expiration; exp.timeout != nil {
		go func() {
			<-exp.timeout
			log.V(1).Infof("expiring handler for task %v", taskID)

			// de-register handler upon expiration
			pw.rw.Lock()
			delete(pw.handlers, taskID)
			pw.rw.Unlock()

			if exp.onEvent != nil {
				h.Lock()
				defer h.Unlock()
				exp.onEvent(taskID)
			}
		}()
	}
}
