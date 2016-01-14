/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	log "github.com/golang/glog"
)

type (
	podWatchFilter func(*RegisteredPod) bool

	podWatchHandler struct {
		predicate podWatchFilter
		action    func(*RegisteredPod) error
		expired   <-chan struct{}
		forever   bool
	}

	podWatcher struct {
		updates  <-chan *RegisteredPod
		rw       sync.RWMutex
		handlers map[string]podWatchHandler
		filters  []podWatchFilter
	}
)

func newPodWatcher(updates <-chan *RegisteredPod) *podWatcher {
	return &podWatcher{
		updates:  updates,
		handlers: make(map[string]podWatchHandler),
	}
}

func (pw *podWatcher) run() {
updateLoop:
	for u := range pw.updates {
		log.V(1).Infof("filtering task %v pod %v/%v", u.taskID, u.Namespace, u.Name)
		for _, f := range pw.filters {
			if !f(u) {
				continue updateLoop
			}
		}
		log.V(1).Infof("handling task %v pod %v/%v", u.taskID, u.Namespace, u.Name)
		h, ok := func() (h podWatchHandler, ok bool) {
			pw.rw.RLock()
			defer pw.rw.RUnlock()
			h, ok = pw.handlers[u.taskID]
			return
		}()
		if ok && h.predicate(u) {
			log.V(1).Infof("executing action for task %v pod %v/%v", u.taskID, u.Namespace, u.Name)
			err := h.action(u)
			if err != nil {
				log.Error(err)
			} else if !h.forever {
				// de-register handler upon successful completion of action
				log.V(1).Infof("de-registering handler for task %v pod %v/%v", u.taskID, u.Namespace, u.Name)
				func() {
					pw.rw.Lock()
					delete(pw.handlers, u.taskID)
					pw.rw.Unlock()
				}()
			}
		}
	}
}

func (pw *podWatcher) filter(f podWatchFilter) {
	pw.rw.Lock()
	defer pw.rw.Unlock()
	pw.filters = append(pw.filters, f)
}

func (pw *podWatcher) forTask(taskID string, h podWatchHandler) {
	pw.rw.Lock()
	defer pw.rw.Unlock()

	pw.handlers[taskID] = h

	if h.expired != nil {
		log.V(1).Infof("expiring handler for task %v", taskID)
		go func() {
			<-h.expired

			// de-register handler upon expiration
			pw.rw.Lock()
			delete(pw.handlers, taskID)
			pw.rw.Unlock()

			h.action(&RegisteredPod{Pod: nil, taskID: taskID})
		}()
	}
}
