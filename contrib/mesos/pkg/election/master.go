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

package election

import (
	"sync"

	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

// MasterElector is an interface for services that can elect masters.
// Important Note: MasterElectors are not inter-operable, all participants in the election need to be
//  using the same underlying implementation of this interface for correct behavior.
type MasterElector interface {
	// Elect makes the caller represented by 'id' enter into a master election for the
	// distributed lock defined by 'path'
	// The returned watch.Interface provides a stream of Master objects which
	// contain the current master.
	// Calling Stop on the returned interface relinquishes ownership (if currently possesed)
	// and removes the caller from the election
	Elect(path, id string) watch.Interface
}

// Service represents anything that can start and stop on demand.
type Service interface {
	Validate(desired, current Master)
	Start()
	Stop()
}

type notifier struct {
	changed chan struct{} // to notify the service loop about changed state

	// desired is updated with every change, current is updated after
	// Start()/Stop() finishes. 'cond' is used to signal that a change
	// might be needed. This handles the case where mastership flops
	// around without calling Start()/Stop() excessively.
	desired, current Master
	lock             sync.Mutex // to protect the desired variable

	// for comparison, to see if we are master.
	id Master

	service Service
}

// Notify runs Elect() on m, and calls Start()/Stop() on s when the
// elected master starts/stops matching 'id'. Never returns.
func Notify(m MasterElector, path, id string, s Service, abort <-chan struct{}) {
	n := &notifier{id: Master(id), service: s}
	n.changed = make(chan struct{})
	finished := runtime.After(func() {
		runtime.Until(func() {
			for {
				w := m.Elect(path, id)
				for {
					select {
					case <-abort:
						return
					case event, open := <-w.ResultChan():
						if !open {
							break
						}
						if event.Type != watch.Modified {
							continue
						}
						electedMaster, ok := event.Object.(Master)
						if !ok {
							glog.Errorf("Unexpected object from election channel: %v", event.Object)
							break
						}

						n.lock.Lock()
						n.desired = electedMaster
						n.lock.Unlock()

						// notify serviceLoop, but don't block. If a change
						// is queued already it will see the new n.desired.
						select {
						case n.changed <- struct{}{}:
						}
					}
				}
			}
		}, 0, abort)
	})
	runtime.Until(func() { n.serviceLoop(finished) }, 0, abort)
}

// serviceLoop waits for changes, and calls Start()/Stop() as needed.
func (n *notifier) serviceLoop(abort <-chan struct{}) {
	for {
		select {
		case <-abort:
			return
		case <-n.changed:
			n.lock.Lock()
			newDesired := n.desired // copy value to avoid race below
			n.lock.Unlock()

			if n.current != n.id && newDesired == n.id {
				n.service.Validate(newDesired, n.current)
				n.service.Start()
			} else if n.current == n.id && newDesired != n.id {
				n.service.Stop()
			}
			n.current = newDesired
		}
	}
}
