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

	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

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
	lock sync.Mutex
	cond *sync.Cond

	// desired is updated with every change, current is updated after
	// Start()/Stop() finishes. 'cond' is used to signal that a change
	// might be needed. This handles the case where mastership flops
	// around without calling Start()/Stop() excessively.
	desired, current Master

	// for comparison, to see if we are master.
	id Master

	service Service
}

// Notify runs Elect() on m, and calls Start()/Stop() on s when the
// elected master starts/stops matching 'id'. Never returns.
func Notify(m MasterElector, path, id string, s Service, abort <-chan struct{}) {
	n := &notifier{id: Master(id), service: s}
	n.cond = sync.NewCond(&n.lock)
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
						func() {
							n.lock.Lock()
							defer n.lock.Unlock()
							n.desired = electedMaster
							if n.desired != n.current {
								n.cond.Signal()
							}
						}()
					}
				}
			}
		}, 0, abort)
	})
	runtime.Until(func() { n.serviceLoop(finished) }, 0, abort)
}

// serviceLoop waits for changes, and calls Start()/Stop() as needed.
func (n *notifier) serviceLoop(abort <-chan struct{}) {
	n.lock.Lock()
	defer n.lock.Unlock()
	for {
		select {
		case <-abort:
			return
		default:
			for n.desired == n.current {
				ch := runtime.After(n.cond.Wait)
				select {
				case <-abort:
					n.cond.Signal() // ensure that Wait() returns
					<-ch
					return
				case <-ch:
					// we were notified and have the lock, proceed..
				}
			}
			if n.current != n.id && n.desired == n.id {
				n.service.Validate(n.desired, n.current)
				n.service.Start()
			} else if n.current == n.id && n.desired != n.id {
				n.service.Stop()
			}
			n.current = n.desired
		}
	}
}
