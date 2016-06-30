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

package election

import (
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
	masters chan Master // elected masters arrive here, should be buffered to better deal with rapidly flapping masters

	// for comparison, to see if we are master.
	id Master

	service Service
}

// Notify runs Elect() on m, and calls Start()/Stop() on s when the
// elected master starts/stops matching 'id'. Never returns.
func Notify(m MasterElector, path, id string, s Service, abort <-chan struct{}) {
	n := &notifier{id: Master(id), service: s, masters: make(chan Master, 1)}
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

					sendElected:
						for {
							select {
							case <-abort:
								return
							case n.masters <- electedMaster:
								break sendElected
							default: // ring full, discard old value and add the new
								select {
								case <-abort:
									return
								case <-n.masters:
								default: // ring was cleared for us?!
								}
							}
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
	var current Master
	for {
		select {
		case <-abort:
			return
		case desired := <-n.masters:
			if current != n.id && desired == n.id {
				n.service.Validate(desired, current)
				n.service.Start()
			} else if current == n.id && desired != n.id {
				n.service.Stop()
			}
			current = desired
		}
	}
}
