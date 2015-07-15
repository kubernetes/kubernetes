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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
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

// Notify runs Elect() on m, and calls Start()/Stop() on s when the
// elected master starts/stops matching 'id'. It only returns when the abort
// channel is closed.
func Notify(m MasterElector, path, id string, s Service, abort <-chan struct{}) {
	for {
		select {
		case <-abort:
			return
		default:
			notify(m, path, Master(id), s, abort)
		}
	}
}

func notify(m MasterElector, path string, id Master, s Service, abort <-chan struct{}) {
	defer util.HandleCrash()
	w := m.Elect(path, string(id))
	defer w.Stop()
	events := w.ResultChan()

	var current Master
	for {
		select {
		case <-abort:
			return
		case event, open := <-events:
			desired, ok := event.Object.(Master)
			switch {
			case !open:
				return
			case !ok || event.Type != watch.Modified:
				continue
			case current != id && desired == id:
				s.Validate(desired, current)
				s.Start()
			case current == id && desired != id:
				s.Stop()
			}
			current = desired
		}
	}
}
