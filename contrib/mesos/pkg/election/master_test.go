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
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/contrib/mesos/pkg/runtime"
)

type slowService struct {
	t  *testing.T
	on bool
	// We explicitly have no lock to prove that
	// Start and Stop are not called concurrently.
	changes chan<- bool
	done    <-chan struct{}
}

func (s *slowService) Validate(d, c Master) {
	// noop
}

func (s *slowService) Start() {
	select {
	case <-s.done:
		return // avoid writing to closed changes chan
	default:
	}
	if s.on {
		s.t.Errorf("started already on service")
	}
	time.Sleep(2 * time.Millisecond)
	s.on = true
	s.changes <- true
}

func (s *slowService) Stop() {
	select {
	case <-s.done:
		return // avoid writing to closed changes chan
	default:
	}
	if !s.on {
		s.t.Errorf("stopped already off service")
	}
	time.Sleep(2 * time.Millisecond)
	s.on = false
	s.changes <- false
}

func Test(t *testing.T) {
	m := NewFake()
	changes := make(chan bool, 1500)
	done := make(chan struct{})
	s := &slowService{t: t, changes: changes, done: done}
	notifyDone := runtime.After(func() { Notify(m, "", "me", s, done) })

	go func() {
		defer close(done)
		for i := 0; i < 500; i++ {
			for _, key := range []string{"me", "notme", "alsonotme"} {
				m.ChangeMaster(Master(key))
			}
		}
	}()

	<-notifyDone
	close(changes)

	changeList := []bool{}
	for {
		change, ok := <-changes
		if !ok {
			break
		}
		changeList = append(changeList, change)
	}

	if len(changeList) > 1000 {
		t.Errorf("unexpected number of changes: %v", len(changeList))
	}
}
