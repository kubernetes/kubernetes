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
	"testing"
	"time"

	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
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

	// change master to "notme" such that the initial m.Elect call inside Notify
	// will trigger an obversable event. We will wait for it to make sure the
	// Notify loop will see those master changes triggered by the go routine below.
	m.ChangeMaster(Master("me"))
	temporaryWatch := m.mux.Watch()
	ch := temporaryWatch.ResultChan()

	notifyDone := runtime.After(func() { Notify(m, "", "me", s, done) })

	// wait for the event triggered by the initial m.Elect of Notify. Then drain
	// the channel to not block anything.
	<-ch
	temporaryWatch.Stop()
	for i := 0; i < len(ch); i += 1 { // go 1.3 and 1.4 compatible loop
		<-ch
	}

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

	changesNum := len(changes)
	if changesNum > 1000 || changesNum == 0 {
		t.Errorf("unexpected number of changes: %v", changesNum)
	}
}
