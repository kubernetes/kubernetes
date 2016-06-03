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
	"sync"

	"k8s.io/kubernetes/pkg/watch"
)

// Fake allows for testing of anything consuming a MasterElector.
type Fake struct {
	mux           *watch.Broadcaster
	currentMaster Master
	lock          sync.Mutex // Protect access of currentMaster
}

// NewFake makes a new fake MasterElector.
func NewFake() *Fake {
	// 0 means block for clients.
	return &Fake{mux: watch.NewBroadcaster(0, watch.WaitIfChannelFull)}
}

func (f *Fake) ChangeMaster(newMaster Master) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.mux.Action(watch.Modified, newMaster)
	f.currentMaster = newMaster
}

func (f *Fake) Elect(path, id string) watch.Interface {
	f.lock.Lock()
	defer f.lock.Unlock()
	w := f.mux.Watch()
	if f.currentMaster != "" {
		f.mux.Action(watch.Modified, f.currentMaster)
	}
	return w
}
