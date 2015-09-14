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

package pleg

type fakeContainerEventWatcher struct {
	ch chan *ContainerEvent
}

var _ ContainerEventWatcher = &fakeContainerEventWatcher{}

func (f *fakeContainerEventWatcher) SetEvents(events []*ContainerEvent) {
	f.createChannelIfEmpty()
	for _, e := range events {
		f.ch <- e
	}
}

func (f *fakeContainerEventWatcher) Watch() (<-chan *ContainerEvent, error) {
	f.createChannelIfEmpty()
	return f.ch, nil
}

func (f *fakeContainerEventWatcher) Stop() {
	close(f.ch)
}

func (f *fakeContainerEventWatcher) createChannelIfEmpty() {
	if f.ch == nil {
		f.ch = make(chan *ContainerEvent, 1000)
	}
}
