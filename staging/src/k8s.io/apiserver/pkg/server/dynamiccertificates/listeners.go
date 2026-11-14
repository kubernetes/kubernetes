/*
Copyright The Kubernetes Authors.

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

package dynamiccertificates

import "sync"

// listenerList is a list of Listeners that is safe for concurrent use.
// The zero value is an empty list ready to use.
type listenerList struct {
	lock      sync.RWMutex
	listeners []Listener
}

// Add registers a listener. It is safe to call concurrently with Notify.
func (l *listenerList) Add(listener Listener) {
	l.lock.Lock()
	defer l.lock.Unlock()
	l.listeners = append(l.listeners, listener)
}

// Notify calls Enqueue on every listener registered at the time of the call.
// Listeners are called without the lock held.
func (l *listenerList) Notify() {
	l.lock.RLock()
	listeners := make([]Listener, len(l.listeners))
	copy(listeners, l.listeners)
	l.lock.RUnlock()

	for _, listener := range listeners {
		listener.Enqueue()
	}
}
