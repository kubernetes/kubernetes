/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubectl

import (
	"os"
	"os/signal"

	"k8s.io/kubernetes/pkg/watch"
)

// WatchLoop loops, passing events in w to fn.
// If user sends interrupt signal, shut down cleanly. Otherwise, never return.
func WatchLoop(w watch.Interface, fn func(watch.Event) error) {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt)
	defer signal.Stop(signals)
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				return
			}
			if err := fn(event); err != nil {
				w.Stop()
			}
		case <-signals:
			w.Stop()
		}
	}
}
