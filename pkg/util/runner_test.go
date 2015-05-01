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

package util

import (
	"fmt"
	"sync"
	"testing"
)

func TestRunner(t *testing.T) {
	var (
		lock   sync.Mutex
		events []string
		funcs  []func(chan struct{})
	)
	done := make(chan struct{}, 20)
	for i := 0; i < 10; i++ {
		iCopy := i
		funcs = append(funcs, func(c chan struct{}) {
			lock.Lock()
			events = append(events, fmt.Sprintf("%v starting\n", iCopy))
			lock.Unlock()
			<-c
			lock.Lock()
			events = append(events, fmt.Sprintf("%v stopping\n", iCopy))
			lock.Unlock()
			done <- struct{}{}
		})
	}

	r := NewRunner(funcs...)
	r.Start()
	r.Stop()
	for i := 0; i < 10; i++ {
		<-done
	}
	if len(events) != 20 {
		t.Errorf("expected 20 events, but got:\n%v\n", events)
	}
}
