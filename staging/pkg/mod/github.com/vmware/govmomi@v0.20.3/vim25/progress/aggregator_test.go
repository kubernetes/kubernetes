/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package progress

import (
	"testing"
	"time"
)

func TestAggregatorNoSinks(t *testing.T) {
	ch := make(chan Report)
	a := NewAggregator(dummySinker{ch})
	a.Done()

	_, ok := <-ch
	if ok {
		t.Errorf("Expected channel to be closed")
	}
}

func TestAggregatorMultipleSinks(t *testing.T) {
	ch := make(chan Report)
	a := NewAggregator(dummySinker{ch})

	for i := 0; i < 5; i++ {
		go func(ch chan<- Report) {
			ch <- dummyReport{}
			ch <- dummyReport{}
			close(ch)
		}(a.Sink())

		<-ch
		<-ch
	}

	a.Done()

	_, ok := <-ch
	if ok {
		t.Errorf("Expected channel to be closed")
	}
}

func TestAggregatorSinkInFlightOnDone(t *testing.T) {
	ch := make(chan Report)
	a := NewAggregator(dummySinker{ch})

	// Simulate upstream
	go func(ch chan<- Report) {
		time.Sleep(1 * time.Millisecond)
		ch <- dummyReport{}
		close(ch)
	}(a.Sink())

	// Drain downstream
	go func(ch <-chan Report) {
		<-ch
	}(ch)

	// This should wait for upstream to complete
	a.Done()

	_, ok := <-ch
	if ok {
		t.Errorf("Expected channel to be closed")
	}
}
