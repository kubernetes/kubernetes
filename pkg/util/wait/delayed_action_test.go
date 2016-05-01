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

package wait

import (
	"testing"
	"time"
)

func TestDelayedAction(t *testing.T) {
	defer func() { nowFn = time.Now }()
	now := time.Now()
	nowFn = func() time.Time { return now }

	// delay when rate is over 10/ms
	a := NewChannelDelayedAction(Interval{Count: 10, Interval: time.Millisecond}, Interval{Count: 100, Interval: 5 * time.Millisecond})
	// start off by running immediately because we're below the rate
	if !a.Run() || a.count != 1 || a.last != now || a.interval != 0 {
		t.Fatal(a)
	}
	if !a.Run() || a.count != 2 || a.last != now || a.interval != 0 {
		t.Fatal(a)
	}

	// switch to delaying because there are enough events instanteously
	a.count = 9
	if a.Run() || a.count != 0 || a.last != now || a.interval != time.Millisecond || a.active == nil {
		t.Fatal(a)
	}
	if a.Run() || a.count != 1 || a.last != now || a.interval != time.Millisecond || a.active == nil {
		t.Fatal(a)
	}
	<-a.After()
	a.Done()
	if !a.last.IsZero() || a.active != nil {
		t.Fatal(a)
	}

	// go back to running immediately, count is carried over
	if !a.Run() || a.count != 2 || a.last != now || a.interval != time.Millisecond {
		t.Fatal(a)
	}

	// go back to deferred
	a.count = 30
	old := now
	now = now.Add(2 * time.Millisecond)
	if a.Run() || a.count != 0 || a.last != old || a.interval != time.Millisecond {
		t.Fatal(a)
	}
	<-a.After()
	a.Done()

	// jump to a higher rate interval
	a.count = 100
	old = now
	now = now.Add(2 * time.Millisecond)
	if a.Run() || a.count != 0 || !a.last.IsZero() || a.interval != 5*time.Millisecond {
		t.Fatal(a)
	}
	<-a.After()
	a.Done()

	// drop down to the lower rate interval
	a.count = 10
	old = now
	now = now.Add(2 * time.Millisecond)
	if a.Run() || a.count != 0 || !a.last.IsZero() || a.interval != time.Millisecond {
		t.Fatal(a)
	}
	<-a.After()
	a.Done()
}
