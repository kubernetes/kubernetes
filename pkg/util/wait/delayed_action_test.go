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
	"math/rand"
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

func TestDelayedActionWait(t *testing.T) {
	// delay when rate is over 10/ms
	a := NewChannelDelayedAction(Interval{Count: 10, Interval: time.Millisecond}, Interval{Count: 100, Interval: 5 * time.Millisecond})

	expectTries := 9

	for i := 0; i < 500; i++ {
		// kick into after
		tries := 0
		for {
			if !a.Run() {
				break
			}
			tries++
		}
		if tries != expectTries {
			t.Fatalf("unexpected number of tries %d %d for %#v", expectTries, tries, a)
		}

		var times int
		switch rand.Int31n(5) {
		case 0:
			times = 1
		case 1:
			times = 5 + int(rand.Int31n(10))
		case 2:
			times = 95 + int(rand.Int31n(10))
		case 3:
			times = 150 + int(rand.Int31n(1000))
		default:
			t.Logf("sleeping until after")
			time.Sleep(5 + time.Duration(rand.Int31n(10))*time.Millisecond)

			expectTries = 9
		}

		if times > 0 {
			t.Logf("running %d times", times)
			for j := 0; j < times; j++ {
				if a.Run() {
					t.Fatalf("should not switch back to defer until after Done(): %#v", a)
				}
			}
			expectTries = 9 - times
			if expectTries < 0 {
				expectTries = 0
			}
		}

		select {
		case <-a.After():
			a.Done()
		case <-time.After(10 * time.Second):
			// this could be flaky, we could set this even higher
			t.Fatalf("timer did not fire within 1s: %#v", a)
		}
	}
}
