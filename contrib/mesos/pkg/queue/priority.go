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

package queue

import (
	"time"
)

type Priority struct {
	ts     time.Time // timestamp
	notify BreakChan // notification channel
}

func (p Priority) Equal(other Priority) bool {
	return p.ts.Equal(other.ts) && p.notify == other.notify
}

func extractFromDelayed(d Delayed) Priority {
	deadline := time.Now().Add(d.GetDelay())
	breaker := BreakChan(nil)
	if breakout, good := d.(Breakout); good {
		breaker = breakout.Breaker()
	}
	return Priority{
		ts:     deadline,
		notify: breaker,
	}
}

func extractFromDeadlined(d Deadlined) (Priority, bool) {
	if ts, ok := d.Deadline(); ok {
		breaker := BreakChan(nil)
		if breakout, good := d.(Breakout); good {
			breaker = breakout.Breaker()
		}
		return Priority{
			ts:     ts,
			notify: breaker,
		}, true
	}
	return Priority{}, false
}
