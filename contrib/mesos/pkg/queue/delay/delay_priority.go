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

package delay

import (
	"time"

	"k8s.io/kubernetes/contrib/mesos/pkg/queue/priority"
)

type DelayPriority struct {
	ts     time.Time // timestamp
	notify BreakChan // notification channel
}

func (p DelayPriority) Equal(other priority.Priority) bool {
	otherVal, ok := other.(DelayPriority)
	if !ok {
		return false
	}

	return p.ts.Equal(otherVal.ts) && p.notify == otherVal.notify
}

func (pq DelayPriority) Before(other priority.Priority) bool {
	otherVal, ok := other.(DelayPriority)
	if !ok {
		return false
	}

	return pq.ts.Before(otherVal.ts)
}

func extractFromDelayed(d Delayed) DelayPriority {
	deadline := time.Now().Add(d.GetDelay())
	breaker := BreakChan(nil)
	if breakout, good := d.(Breakout); good {
		breaker = breakout.Breaker()
	}

	return DelayPriority{
		ts:     deadline,
		notify: breaker,
	}
}

func extractFromDeadlined(d Deadlined) (DelayPriority, bool) {
	if ts, ok := d.Deadline(); ok {
		breaker := BreakChan(nil)
		if breakout, good := d.(Breakout); good {
			breaker = breakout.Breaker()
		}
		return DelayPriority{
			ts:     ts,
			notify: breaker,
		}, true
	}
	return DelayPriority{}, false
}
