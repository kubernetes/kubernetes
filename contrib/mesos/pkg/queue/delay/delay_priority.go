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

type Priority struct {
	eventTime time.Time
	notify    BreakChan // notification channel
}

func NewPriority(eventTime time.Time) Priority {
	return Priority{eventTime: eventTime}
}

func (p Priority) Equal(other priority.Priority) bool {
	otherVal, ok := other.(Priority)
	if !ok {
		return false
	}

	return p.eventTime.Equal(otherVal.eventTime) && p.notify == otherVal.notify
}

func (pq Priority) Before(other priority.Priority) bool {
	otherVal, ok := other.(Priority)
	if !ok {
		return false
	}

	return pq.eventTime.Before(otherVal.eventTime)
}

func NewDelayedPriority(d Delayed) Priority {
	eventTime := time.Now().Add(d.GetDelay())
	var breaker BreakChan
	if breakout, good := d.(Breakout); good {
		breaker = breakout.Breaker()
	}

	return Priority{
		eventTime: eventTime,
		notify:    breaker,
	}
}

func NewScheduledPriority(d Scheduled) (Priority, bool) {
	if eventTime, ok := d.EventTime(); ok {
		breaker := BreakChan(nil)
		if breakout, good := d.(Breakout); good {
			breaker = breakout.Breaker()
		}
		return Priority{
			eventTime: eventTime,
			notify:    breaker,
		}, true
	}
	return Priority{}, false
}
