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

	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue/priority"
)

type Delayed interface {
	// return the remaining delay; a non-positive value indicates no delay
	GetDelay() time.Duration
}

//TODO: did this need to match Context.Deadline()
type Scheduled interface {
	// when ok, returns the time when this event should be scheduled
	EventTime() (t time.Time, ok bool)
}

// No objects are ever expected to be sent over this channel. References to BreakChan
// instances may be nil (always blocking). Signalling over this channel is performed by
// closing the channel. As such there can only ever be a single signal sent over the
// lifetime of the channel.
type BreakChan <-chan struct{}

// an optional interface to be implemented by Delayed objects; returning a nil
// channel from Breaker() results in waiting the full delay duration
type Breakout interface {
	// return a channel that signals early departure from a blocking delay
	Breaker() BreakChan
}

type UniqueDelayed interface {
	queue.UniqueID
	Delayed
}

type UniqueScheduled interface {
	queue.UniqueID
	Scheduled
}

type UniqueItem interface {
	queue.UniqueID
	priority.Item
}
