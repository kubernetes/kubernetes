/*
Copyright 2021 The Kubernetes Authors.

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

package eventclock

import (
	"time"

	baseclock "k8s.io/utils/clock"
)

// EventFunc does some work that needs to be done at or after the
// given time.
type EventFunc func(time.Time)

// EventClock is an active clock abstraction for use in code that is
// testable with a fake clock that itself determines how time may be
// advanced.  The timing paradigm is invoking EventFuncs rather than
// synchronizing through channels, so that the fake clock has a handle
// on when associated activity is done.
type Interface interface {
	baseclock.PassiveClock

	// Sleep returns after the given duration (or more).
	Sleep(d time.Duration)

	// EventAfterDuration invokes the given EventFunc after the given duration (or more),
	// passing the time when the invocation was launched.
	EventAfterDuration(f EventFunc, d time.Duration)

	// EventAfterTime invokes the given EventFunc at the given time or later,
	// passing the time when the invocation was launched.
	EventAfterTime(f EventFunc, t time.Time)
}
