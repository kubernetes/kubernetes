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

package clock

import (
	"time"

	baseclock "k8s.io/utils/clock"
)

// EventFunc does some work that needs to be done at or after the
// given time. After this function returns, associated work may continue
//  on other goroutines only if they are counted by the GoRoutineCounter
// of the FakeEventClock handling this EventFunc.
type EventFunc func(time.Time)

// EventClock fires event on time
type EventClock interface {
	baseclock.PassiveClock
	Sleep(d time.Duration)
	EventAfterDuration(f EventFunc, d time.Duration)
	EventAfterTime(f EventFunc, t time.Time)
}
