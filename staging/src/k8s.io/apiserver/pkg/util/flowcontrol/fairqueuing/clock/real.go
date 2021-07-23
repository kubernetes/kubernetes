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

	"k8s.io/utils/clock"
)

// RealEventClock fires event on real world time
type RealEventClock struct {
	clock.RealClock
}

// EventAfterDuration schedules an EventFunc
func (RealEventClock) EventAfterDuration(f EventFunc, d time.Duration) {
	ch := time.After(d)
	go func() {
		t := <-ch
		f(t)
	}()
}

// EventAfterTime schedules an EventFunc
func (r RealEventClock) EventAfterTime(f EventFunc, t time.Time) {
	r.EventAfterDuration(f, time.Until(t))
}
