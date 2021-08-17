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

package testing

import (
	"time"

	"k8s.io/utils/clock"
)

var (
	_ = clock.PassiveClock(&SimpleIntervalClock{})
)

// SimpleIntervalClock implements clock.PassiveClock, but each invocation of Now steps the clock forward the specified duration
type SimpleIntervalClock struct {
	Time     time.Time
	Duration time.Duration
}

// Now returns i's time.
func (i *SimpleIntervalClock) Now() time.Time {
	i.Time = i.Time.Add(i.Duration)
	return i.Time
}

// Since returns time since the time in i.
func (i *SimpleIntervalClock) Since(ts time.Time) time.Duration {
	return i.Time.Sub(ts)
}
