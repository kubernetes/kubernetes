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

package util

import (
	"time"
)

// Clock allows for injecting fake or real clocks into code that
// needs to do arbitrary things based on time.
type Clock interface {
	Now() time.Time
	Since(time.Time) time.Duration
}

// RealClock really calls time.Now()
type RealClock struct{}

// Now returns the current time.
func (r RealClock) Now() time.Time {
	return time.Now()
}

// Since returns time since the specified timestamp.
func (r RealClock) Since(ts time.Time) time.Duration {
	return time.Since(ts)
}

// FakeClock implements Clock, but returns an arbitrary time.
type FakeClock struct {
	Time time.Time
}

// Now returns f's time.
func (f *FakeClock) Now() time.Time {
	return f.Time
}

// Since returns time since the time in f.
func (f *FakeClock) Since(ts time.Time) time.Duration {
	return f.Time.Sub(ts)
}

// Move clock by Duration
func (f *FakeClock) Step(d time.Duration) {
	f.Time = f.Time.Add(d)
}

// IntervalClock implements Clock, but each invocation of Now steps the clock forward the specified duration
type IntervalClock struct {
	Time     time.Time
	Duration time.Duration
}

// Now returns i's time.
func (i *IntervalClock) Now() time.Time {
	i.Time = i.Time.Add(i.Duration)
	return i.Time
}

// Since returns time since the time in i.
func (i *IntervalClock) Since(ts time.Time) time.Duration {
	return i.Time.Sub(ts)
}
