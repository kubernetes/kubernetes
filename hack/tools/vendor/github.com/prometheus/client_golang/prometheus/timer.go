// Copyright 2016 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus

import "time"

// Timer is a helper type to time functions. Use NewTimer to create new
// instances.
type Timer struct {
	begin    time.Time
	observer Observer
}

// NewTimer creates a new Timer. The provided Observer is used to observe a
// duration in seconds. Timer is usually used to time a function call in the
// following way:
//    func TimeMe() {
//        timer := NewTimer(myHistogram)
//        defer timer.ObserveDuration()
//        // Do actual work.
//    }
func NewTimer(o Observer) *Timer {
	return &Timer{
		begin:    time.Now(),
		observer: o,
	}
}

// ObserveDuration records the duration passed since the Timer was created with
// NewTimer. It calls the Observe method of the Observer provided during
// construction with the duration in seconds as an argument. The observed
// duration is also returned. ObserveDuration is usually called with a defer
// statement.
//
// Note that this method is only guaranteed to never observe negative durations
// if used with Go1.9+.
func (t *Timer) ObserveDuration() time.Duration {
	d := time.Since(t.begin)
	if t.observer != nil {
		t.observer.Observe(d.Seconds())
	}
	return d
}
