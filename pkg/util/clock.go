/*
Copyright 2014 Google Inc. All rights reserved.

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
}

// RealClock really calls time.Now()
type RealClock struct{}

// Now returns the current time.
func (r RealClock) Now() time.Time {
	return time.Now()
}

// FakeClock implements Clock, but returns an arbitary time.
type FakeClock struct {
	Time time.Time
}

// Now returns f's time.
func (f *FakeClock) Now() time.Time {
	return f.Time
}
