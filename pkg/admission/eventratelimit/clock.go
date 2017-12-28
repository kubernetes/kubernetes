/*
Copyright 2017 The Kubernetes Authors.

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

package eventratelimit

import (
	"time"
)

// realClock implements flowcontrol.Clock in terms of standard time functions.
type realClock struct{}

// Now is identical to time.Now.
func (realClock) Now() time.Time {
	return time.Now()
}

// Sleep is identical to time.Sleep.
func (realClock) Sleep(d time.Duration) {
	time.Sleep(d)
}
