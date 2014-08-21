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

package scheduler

import (
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestRoundRobinScheduler(t *testing.T) {
	st := schedulerTester{
		t:            t,
		scheduler:    NewRoundRobinScheduler(),
		minionLister: FakeMinionLister{"m1", "m2", "m3", "m4"},
	}
	st.expectSchedule(api.Pod{}, "m1")
	st.expectSchedule(api.Pod{}, "m2")
	st.expectSchedule(api.Pod{}, "m3")
	st.expectSchedule(api.Pod{}, "m4")
}
