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
	"math/rand"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestRandomFitSchedulerNothingScheduled(t *testing.T) {
	fakeRegistry := FakePodLister{}
	r := rand.New(rand.NewSource(0))
	st := schedulerTester{
		t:            t,
		scheduler:    MakeRandomFitScheduler(&fakeRegistry, r),
		minionLister: FakeMinionLister{"m1", "m2", "m3"},
	}
	st.expectSchedule(api.Pod{}, "m3")
}

func TestRandomFitSchedulerFirstScheduled(t *testing.T) {
	fakeRegistry := FakePodLister{
		makePod("m1", 8080),
	}
	r := rand.New(rand.NewSource(0))
	st := schedulerTester{
		t:            t,
		scheduler:    MakeRandomFitScheduler(fakeRegistry, r),
		minionLister: FakeMinionLister{"m1", "m2", "m3"},
	}
	st.expectSchedule(makePod("", 8080), "m3")
}

func TestRandomFitSchedulerFirstScheduledComplicated(t *testing.T) {
	fakeRegistry := FakePodLister{
		makePod("m1", 80, 8080),
		makePod("m2", 8081, 8082, 8083),
		makePod("m3", 80, 443, 8085),
	}
	r := rand.New(rand.NewSource(0))
	st := schedulerTester{
		t:            t,
		scheduler:    MakeRandomFitScheduler(fakeRegistry, r),
		minionLister: FakeMinionLister{"m1", "m2", "m3"},
	}
	st.expectSchedule(makePod("", 8080, 8081), "m3")
}

func TestRandomFitSchedulerFirstScheduledImpossible(t *testing.T) {
	fakeRegistry := FakePodLister{
		makePod("m1", 8080),
		makePod("m2", 8081),
		makePod("m3", 8080),
	}
	r := rand.New(rand.NewSource(0))
	st := schedulerTester{
		t:            t,
		scheduler:    MakeRandomFitScheduler(fakeRegistry, r),
		minionLister: FakeMinionLister{"m1", "m2", "m3"},
	}
	st.expectFailure(makePod("", 8080, 8081))
}
