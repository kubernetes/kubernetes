/*
Copyright 2022 The Kubernetes Authors.

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

package metrics

import (
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

type testItem struct {
	name string
}

func TestQueueMetrics(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	metrics := queueMetrics{
		clock:           fakeClock,
		startRetryTimes: map[interface{}]time.Time{},
	}

	a := testItem{name: "a"}
	b := testItem{name: "b"}
	c := testItem{name: "c"}

	metrics.AddRateLimited(a)
	metrics.AddRateLimited(a)

	if len(metrics.GetRetrySinceDurations()) != 1 {
		t.Errorf("metrics should list one retrying item")
	}

	metrics.AddRateLimited(b)
	fakeClock.SetTime(fakeClock.Now().Add(5 * time.Minute))
	metrics.AddRateLimited(c)
	fakeClock.SetTime(fakeClock.Now().Add(1 * time.Minute))
	metrics.Forget(a)

	retryDurations := metrics.GetRetrySinceDurations()
	if len(retryDurations) != 2 {
		t.Errorf("metrics should list two retrying items")
	}
	if retryDurations[b] != 6*time.Minute {
		t.Errorf("%v item should have been retrying for %d minutes", b.name, 6)
	}
	if retryDurations[c] != 1*time.Minute {
		t.Errorf("%v item should have been retrying for %d minutes", c.name, 1)
	}

	metrics.Forget(c)
	metrics.Forget(b)
	metrics.Forget(b)

	if len(metrics.GetRetrySinceDurations()) != 0 {
		t.Errorf("metrics should list 0 retrying items")
	}
}
