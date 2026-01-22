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

package timer

import (
	"testing"
	"time"

	"github.com/onsi/gomega"
)

var currentTime time.Time

func init() {
	setCurrentTimeSinceEpoch(0)
	now = func() time.Time { return currentTime }
}

func setCurrentTimeSinceEpoch(duration time.Duration) {
	currentTime = time.Unix(0, duration.Nanoseconds())
}

func testUsageWithDefer(timer *TestPhaseTimer) {
	defer timer.StartPhase(33, "two").End()
	setCurrentTimeSinceEpoch(6*time.Second + 500*time.Millisecond)
}

func TestTimer(t *testing.T) {
	gomega.RegisterTestingT(t)

	timer := NewTestPhaseTimer()
	setCurrentTimeSinceEpoch(1 * time.Second)
	phaseOne := timer.StartPhase(1, "one")
	setCurrentTimeSinceEpoch(3 * time.Second)
	testUsageWithDefer(timer)

	gomega.Expect(timer.PrintJSON()).To(gomega.MatchJSON(`{
		"version": "v1",
		"dataItems": [
			{
				"data": {
					"001-one": 5.5,
					"033-two": 3.5
				},
				"unit": "s",
				"labels": {
					"test": "phases",
					"ended": "false"
				}
			}
		]
	}`))
	gomega.Expect(timer.PrintHumanReadable()).To(gomega.Equal(`Phase 001-one: 5.5s so far
Phase 033-two: 3.5s
`))

	setCurrentTimeSinceEpoch(7*time.Second + 500*time.Millisecond)
	phaseOne.End()

	gomega.Expect(timer.PrintJSON()).To(gomega.MatchJSON(`{
		"version": "v1",
		"dataItems": [
			{
				"data": {
					"001-one": 6.5,
					"033-two": 3.5
				},
				"unit": "s",
				"labels": {
					"test": "phases"
				}
			}
		]
	}`))
	gomega.Expect(timer.PrintHumanReadable()).To(gomega.Equal(`Phase 001-one: 6.5s
Phase 033-two: 3.5s
`))
}
