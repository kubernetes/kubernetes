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

package request

import (
	"context"
	"testing"
	"time"

	clocktesting "k8s.io/utils/clock/testing"
)

func TestLatencyTrackersFrom(t *testing.T) {
	type testCase struct {
		Durations    []time.Duration
		SumDurations time.Duration
		MaxDuration  time.Duration
	}
	tc := testCase{
		Durations:    []time.Duration{100, 200, 300, 200, 400, 300, 100},
		SumDurations: 1600,
		MaxDuration:  400,
	}
	t.Run("TestLatencyTrackersFrom", func(t *testing.T) {
		parent := context.TODO()
		_, ok := LatencyTrackersFrom(parent)
		if ok {
			t.Error("expected LatencyTrackersFrom to not be initialized")
		}

		clk := clocktesting.FakeClock{}
		ctx := WithLatencyTrackersAndCustomClock(parent, &clk)
		wd, ok := LatencyTrackersFrom(ctx)
		if !ok {
			t.Error("expected LatencyTrackersFrom to be initialized")
		}
		if wd.MutatingWebhookTracker.GetLatency() != 0 || wd.ValidatingWebhookTracker.GetLatency() != 0 || wd.APFQueueWaitTracker.GetLatency() != 0 {
			t.Error("expected values to be initialized to 0")
		}

		for _, d := range tc.Durations {
			wd.MutatingWebhookTracker.Track(func() { clk.Step(d) })
			wd.ValidatingWebhookTracker.Track(func() { clk.Step(d) })
			wd.APFQueueWaitTracker.Track(func() { clk.Step(d) })
		}

		wd, ok = LatencyTrackersFrom(ctx)
		if !ok {
			t.Errorf("expected webhook duration to be initialized")
		}

		if wd.MutatingWebhookTracker.GetLatency() != tc.SumDurations {
			t.Errorf("expected admit duration: %q, but got: %q", tc.SumDurations, wd.MutatingWebhookTracker.GetLatency())
		}

		if wd.ValidatingWebhookTracker.GetLatency() != tc.MaxDuration {
			t.Errorf("expected validate duration: %q, but got: %q", tc.MaxDuration, wd.ValidatingWebhookTracker.GetLatency())
		}

		if wd.APFQueueWaitTracker.GetLatency() != tc.MaxDuration {
			t.Errorf("expected priority & fairness duration: %q, but got: %q", tc.MaxDuration, wd.APFQueueWaitTracker.GetLatency())
		}
	})
}
