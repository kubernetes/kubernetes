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

func TestWebhookDurationFrom(t *testing.T) {
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
	t.Run("TestWebhookDurationFrom", func(t *testing.T) {
		parent := context.TODO()
		_, ok := WebhookDurationFrom(parent)
		if ok {
			t.Error("expected WebhookDurationFrom to not be initialized")
		}

		clk := clocktesting.FakeClock{}
		ctx := WithWebhookDurationAndCustomClock(parent, &clk)
		wd, ok := WebhookDurationFrom(ctx)
		if !ok {
			t.Error("expected webhook duration to be initialized")
		}
		if wd.AdmitTracker.GetLatency() != 0 || wd.ValidateTracker.GetLatency() != 0 {
			t.Error("expected values to be initialized to 0")
		}

		for _, d := range tc.Durations {
			wd.AdmitTracker.Track(func() { clk.Step(d) })
			wd.ValidateTracker.Track(func() { clk.Step(d) })
		}

		wd, ok = WebhookDurationFrom(ctx)
		if !ok {
			t.Errorf("expected webhook duration to be initialized")
		}

		if wd.AdmitTracker.GetLatency() != tc.SumDurations {
			t.Errorf("expected admit duration: %q, but got: %q", tc.SumDurations, wd.AdmitTracker.GetLatency())
		}

		if wd.ValidateTracker.GetLatency() != tc.MaxDuration {
			t.Errorf("expected validate duration: %q, but got: %q", tc.MaxDuration, wd.ValidateTracker.GetLatency())
		}
	})
}
