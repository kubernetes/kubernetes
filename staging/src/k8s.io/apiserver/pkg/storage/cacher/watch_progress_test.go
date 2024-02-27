/*
Copyright 2023 The Kubernetes Authors.

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

package cacher

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

var (
	pollPeriod      = time.Millisecond
	minimalNoChange = 20 * time.Millisecond
	pollTimeout     = 5 * time.Second
)

func TestConditionalProgressRequester(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	logger := klog.FromContext(ctx)

	clock := testingclock.NewFakeClock(time.Now())
	pr := newTestConditionalProgressRequester(clock)
	stopCh := make(chan struct{})
	go pr.Run(stopCh)
	var wantRequestsSent int32
	var requestsSent int32

	logger.Info("Wait for ticker to be created")
	for !clock.HasWaiters() {
		time.Sleep(pollPeriod)
	}

	logger.Info("No progress requests if no-one is waiting")
	clock.Step(progressRequestPeriod * 2)

	if err := pollConditionNoChange(pollPeriod, minimalNoChange, pollTimeout, func() bool {
		requestsSent = pr.progressRequestsSentCount.Load()
		return requestsSent == wantRequestsSent
	}); err != nil {
		t.Fatalf("Failed to wait progress requests, err: %s, want: %d , got %d", err, wantRequestsSent, requestsSent)
	}

	logger.Info("Adding waiters allows progress request to be sent")
	pr.Add()
	wantRequestsSent++
	if err := pollConditionNoChange(pollPeriod, minimalNoChange, pollTimeout, func() bool {
		requestsSent = pr.progressRequestsSentCount.Load()
		return requestsSent == wantRequestsSent
	}); err != nil {
		t.Fatalf("Failed to wait progress requests, err: %s, want: %d , got %d", err, wantRequestsSent, requestsSent)
	}

	logger.Info("Periodically request progress to be sent every period")
	for wantRequestsSent < 10 {
		clock.Step(progressRequestPeriod)
		wantRequestsSent++

		if err := pollConditionNoChange(pollPeriod, minimalNoChange, pollTimeout, func() bool {
			requestsSent = pr.progressRequestsSentCount.Load()
			return requestsSent == wantRequestsSent
		}); err != nil {
			t.Fatalf("Failed to wait progress requests, err: %s, want: %d , got %d", err, wantRequestsSent, requestsSent)
		}
	}
	pr.Remove()

	logger.Info("No progress requests if no-one is waiting")
	clock.Step(progressRequestPeriod * 2)
	if err := pollConditionNoChange(pollPeriod, minimalNoChange, pollTimeout, func() bool {
		requestsSent = pr.progressRequestsSentCount.Load()
		return requestsSent == wantRequestsSent
	}); err != nil {
		t.Fatalf("Failed to wait progress requests, err: %s, want: %d , got %d", err, wantRequestsSent, requestsSent)
	}

	logger.Info("No progress after stopping")
	close(stopCh)
	if err := pollConditionNoChange(pollPeriod, minimalNoChange, pollTimeout, func() bool {
		requestsSent = pr.progressRequestsSentCount.Load()
		return requestsSent == wantRequestsSent
	}); err != nil {
		t.Fatalf("Failed to wait progress requests, err: %s, want: %d , got %d", err, wantRequestsSent, requestsSent)
	}
	pr.Add()
	clock.Step(progressRequestPeriod * 2)
	if err := pollConditionNoChange(pollPeriod, minimalNoChange, pollTimeout, func() bool {
		requestsSent = pr.progressRequestsSentCount.Load()
		return requestsSent == wantRequestsSent
	}); err != nil {
		t.Fatalf("Failed to wait progress requests, err: %s, want: %d , got %d", err, wantRequestsSent, requestsSent)
	}
}

func newTestConditionalProgressRequester(clock clock.WithTicker) *testConditionalProgressRequester {
	pr := &testConditionalProgressRequester{}
	pr.conditionalProgressRequester = newConditionalProgressRequester(pr.RequestWatchProgress, clock, nil)
	return pr
}

type testConditionalProgressRequester struct {
	*conditionalProgressRequester
	progressRequestsSentCount atomic.Int32
}

func (pr *testConditionalProgressRequester) RequestWatchProgress(ctx context.Context) error {
	pr.progressRequestsSentCount.Add(1)
	return nil
}

func pollConditionNoChange(interval, stable, timeout time.Duration, condition func() bool) error {
	passCounter := 0
	requiredNumberOfPasses := int(stable/interval) + 1
	return wait.Poll(interval, timeout, func() (done bool, err error) {
		if condition() {
			passCounter++
		} else {
			passCounter = 0
		}
		return passCounter >= requiredNumberOfPasses, nil
	})
}
