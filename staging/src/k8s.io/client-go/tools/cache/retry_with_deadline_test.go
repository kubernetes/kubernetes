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

package cache

import (
	"fmt"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	testingclock "k8s.io/utils/clock/testing"
)

type retryScenarioStep struct {
	clockStep time.Duration
	err       error
	wantRetry bool
}

func TestRetryWithDeadline(t *testing.T) {
	internalError := apierrors.NewInternalError(fmt.Errorf("etcdserver: no leader"))
	otherError := fmt.Errorf("some other error")

	testCases := []struct {
		name        string
		duration    time.Duration
		reset       time.Duration
		isRetryable func(error) bool
		scenario    []retryScenarioStep
	}{
		{
			name:        "Never retry when duration is zero",
			duration:    time.Duration(0),
			reset:       time.Second * 30,
			isRetryable: func(err error) bool { return false },
			scenario: []retryScenarioStep{
				{
					clockStep: time.Second * 1,
					err:       nil,
					wantRetry: false,
				},
				{
					clockStep: time.Second * 0,
					err:       internalError,
					wantRetry: false,
				},
				{
					clockStep: time.Second * 1,
					err:       otherError,
					wantRetry: false,
				},
			},
		},
		{
			name:        "Retry when internal error happens only within duration",
			duration:    time.Second * 1,
			reset:       time.Second * 30,
			isRetryable: apierrors.IsInternalError,
			scenario: []retryScenarioStep{
				{
					clockStep: time.Second * 1,
					err:       internalError,
					wantRetry: true,
				},
				{
					clockStep: time.Second * 1,
					err:       internalError,
					wantRetry: true,
				},
				{
					clockStep: time.Second * 1,
					err:       internalError,
					wantRetry: false,
				},
			},
		},
		{
			name:        "Don't retry when other error happens",
			duration:    time.Second * 1,
			reset:       time.Second * 30,
			isRetryable: apierrors.IsInternalError,
			scenario: []retryScenarioStep{
				{
					clockStep: time.Second * 1,
					err:       otherError,
					wantRetry: false,
				},
			},
		},
		{
			name:        "Ignore other errors for retries",
			duration:    time.Second * 1,
			reset:       time.Second * 30,
			isRetryable: apierrors.IsInternalError,
			scenario: []retryScenarioStep{
				{
					clockStep: time.Second * 1,
					err:       internalError,
					wantRetry: true,
				},
				{
					clockStep: time.Second * 0,
					err:       otherError,
					wantRetry: true,
				},
				{
					clockStep: time.Second * 1,
					err:       internalError,
					wantRetry: true,
				},
			},
		},
	}

	for _, tc := range testCases {
		fakeClock := testingclock.NewFakeClock(time.Now())
		retry := NewRetryWithDeadline(tc.duration, tc.reset, tc.isRetryable, fakeClock)

		for i, step := range tc.scenario {
			fakeClock.Step(step.clockStep)
			retry.After(step.err)
			result := retry.ShouldRetry()
			if result != step.wantRetry {
				t.Errorf("%v unexpected retry, step %d, result %v want %v", tc, i, result, step.wantRetry)
				break
			}
		}
	}
}
