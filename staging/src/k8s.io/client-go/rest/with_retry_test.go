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

package rest

import (
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
)

var alwaysRetryError = IsRetryableErrorFunc(func(_ *http.Request, _ error) bool {
	return true
})

func TestNextRetry(t *testing.T) {
	fakeError := errors.New("fake error")
	tests := []struct {
		name               string
		attempts           int
		maxRetries         int
		request            *http.Request
		response           *http.Response
		err                error
		retryableErrFunc   IsRetryableErrorFunc
		retryExpected      []bool
		retryAfterExpected []*RetryAfter
	}{
		{
			name:               "bad input, response and err are nil",
			maxRetries:         2,
			attempts:           1,
			request:            &http.Request{},
			response:           nil,
			err:                nil,
			retryExpected:      []bool{false},
			retryAfterExpected: []*RetryAfter{nil},
		},
		{
			name:          "zero maximum retry",
			maxRetries:    0,
			attempts:      1,
			request:       &http.Request{},
			response:      retryAfterResponse(),
			err:           nil,
			retryExpected: []bool{false},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
				},
			},
		},
		{
			name:       "server returned a retryable error",
			maxRetries: 3,
			attempts:   1,
			request:    &http.Request{},
			response:   nil,
			err:        fakeError,
			retryableErrFunc: func(_ *http.Request, err error) bool {
				if err == fakeError {
					return true
				}
				return false
			},
			retryExpected: []bool{true},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
					Wait:    time.Second,
					Reason:  "retries: 1, retry-after: 1s - retry-reason: due to retryable error, error: fake error",
				},
			},
		},
		{
			name:       "server returned a retryable HTTP 429 response",
			maxRetries: 3,
			attempts:   1,
			request:    &http.Request{},
			response: &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header: http.Header{
					"Retry-After":                    []string{"2"},
					"X-Kubernetes-Pf-Flowschema-Uid": []string{"fs-1"},
				},
			},
			err:           nil,
			retryExpected: []bool{true},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
					Wait:    2 * time.Second,
					Reason:  `retries: 1, retry-after: 2s - retry-reason: due to server-side throttling, FlowSchema UID: "fs-1"`,
				},
			},
		},
		{
			name:       "server returned a retryable HTTP 5xx response",
			maxRetries: 3,
			attempts:   1,
			request:    &http.Request{},
			response: &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Header: http.Header{
					"Retry-After": []string{"3"},
				},
			},
			err:           nil,
			retryExpected: []bool{true},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
					Wait:    3 * time.Second,
					Reason:  "retries: 1, retry-after: 3s - retry-reason: 503",
				},
			},
		},
		{
			name:       "server returned a non response without without a Retry-After header",
			maxRetries: 1,
			attempts:   1,
			request:    &http.Request{},
			response: &http.Response{
				StatusCode: http.StatusTooManyRequests,
				Header:     http.Header{},
			},
			err:           nil,
			retryExpected: []bool{false},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
				},
			},
		},
		{
			name:       "both response and err are set, err takes precedence",
			maxRetries: 1,
			attempts:   1,
			request:    &http.Request{},
			response:   retryAfterResponse(),
			err:        fakeError,
			retryableErrFunc: func(_ *http.Request, err error) bool {
				if err == fakeError {
					return true
				}
				return false
			},
			retryExpected: []bool{true},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
					Wait:    time.Second,
					Reason:  "retries: 1, retry-after: 1s - retry-reason: due to retryable error, error: fake error",
				},
			},
		},
		{
			name:             "all retries are exhausted",
			maxRetries:       3,
			attempts:         4,
			request:          &http.Request{},
			response:         nil,
			err:              fakeError,
			retryableErrFunc: alwaysRetryError,
			retryExpected:    []bool{true, true, true, false},
			retryAfterExpected: []*RetryAfter{
				{
					Attempt: 1,
					Wait:    time.Second,
					Reason:  "retries: 1, retry-after: 1s - retry-reason: due to retryable error, error: fake error",
				},
				{
					Attempt: 2,
					Wait:    time.Second,
					Reason:  "retries: 2, retry-after: 1s - retry-reason: due to retryable error, error: fake error",
				},
				{
					Attempt: 3,
					Wait:    time.Second,
					Reason:  "retries: 3, retry-after: 1s - retry-reason: due to retryable error, error: fake error",
				},
				{
					Attempt: 4,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			r := &withRetry{maxRetries: test.maxRetries}

			retryGot := make([]bool, 0)
			retryAfterGot := make([]*RetryAfter, 0)
			for i := 0; i < test.attempts; i++ {
				retryAfter, retry := r.NextRetry(test.request, test.response, test.err, test.retryableErrFunc)
				retryGot = append(retryGot, retry)
				retryAfterGot = append(retryAfterGot, retryAfter)
			}

			if !reflect.DeepEqual(test.retryExpected, retryGot) {
				t.Errorf("Expected retry: %t, but got: %t", test.retryExpected, retryGot)
			}
			if !reflect.DeepEqual(test.retryAfterExpected, retryAfterGot) {
				t.Errorf("Expected retry-after parameters to match, but got: %s", cmp.Diff(test.retryAfterExpected, retryAfterGot))
			}
		})
	}
}
