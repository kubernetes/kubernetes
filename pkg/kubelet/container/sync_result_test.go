/*
Copyright 2015 The Kubernetes Authors.

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

package container

import (
	"errors"
	"fmt"
	"testing"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

func TestPodSyncResult(t *testing.T) {
	okResults := []*SyncResult{
		NewSyncResult(StartContainer, "container_0"),
		NewSyncResult(SetupNetwork, "pod"),
	}
	errResults := []*SyncResult{
		NewSyncResult(KillContainer, "container_1"),
		NewSyncResult(TeardownNetwork, "pod"),
	}
	errResults[0].Fail(errors.New("error_0"), "message_0")
	errResults[1].Fail(errors.New("error_1"), "message_1")

	// If the PodSyncResult doesn't contain error result, it should not be error
	result := PodSyncResult{}
	result.AddSyncResult(okResults...)
	if result.Error() != nil {
		t.Errorf("PodSyncResult should not be error: %v", result)
	}

	// If the PodSyncResult contains error result, it should be error
	result = PodSyncResult{}
	result.AddSyncResult(okResults...)
	result.AddSyncResult(errResults...)
	if result.Error() == nil {
		t.Errorf("PodSyncResult should be error: %v", result)
	}

	// If the PodSyncResult is failed, it should be error
	result = PodSyncResult{}
	result.AddSyncResult(okResults...)
	result.Fail(errors.New("error"))
	if result.Error() == nil {
		t.Errorf("PodSyncResult should be error: %v", result)
	}

	// If the PodSyncResult is added an error PodSyncResult, it should be error
	errResult := PodSyncResult{}
	errResult.AddSyncResult(errResults...)
	result = PodSyncResult{}
	result.AddSyncResult(okResults...)
	result.AddPodSyncResult(errResult)
	if result.Error() == nil {
		t.Errorf("PodSyncResult should be error: %v", result)
	}
}

// myCustomError is a custom error type for testing.
type myCustomError struct {
	msg string
}

func (e *myCustomError) Error() string {
	return e.msg
}

func TestPodSyncResultPreservesOriginalErrorType(t *testing.T) {
	t.Run("with custom error in SyncResult", func(t *testing.T) {
		customErr := &myCustomError{msg: "a special error"}
		syncResult := NewSyncResult(KillContainer, "container_1")
		syncResult.Fail(customErr, "a special message")
		result := &PodSyncResult{
			SyncResults: []*SyncResult{syncResult},
		}

		aggErr := result.Error()
		if aggErr == nil {
			t.Fatal("expected an aggregate error, but got nil")
		}

		var ae utilerrors.Aggregate
		if !errors.As(aggErr, &ae) {
			t.Fatalf("expected an aggregate error, but got %q", aggErr)
		}
		errs := ae.Errors()
		foundCustomErr := false
		for _, err := range errs {
			var ce *myCustomError
			if errors.As(err, &ce) && ce == customErr {
				foundCustomErr = true
			}
		}
		if !foundCustomErr {
			t.Errorf("expected custom error not found in aggregate error. got %q, want %q", aggErr, customErr)
		}
	})

	t.Run("with custom error in SyncError", func(t *testing.T) {
		customErr := &myCustomError{msg: "a special sync error"}
		result := &PodSyncResult{}
		result.Fail(customErr)

		aggErr := result.Error()
		if aggErr == nil {
			t.Fatal("expected an aggregate error for SyncError, but got nil")
		}

		var ae utilerrors.Aggregate
		if !errors.As(aggErr, &ae) {
			t.Fatalf("expected an aggregate error, but got %q", aggErr)
		}
		errs := ae.Errors()
		foundCustomErr := false
		for _, err := range errs {
			var ce *myCustomError
			if errors.As(err, &ce) && ce == customErr {
				foundCustomErr = true
			}
		}
		if !foundCustomErr {
			t.Errorf("expected custom error not found in aggregate error. got %q, want %q", aggErr, customErr)
		}
	})
}

func TestMinBackoffExpiration(t *testing.T) {
	now := time.Now()
	testCases := []struct {
		name            string
		err             error
		expectedBackoff time.Time
		expectedFound   bool
	}{
		{
			name:            "nil error",
			err:             nil,
			expectedBackoff: time.Time{},
			expectedFound:   false,
		},
		{
			name:            "simple error",
			err:             errors.New("generic error"),
			expectedBackoff: time.Time{},
			expectedFound:   false,
		},
		{
			name:            "BackoffError",
			err:             NewBackoffError(errors.New("backoff"), now.Add(5*time.Second)),
			expectedBackoff: now.Add(5 * time.Second),
			expectedFound:   true,
		},
		{
			name:            "wrapped BackoffError",
			err:             fmt.Errorf("wrapped: %w", NewBackoffError(errors.New("backoff"), now.Add(3*time.Second))),
			expectedBackoff: now.Add(3 * time.Second),
			expectedFound:   true,
		},
		{
			name:            "aggregate with no BackoffError",
			err:             utilerrors.NewAggregate([]error{errors.New("err1"), errors.New("err2")}),
			expectedBackoff: time.Time{},
			expectedFound:   false,
		},
		{
			name: "aggregate with one BackoffError",
			err: utilerrors.NewAggregate([]error{
				errors.New("err1"),
				NewBackoffError(errors.New("backoff"), now.Add(7*time.Second)),
			}),
			expectedBackoff: now.Add(7 * time.Second),
			expectedFound:   true,
		},
		{
			name: "aggregate with multiple BackoffErrors, returns minimum",
			err: utilerrors.NewAggregate([]error{
				NewBackoffError(errors.New("backoff1"), now.Add(10*time.Second)),
				NewBackoffError(errors.New("backoff2"), now.Add(3*time.Second)),
				errors.New("err1"),
				NewBackoffError(errors.New("backoff3"), now.Add(5*time.Second)),
			}),
			expectedBackoff: now.Add(3 * time.Second),
			expectedFound:   true,
		},
		{
			name: "wrapped aggregate with BackoffError",
			err: fmt.Errorf("wrapped: %w", utilerrors.NewAggregate([]error{
				NewBackoffError(errors.New("backoff1"), now.Add(10*time.Second)),
				NewBackoffError(errors.New("backoff2"), now.Add(3*time.Second)),
			})),
			expectedBackoff: now.Add(3 * time.Second),
			expectedFound:   true,
		},
		{
			name: "nested aggregate with BackoffError",
			err: utilerrors.NewAggregate([]error{
				errors.New("err1"),
				utilerrors.NewAggregate([]error{
					NewBackoffError(errors.New("backoff nested"), now.Add(2*time.Second)),
				}),
				NewBackoffError(errors.New("backoff outer"), now.Add(4*time.Second)),
			}),
			expectedBackoff: now.Add(2 * time.Second),
			expectedFound:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			backoff, found := MinBackoffExpiration(tc.err)
			if found != tc.expectedFound {
				t.Errorf("expected found=%t, got %t", tc.expectedFound, found)
			}
			if !backoff.Equal(tc.expectedBackoff) {
				t.Errorf("expected backoff=%v, got %v", tc.expectedBackoff, backoff)
			}
		})
	}
}
