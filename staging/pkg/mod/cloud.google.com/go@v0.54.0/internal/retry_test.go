// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestRetry(t *testing.T) {
	ctx := context.Background()
	// Without a context deadline, retry will run until the function
	// says not to retry any more.
	n := 0
	endRetry := errors.New("end retry")
	err := retry(ctx, gax.Backoff{},
		func() (bool, error) {
			n++
			if n < 10 {
				return false, nil
			}
			return true, endRetry
		},
		func(context.Context, time.Duration) error { return nil })
	if got, want := err, endRetry; got != want {
		t.Errorf("got %v, want %v", err, endRetry)
	}
	if n != 10 {
		t.Errorf("n: got %d, want %d", n, 10)
	}

	// If the context has a deadline, sleep will return an error
	// and end the function.
	n = 0
	err = retry(ctx, gax.Backoff{},
		func() (bool, error) { return false, nil },
		func(context.Context, time.Duration) error {
			n++
			if n < 10 {
				return nil
			}
			return context.DeadlineExceeded
		})
	if err == nil {
		t.Error("got nil, want error")
	}
}

func TestRetryPreserveError(t *testing.T) {
	// Retry tries to preserve the type and other information from
	// the last error returned by the function.
	err := retry(context.Background(), gax.Backoff{},
		func() (bool, error) {
			return false, status.Error(codes.NotFound, "not found")
		},
		func(context.Context, time.Duration) error {
			return context.DeadlineExceeded
		})
	got, ok := status.FromError(err)
	if !ok {
		t.Fatalf("got %T, wanted a status", got)
	}
	if g, w := got.Code(), codes.NotFound; g != w {
		t.Errorf("got code %v, want %v", g, w)
	}
	wantMessage := fmt.Sprintf("retry failed with %v; last error: not found", context.DeadlineExceeded)
	if g, w := got.Message(), wantMessage; g != w {
		t.Errorf("got message %q, want %q", g, w)
	}
}
