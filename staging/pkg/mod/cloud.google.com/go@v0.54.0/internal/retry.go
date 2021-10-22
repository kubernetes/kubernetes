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
	"time"

	gax "github.com/googleapis/gax-go/v2"
)

// Retry calls the supplied function f repeatedly according to the provided
// backoff parameters. It returns when one of the following occurs:
// When f's first return value is true, Retry immediately returns with f's second
// return value.
// When the provided context is done, Retry returns with an error that
// includes both ctx.Error() and the last error returned by f.
func Retry(ctx context.Context, bo gax.Backoff, f func() (stop bool, err error)) error {
	return retry(ctx, bo, f, gax.Sleep)
}

func retry(ctx context.Context, bo gax.Backoff, f func() (stop bool, err error),
	sleep func(context.Context, time.Duration) error) error {
	var lastErr error
	for {
		stop, err := f()
		if stop {
			return err
		}
		// Remember the last "real" error from f.
		if err != nil && err != context.Canceled && err != context.DeadlineExceeded {
			lastErr = err
		}
		p := bo.Pause()
		if cerr := sleep(ctx, p); cerr != nil {
			if lastErr != nil {
				return Annotatef(lastErr, "retry failed with %v; last error", cerr)
			}
			return cerr
		}
	}
}
