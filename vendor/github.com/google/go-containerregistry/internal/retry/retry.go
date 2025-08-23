// Copyright 2019 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package retry provides methods for retrying operations. It is a thin wrapper
// around k8s.io/apimachinery/pkg/util/wait to make certain operations easier.
package retry

import (
	"context"
	"errors"
	"fmt"

	"github.com/google/go-containerregistry/internal/retry/wait"
)

// Backoff is an alias of our own wait.Backoff to avoid name conflicts with
// the kubernetes wait package. Typing retry.Backoff is aesier than fixing
// the wrong import every time you use wait.Backoff.
type Backoff = wait.Backoff

// This is implemented by several errors in the net package as well as our
// transport.Error.
type temporary interface {
	Temporary() bool
}

// IsTemporary returns true if err implements Temporary() and it returns true.
func IsTemporary(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if te, ok := err.(temporary); ok && te.Temporary() {
		return true
	}
	return false
}

// IsNotNil returns true if err is not nil.
func IsNotNil(err error) bool {
	return err != nil
}

// Predicate determines whether an error should be retried.
type Predicate func(error) (retry bool)

// Retry retries a given function, f, until a predicate is satisfied, using
// exponential backoff. If the predicate is never satisfied, it will return the
// last error returned by f.
func Retry(f func() error, p Predicate, backoff wait.Backoff) (err error) {
	if f == nil {
		return fmt.Errorf("nil f passed to retry")
	}
	if p == nil {
		return fmt.Errorf("nil p passed to retry")
	}

	condition := func() (bool, error) {
		err = f()
		if p(err) {
			return false, nil
		}
		return true, err
	}

	wait.ExponentialBackoff(backoff, condition)
	return
}

type contextKey string

var key = contextKey("never")

// Never returns a context that signals something should not be retried.
// This is a hack and can be used to communicate across package boundaries
// to avoid retry amplification.
func Never(ctx context.Context) context.Context {
	return context.WithValue(ctx, key, true)
}

// Ever returns true if the context was wrapped by Never.
func Ever(ctx context.Context) bool {
	return ctx.Value(key) == nil
}
