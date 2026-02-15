/*
Copyright 2016 The Kubernetes Authors.

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

package retry

import (
	"context"
	stderrors "errors"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/wait"
)

// DefaultRetry is the recommended retry for a conflict where multiple clients
// are making changes to the same resource.
var DefaultRetry = wait.Backoff{
	Steps:    5,
	Duration: 10 * time.Millisecond,
	Factor:   1.0,
	Jitter:   0.1,
}

// DefaultBackoff is the recommended backoff for a conflict where a client
// may be attempting to make an unrelated modification to a resource under
// active management by one or more controllers.
var DefaultBackoff = wait.Backoff{
	Steps:    4,
	Duration: 10 * time.Millisecond,
	Factor:   5.0,
	Jitter:   0.1,
}

// OnErrorWithContext allows the caller to retry fn in case the error returned by fn
// is retriable according to the provided function. backoff defines the maximum retries
// and the wait interval between two retries.
//
// The context is used to allow the caller to cancel the retry operation.
// If the context is cancelled, OnErrorWithContext will return ctx.Err().
func OnErrorWithContext(ctx context.Context, backoff wait.Backoff, retriable func(error) bool, fn func(context.Context) error) error {
	var lastErr error
	err := wait.ExponentialBackoffWithContext(ctx, backoff, func(ctx context.Context) (bool, error) {
		err := fn(ctx)
		switch {
		case err == nil:
			return true, nil
		case retriable(err):
			lastErr = err
			return false, nil
		default:
			return false, err
		}
	})
	// If the context was cancelled or the deadline was exceeded, return the context error.
	// Otherwise, if we timed out (ran out of steps), return the last error we saw.
	if stderrors.Is(err, context.Canceled) || stderrors.Is(err, context.DeadlineExceeded) {
		return err
	}
	if wait.Interrupted(err) {
		return lastErr
	}
	return err
}

// OnError allows the caller to retry fn in case the error returned by fn is retriable
// according to the provided function. backoff defines the maximum retries and the wait
// interval between two retries.
func OnError(backoff wait.Backoff, retriable func(error) bool, fn func() error) error {
	return OnErrorWithContext(context.Background(), backoff, retriable, func(context.Context) error {
		return fn()
	})
}

// RetryOnConflict is used to make an update to a resource when you have to worry about
// conflicts caused by other code making unrelated updates to the resource at the same
// time. fn should fetch the resource to be modified, make appropriate changes to it, try
// to update it, and return (unmodified) the error from the update function. On a
// successful update, RetryOnConflict will return nil. If the update function returns a
// "Conflict" error, RetryOnConflict will wait some amount of time as described by
// backoff, and then try again. On a non-"Conflict" error, or if it retries too many times
// and gives up, RetryOnConflict will return an error to the caller.
//
//	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
//	    // Fetch the resource here; you need to refetch it on every try, since
//	    // if you got a conflict on the last update attempt then you need to get
//	    // the current version before making your own changes.
//	    pod, err := c.Pods("mynamespace").Get(name, metav1.GetOptions{})
//	    if err != nil {
//	        return err
//	    }
//
//	    // Make whatever updates to the resource are needed
//	    pod.Status.Phase = v1.PodFailed
//
//	    // Try to update
//	    _, err = c.Pods("mynamespace").UpdateStatus(pod)
//	    // You have to return err itself here (not wrapped inside another error)
//	    // so that RetryOnConflict can identify it correctly.
//	    return err
//	})
//	if err != nil {
//	    // May be conflict if max retries were hit, or may be something unrelated
//	    // like permissions or a network error
//	    return err
//	}
//	...
//
// TODO: Make Backoff an interface?
func RetryOnConflict(backoff wait.Backoff, fn func() error) error {
	return RetryOnConflictWithContext(context.Background(), backoff, func(context.Context) error {
		return fn()
	})
}

// RetryOnConflictWithContext is used to make an update to a resource when you have
// to worry about conflicts caused by other code making unrelated updates to the
// resource at the same time.
//
// The context is used to allow the caller to cancel the retry operation.
// If the context is cancelled, RetryOnConflictWithContext will return ctx.Err().
func RetryOnConflictWithContext(ctx context.Context, backoff wait.Backoff, fn func(context.Context) error) error {
	return OnErrorWithContext(ctx, backoff, errors.IsConflict, fn)
}
