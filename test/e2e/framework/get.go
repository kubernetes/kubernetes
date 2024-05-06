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

package framework

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/onsi/gomega"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// GetFunc is a function which retrieves a certain object.
type GetFunc[T any] func(ctx context.Context) (T, error)

// APIGetFunc is a get functions as used in client-go.
type APIGetFunc[T any] func(ctx context.Context, name string, getOptions metav1.GetOptions) (T, error)

// APIListFunc is a list functions as used in client-go.
type APIListFunc[T any] func(ctx context.Context, listOptions metav1.ListOptions) (T, error)

// GetObject takes a get function like clientset.CoreV1().Pods(ns).Get
// and the parameters for it and returns a function that executes that get
// operation in a [gomega.Eventually] or [gomega.Consistently].
//
// Delays and retries are handled by [HandleRetry]. A "not found" error is
// a fatal error that causes polling to stop immediately. If that is not
// desired, then wrap the result with [IgnoreNotFound].
func GetObject[T any](get APIGetFunc[T], name string, getOptions metav1.GetOptions) GetFunc[T] {
	return HandleRetry(func(ctx context.Context) (T, error) {
		return get(ctx, name, getOptions)
	})
}

// ListObjects takes a list function like clientset.CoreV1().Pods(ns).List
// and the parameters for it and returns a function that executes that list
// operation in a [gomega.Eventually] or [gomega.Consistently].
//
// Delays and retries are handled by [HandleRetry].
func ListObjects[T any](list APIListFunc[T], listOptions metav1.ListOptions) GetFunc[T] {
	return HandleRetry(func(ctx context.Context) (T, error) {
		return list(ctx, listOptions)
	})
}

// HandleRetry wraps an arbitrary get function. When the wrapped function
// returns an error, HandleGetError will decide whether the call should be
// retried and if requested, will sleep before doing so.
//
// This is meant to be used inside [gomega.Eventually] or [gomega.Consistently].
func HandleRetry[T any](get GetFunc[T]) GetFunc[T] {
	return func(ctx context.Context) (T, error) {
		t, err := get(ctx)
		if err != nil {
			if retry, delay := ShouldRetry(err); retry {
				if delay > 0 {
					// We could return
					// gomega.TryAgainAfter(delay) here,
					// but then we need to funnel that
					// error through any other
					// wrappers. Waiting directly is simpler.
					ctx, cancel := context.WithTimeout(ctx, delay)
					defer cancel()
					<-ctx.Done()
				}
				return t, err
			}
			// Give up polling immediately.
			var null T
			return t, gomega.StopTrying(fmt.Sprintf("Unexpected final error while getting %T", null)).Wrap(err)
		}
		return t, nil
	}
}

// ShouldRetry decides whether to retry an API request. Optionally returns a
// delay to retry after.
func ShouldRetry(err error) (retry bool, retryAfter time.Duration) {
	// if the error sends the Retry-After header, we respect it as an explicit confirmation we should retry.
	if delay, shouldRetry := apierrors.SuggestsClientDelay(err); shouldRetry {
		return shouldRetry, time.Duration(delay) * time.Second
	}

	// these errors indicate a transient error that should be retried.
	if apierrors.IsTimeout(err) ||
		apierrors.IsTooManyRequests(err) ||
		apierrors.IsServiceUnavailable(err) ||
		errors.As(err, &transientError{}) {
		return true, 0
	}

	return false, 0
}

// RetryNotFound wraps an arbitrary get function. When the wrapped function
// encounters a "not found" error, that error is treated as a transient problem
// and polling continues.
//
// This is meant to be used inside [gomega.Eventually] or [gomega.Consistently].
func RetryNotFound[T any](get GetFunc[T]) GetFunc[T] {
	return func(ctx context.Context) (T, error) {
		t, err := get(ctx)
		if apierrors.IsNotFound(err) {
			// If we are wrapping HandleRetry, then the error will
			// be gomega.StopTrying. We need to get rid of that,
			// otherwise gomega.Eventually will stop.
			var stopTryingErr gomega.PollingSignalError
			if errors.As(err, &stopTryingErr) {
				if wrappedErr := errors.Unwrap(stopTryingErr); wrappedErr != nil {
					err = wrappedErr
				}
			}

			// Mark the error as transient in case that we get
			// wrapped by HandleRetry.
			err = transientError{error: err}
		}
		return t, err
	}
}

// transientError wraps some other error and indicates that the
// wrapper error is something that may go away.
type transientError struct {
	error
}

func (err transientError) Unwrap() error {
	return err.error
}
