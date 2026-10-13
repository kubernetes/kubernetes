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

package rest

import (
	"context"
	"fmt"
	"net/url"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/klog/v2"
)

// Set of resp. Codes that we backoff for.
// In general these should be errors that indicate a server is overloaded.
// These shouldn't be configured by any user, we set them based on conventions
// described in
var serverIsOverloadedSet = sets.NewInt(429)
var maxResponseCode = 499

//go:generate mockery

// Deprecated: BackoffManager.Sleep ignores the caller's context. Use BackoffManagerWithContext instead.
type BackoffManager interface {
	UpdateBackoff(actualURL *url.URL, err error, responseCode int)
	CalculateBackoff(actualURL *url.URL) time.Duration
	Sleep(d time.Duration)
}

type BackoffManagerWithContext interface {
	UpdateBackoffWithContext(ctx context.Context, actualURL *url.URL, err error, responseCode int)
	CalculateBackoffWithContext(ctx context.Context, actualURL *url.URL) time.Duration
	SleepWithContext(ctx context.Context, d time.Duration)
}

var _ BackoffManager = &URLBackoff{}
var _ BackoffManagerWithContext = &URLBackoff{}

// URLBackoff struct implements the semantics on top of Backoff which
// we need for URL specific exponential backoff.
type URLBackoff struct {
	// Uses backoff as underlying implementation.
	Backoff *flowcontrol.Backoff
}

// NoBackoff is a stub implementation, can be used for mocking or else as a default.
type NoBackoff struct {
}

func (n *NoBackoff) UpdateBackoff(actualURL *url.URL, err error, responseCode int) {
	// do nothing.
}

func (n *NoBackoff) UpdateBackoffWithContext(ctx context.Context, actualURL *url.URL, err error, responseCode int) {
	// do nothing.
}

func (n *NoBackoff) CalculateBackoff(actualURL *url.URL) time.Duration {
	return 0 * time.Second
}

func (n *NoBackoff) CalculateBackoffWithContext(ctx context.Context, actualURL *url.URL) time.Duration {
	return 0 * time.Second
}

func (n *NoBackoff) Sleep(d time.Duration) {
	time.Sleep(d)
}

func (n *NoBackoff) SleepWithContext(ctx context.Context, d time.Duration) {
	if d == 0 {
		return
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	}
}

// Disable makes the backoff trivial, i.e., sets it to zero.  This might be used
// by tests which want to run 1000s of mock requests without slowing down.
func (b *URLBackoff) Disable() {
	b.Backoff = flowcontrol.NewBackOff(0*time.Second, 0*time.Second)
}

// baseUrlKey returns the key which urls will be mapped to.
// For example, 127.0.0.1:8080/api/v2/abcde -> 127.0.0.1:8080.
func (b *URLBackoff) baseUrlKey(rawurl *url.URL) string {
	// Simple implementation for now, just the host.
	// We may backoff specific paths (i.e. "pods") differentially
	// in the future.
	host, err := url.Parse(rawurl.String())
	if err != nil {
		panic(fmt.Sprintf("Error parsing bad URL %q: %v", rawurl, err))
	}
	return host.Host
}

// UpdateBackoff updates backoff metadata
func (b *URLBackoff) UpdateBackoff(actualURL *url.URL, err error, responseCode int) {
	b.UpdateBackoffWithContext(context.Background(), actualURL, err, responseCode)
}

// UpdateBackoffWithContext updates backoff metadata
func (b *URLBackoff) UpdateBackoffWithContext(ctx context.Context, actualURL *url.URL, err error, responseCode int) {
	// range for retry counts that we store is [0,13]
	if responseCode > maxResponseCode || serverIsOverloadedSet.Has(responseCode) {
		b.Backoff.Next(b.baseUrlKey(actualURL), b.Backoff.Clock.Now())
		return
	} else if responseCode >= 300 || err != nil {
		klog.FromContext(ctx).V(4).Info("Client is returning errors", "code", responseCode, "err", err)
	}

	//If we got this far, there is no backoff required for this URL anymore.
	b.Backoff.Reset(b.baseUrlKey(actualURL))
}

// CalculateBackoff takes a url and back's off exponentially,
// based on its knowledge of existing failures.
func (b *URLBackoff) CalculateBackoff(actualURL *url.URL) time.Duration {
	return b.Backoff.Get(b.baseUrlKey(actualURL))
}

// CalculateBackoffWithContext takes a url and back's off exponentially,
// based on its knowledge of existing failures.
func (b *URLBackoff) CalculateBackoffWithContext(_ context.Context, actualURL *url.URL) time.Duration {
	return b.CalculateBackoff(actualURL)
}

func (b *URLBackoff) Sleep(d time.Duration) {
	b.Backoff.Clock.Sleep(d)
}

func (b *URLBackoff) SleepWithContext(ctx context.Context, d time.Duration) {
	if d == 0 {
		return
	}
	t := b.Backoff.Clock.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C():
	}
}

// backoffManagerNopContext wraps a BackoffManager and adds the *WithContext methods.
type backoffManagerNopContext struct {
	BackoffManager
}

var _ BackoffManager = &backoffManagerNopContext{}
var _ BackoffManagerWithContext = &backoffManagerNopContext{}

func (b *backoffManagerNopContext) UpdateBackoffWithContext(ctx context.Context, actualURL *url.URL, err error, responseCode int) {
	b.UpdateBackoff(actualURL, err, responseCode)
}

func (b *backoffManagerNopContext) CalculateBackoffWithContext(ctx context.Context, actualURL *url.URL) time.Duration {
	return b.CalculateBackoff(actualURL)
}

func (b *backoffManagerNopContext) SleepWithContext(ctx context.Context, d time.Duration) {
	b.Sleep(d)
}
