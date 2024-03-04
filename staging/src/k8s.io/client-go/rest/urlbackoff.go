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

type BackoffManager interface {
	//logcheck:context // Use UpdateBackoffWithContext instead in code which supports contextual logging.
	UpdateBackoff(actualUrl *url.URL, err error, responseCode int)

	UpdateBackoffWithContext(ctx context.Context, actualUrl *url.URL, err error, responseCode int)

	//logcheck:context // Use CalculateBackoffWithContext instead in code which supports contextual logging.
	CalculateBackoff(actualUrl *url.URL) time.Duration

	CalculateBackoffWithContext(ctx context.Context, actualUrl *url.URL) time.Duration

	Sleep(d time.Duration)
}

// URLBackoff struct implements the semantics on top of Backoff which
// we need for URL specific exponential backoff.
type URLBackoff struct {
	// Uses backoff as underlying implementation.
	Backoff *flowcontrol.Backoff
}

// NewURLBackoff creates a new URLBackoff pointer instance.
func NewURLBackoff(backoff *flowcontrol.Backoff) *URLBackoff {
	return &URLBackoff{
		Backoff: backoff,
	}
}

// NoBackoff is a stub implementation, can be used for mocking or else as a default.
type NoBackoff struct {
}

//logcheck:context // Use UpdateBackoffWithContext instead in code which supports contextual logging.
func (n *NoBackoff) UpdateBackoff(actualUrl *url.URL, err error, responseCode int) {
	n.UpdateBackoffWithContext(context.Background(), actualUrl, err, responseCode)
}

func (n *NoBackoff) UpdateBackoffWithContext(ctx context.Context, actualUrl *url.URL, err error, responseCode int) {
	// do nothing.
}

//logcheck:context // Use CalculateBackoffWithContext instead in code which supports contextual logging.
func (n *NoBackoff) CalculateBackoff(actualUrl *url.URL) time.Duration {
	return n.CalculateBackoffWithContext(context.Background(), actualUrl)
}

func (n *NoBackoff) CalculateBackoffWithContext(ctx context.Context, actualUrl *url.URL) time.Duration {
	return 0 * time.Second
}

func (n *NoBackoff) Sleep(d time.Duration) {
	time.Sleep(d)
}

// Disable makes the backoff trivial, i.e., sets it to zero.  This might be used
// by tests which want to run 1000s of mock requests without slowing down.
func (b *URLBackoff) Disable(ctx context.Context) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("disabling backoff strategy")
	b.Backoff = flowcontrol.NewBackOff(0*time.Second, 0*time.Second)
}

// baseUrlKey returns the key which urls will be mapped to.
// For example, 127.0.0.1:8080/api/v2/abcde -> 127.0.0.1:8080.
func (b *URLBackoff) baseUrlKey(rawurl *url.URL, logger klog.Logger) string {
	// Simple implementation for now, just the host.
	// We may backoff specific paths (i.e. "pods") differentially
	// in the future.
	host, err := url.Parse(rawurl.String())
	if err != nil {
		logger.V(4).Error(err, "error extracting url", "rawurl", rawurl)
		panic("bad url!")
	}
	return host.Host
}

// UpdateBackoff updates backoff metadata
//
//logcheck:context // Use UpdateBackoffWithContext instead in code which supports contextual logging.
func (b *URLBackoff) UpdateBackoff(actualUrl *url.URL, err error, responseCode int) {
	b.UpdateBackoffWithContext(context.Background(), actualUrl, err, responseCode)
}

// UpdateBackoffWithContext updates backoff metadata and supports contextual logging
func (b *URLBackoff) UpdateBackoffWithContext(ctx context.Context, actualUrl *url.URL, err error, responseCode int) {
	logger := klog.FromContext(ctx)
	// range for retry counts that we store is [0,13]
	if responseCode > maxResponseCode || serverIsOverloadedSet.Has(responseCode) {
		b.Backoff.Next(b.baseUrlKey(actualUrl, logger), b.Backoff.Clock.Now())
		return
	} else if responseCode >= 300 || err != nil {
		logger.V(4).Info("client is returning errors", "code", responseCode, "err", err)
	}

	//If we got this far, there is no backoff required for this URL anymore.
	b.Backoff.Reset(b.baseUrlKey(actualUrl, logger))
}

// CalculateBackoff takes a url and back's off exponentially, based on its knowledge of existing failures.
//
//logcheck:context // Use CalculateBackoffWithContext instead in code which supports contextual logging.
func (b *URLBackoff) CalculateBackoff(actualUrl *url.URL) time.Duration {
	return b.CalculateBackoffWithContext(context.Background(), actualUrl)
}

// CalculateBackoff takes a url and back's off exponentially, based on its knowledge of existing failures.
// This method supports contextual logging.
func (b *URLBackoff) CalculateBackoffWithContext(ctx context.Context, actualUrl *url.URL) time.Duration {
	logger := klog.FromContext(ctx)
	return b.Backoff.Get(b.baseUrlKey(actualUrl, logger))
}

func (b *URLBackoff) Sleep(d time.Duration) {
	b.Backoff.Clock.Sleep(d)
}
