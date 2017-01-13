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
	"net/url"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/pkg/util/flowcontrol"
)

// Set of resp. Codes that we backoff for.
// In general these should be errors that indicate a server is overloaded.
// These shouldn't be configured by any user, we set them based on conventions
// described in
var serverIsOverloadedSet = sets.NewInt(429)
var maxResponseCode = 499

type BackoffManager interface {
	UpdateBackoff(actualUrl *url.URL, err error, responseCode int)
	CalculateBackoff(actualUrl *url.URL) time.Duration
	Sleep(d time.Duration)
}

// URLBackoff struct implements the semantics on top of Backoff which
// we need for URL specific exponential backoff.
type URLBackoff struct {
	// Uses backoff as underlying implementation.
	Backoff *flowcontrol.Backoff
}

// NoBackoff is a stub implementation, can be used for mocking or else as a default.
type NoBackoff struct {
}

func (n *NoBackoff) UpdateBackoff(actualUrl *url.URL, err error, responseCode int) {
	// do nothing.
}

func (n *NoBackoff) CalculateBackoff(actualUrl *url.URL) time.Duration {
	return 0 * time.Second
}

func (n *NoBackoff) Sleep(d time.Duration) {
	time.Sleep(d)
}

// Disable makes the backoff trivial, i.e., sets it to zero.  This might be used
// by tests which want to run 1000s of mock requests without slowing down.
func (b *URLBackoff) Disable() {
	glog.V(4).Infof("Disabling backoff strategy")
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
		glog.V(4).Infof("Error extracting url: %v", rawurl)
		panic("bad url!")
	}
	return host.Host
}

// UpdateBackoff updates backoff metadata
func (b *URLBackoff) UpdateBackoff(actualUrl *url.URL, err error, responseCode int) {
	// range for retry counts that we store is [0,13]
	if responseCode > maxResponseCode || serverIsOverloadedSet.Has(responseCode) {
		b.Backoff.Next(b.baseUrlKey(actualUrl), b.Backoff.Clock.Now())
		return
	} else if responseCode >= 300 || err != nil {
		glog.V(4).Infof("Client is returning errors: code %v, error %v", responseCode, err)
	}

	//If we got this far, there is no backoff required for this URL anymore.
	b.Backoff.Reset(b.baseUrlKey(actualUrl))
}

// CalculateBackoff takes a url and back's off exponentially,
// based on its knowledge of existing failures.
func (b *URLBackoff) CalculateBackoff(actualUrl *url.URL) time.Duration {
	return b.Backoff.Get(b.baseUrlKey(actualUrl))
}

func (b *URLBackoff) Sleep(d time.Duration) {
	b.Backoff.Clock.Sleep(d)
}
