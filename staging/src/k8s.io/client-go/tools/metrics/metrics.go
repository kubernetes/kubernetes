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

// Package metrics provides abstractions for registering which metrics
// to record.
package metrics

import (
	"net/url"
	"sync"
	"time"
)

var registerMetrics sync.Once

// DurationMetric is a measurement of some amount of time.
type DurationMetric interface {
	Observe(duration time.Duration)
}

// ExpiryMetric sets some time of expiry. If nil, assume not relevant.
type ExpiryMetric interface {
	Set(expiry *time.Time)
}

// LatencyMetric observes client latency partitioned by verb and url.
type LatencyMetric interface {
	Observe(verb string, u url.URL, latency time.Duration)
}

// ResultMetric counts response codes partitioned by method and host.
type ResultMetric interface {
	Increment(code string, method string, host string)
}

var (
	// ClientCertExpiry is the expiry time of a client certificate
	ClientCertExpiry ExpiryMetric = noopExpiry{}
	// ClientCertRotationAge is the age of a certificate that has just been rotated.
	ClientCertRotationAge DurationMetric = noopDuration{}
	// RequestLatency is the latency metric that rest clients will update.
	RequestLatency LatencyMetric = noopLatency{}
	// RequestResult is the result metric that rest clients will update.
	RequestResult ResultMetric = noopResult{}
)

// RegisterOpts contains all the metrics to register. Metrics may be nil.
type RegisterOpts struct {
	ClientCertExpiry      ExpiryMetric
	ClientCertRotationAge DurationMetric
	RequestLatency        LatencyMetric
	RequestResult         ResultMetric
}

// Register registers metrics for the rest client to use. This can
// only be called once.
func Register(opts RegisterOpts) {
	registerMetrics.Do(func() {
		if opts.ClientCertExpiry != nil {
			ClientCertExpiry = opts.ClientCertExpiry
		}
		if opts.ClientCertRotationAge != nil {
			ClientCertRotationAge = opts.ClientCertRotationAge
		}
		if opts.RequestLatency != nil {
			RequestLatency = opts.RequestLatency
		}
		if opts.RequestResult != nil {
			RequestResult = opts.RequestResult
		}
	})
}

type noopDuration struct{}

func (noopDuration) Observe(time.Duration) {}

type noopExpiry struct{}

func (noopExpiry) Set(*time.Time) {}

type noopLatency struct{}

func (noopLatency) Observe(string, url.URL, time.Duration) {}

type noopResult struct{}

func (noopResult) Increment(string, string, string) {}
