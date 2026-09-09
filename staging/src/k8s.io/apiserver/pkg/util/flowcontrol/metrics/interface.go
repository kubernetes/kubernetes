/*
Copyright 2019 The Kubernetes Authors.

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

package metrics

// Gauge is the methods of a gauge that are used by instrumented code.
type Gauge interface {
	Set(float64)
	Inc()
	Dec()
	Add(float64)
	SetToCurrentTime()
}

// RatioedGauge tracks ratios.
// The numerator is set/changed through the Gauge methods,
// and the denominator can be updated through the SetDenominator method.
// A ratio is tracked whenever the numerator or denominator is set/changed.
type RatioedGauge interface {
	Gauge

	// SetDenominator sets the denominator to use until it is changed again
	SetDenominator(float64)
}

// RatioedGaugeVec creates related observers that are
// differentiated by a series of label values
type RatioedGaugeVec interface {
	// NewForLabelValuesSafe makes a new vector member for the given tuple of label values,
	// initialized with the given numerator and denominator.
	// Unlike the usual Vec WithLabelValues method, this is intended to be called only
	// once per vector member (at the start of its lifecycle).
	// The "Safe" part is saying that the returned object will function properly after metric registration
	// even if this method is called before registration.
	NewForLabelValuesSafe(initialNumerator, initialDenominator float64, labelValues []string) RatioedGauge
}

//////////////////////////////// Pairs ////////////////////////////////
//
// API Priority and Fairness tends to use RatioedGaugeVec members in pairs,
// one for requests waiting in a queue and one for requests being executed.
// The following definitions are a convenience layer that adds support for that
// particular pattern of usage.

// RatioedGaugePair is a corresponding pair of gauges, one for the
// number of requests waiting in queue(s) and one for the number of
// requests being executed.
type RatioedGaugePair struct {
	// RequestsWaiting is given observations of the number of currently queued requests
	RequestsWaiting RatioedGauge

	// RequestsExecuting is given observations of the number of requests currently executing
	RequestsExecuting RatioedGauge
}
