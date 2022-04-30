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

const (
	LabelNamePhase      = "phase"
	LabelValueWaiting   = "waiting"
	LabelValueExecuting = "executing"
)

// Observer is something that can be given numeric observations.
// TODO: Change the names here to use "Gauge" in place of "Observer", to
// follow conventions in lower layers, where "Observer" implies sampling and
// "Gauge" is for a variable (all values seen, no sampling).
type Observer interface {
	// Observe takes an observation
	Observe(float64)
}

// RatioedObserver tracks ratios.
// The numerator is set through the Observer methods,
// and the denominator can be updated through the SetDenominator method.
type RatioedObserver interface {
	Observer

	// SetDenominator sets the denominator to use until it is changed again
	SetDenominator(float64)
}

// RatioedbserverVec creates related observers that are
// differentiated by a series of label values.
type RatioedObserverVec interface {
	// WithLabelValues will return an error if this vec is not hidden and not yet registered
	// or there is a syntactic problem with the labelValues.
	WithLabelValuesChecked(initialDenominator float64, labelValues ...string) (RatioedObserver, error)

	// WithLabelValuesSafe is the same as WithLabelValuesChecked in cases where that does not
	// return an error.  When the unsafe version returns an error due to the vector not being
	// registered yet, the safe version returns an object that implements its methods
	// by first calling WithLabelValuesChecked and then delegating to whatever observer that returns.
	// In the other error cases the object returned here is a noop.
	WithLabelValuesSafe(initialDenominator float64, labelValues ...string) RatioedObserver
}

// RatioedObserverPair is a corresponding pair of observers, one for the
// number of requests waiting in queue(s) and one for the number of
// requests being executed
type RatioedObserverPair struct {
	// RequestsWaiting is given observations of the number of currently queued requests
	RequestsWaiting RatioedObserver

	// RequestsExecuting is given observations of the number of requests currently executing
	RequestsExecuting RatioedObserver
}

// RatioedObserverPairVec generates pairs
type RatioedObserverPairVec interface {
	// WithLabelValues will return an error if this pair is not hidden and not yet registered
	// or there is a syntactic problem with the labelValues.
	WithLabelValuesChecked(initialWaitingDenominator, initialExecutingDenominator float64, labelValues ...string) (RatioedObserverPair, error)

	// WithLabelValuesSafe uses the safe version of each pair member.
	WithLabelValuesSafe(initialWaitingDenominator, initialExecutingDenominator float64, labelValues ...string) RatioedObserverPair
}
