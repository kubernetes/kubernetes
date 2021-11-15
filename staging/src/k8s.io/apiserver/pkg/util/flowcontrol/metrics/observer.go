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

// Observer is something that can be given numeric observations.
type Observer interface {
	// Observe takes an observation
	Observe(float64)
}

//  ChangeObserver extends Observer with the ability to take
// an observation that is relative to the previous observation.
type ChangeObserver interface {
	Observer

	// Observe a new value that differs by the given amount from the previous observation.
	Add(float64)
}

// RatioedChangeObserver tracks ratios.
// The numerator is set/changed through the ChangeObserver methods,
// and the denominator can be updated through the SetDenominator method.
// A ratio is tracked whenever the numerator is set/changed.
type RatioedChangeObserver interface {
	ChangeObserver

	// SetDenominator sets the denominator to use until it is changed again
	SetDenominator(float64)
}

// RatioedChangeObserverGenerator creates related observers that are
// differentiated by a series of label values
type RatioedChangeObserverGenerator interface {
	Generate(initialNumerator, initialDenominator float64, labelValues []string) RatioedChangeObserver
}

// RatioedChangeObserverPair is a corresponding pair of observers, one for the
// number of requests waiting in queue(s) and one for the number of
// requests being executed
type RatioedChangeObserverPair struct {
	// RequestsWaiting is given observations of the number of currently queued requests
	RequestsWaiting RatioedChangeObserver

	// RequestsExecuting is given observations of the number of requests currently executing
	RequestsExecuting RatioedChangeObserver
}

// RatioedChangeObserverPairGenerator generates pairs
type RatioedChangeObserverPairGenerator interface {
	Generate(initialWaitingDenominator, initialExecutingDenominator float64, labelValues []string) RatioedChangeObserverPair
}
