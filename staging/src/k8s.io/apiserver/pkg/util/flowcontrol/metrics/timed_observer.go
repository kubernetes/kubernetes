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

// TimedObserver gets informed about the values assigned to a variable
// `X float64` over time, and reports on the ratio `X/X1`.
type TimedObserver interface {
	// Add notes a change to the variable
	Add(deltaX float64)

	// Set notes a setting of the variable
	Set(x float64)

	// SetX1 changes the value to use for X1
	SetX1(x1 float64)
}

// TimedObserverGenerator creates related observers that are
// differentiated by a series of label values
type TimedObserverGenerator interface {
	Generate(x, x1 float64, labelValues []string) TimedObserver
}

// TimedObserverPair is a corresponding pair of observers, one for the
// number of requests waiting in queue(s) and one for the number of
// requests being executed
type TimedObserverPair struct {
	// RequestsWaiting is given observations of the number of currently queued requests
	RequestsWaiting TimedObserver

	// RequestsExecuting is given observations of the number of requests currently executing
	RequestsExecuting TimedObserver
}

// TimedObserverPairGenerator generates pairs
type TimedObserverPairGenerator interface {
	Generate(waiting1, executing1 float64, labelValues []string) TimedObserverPair
}
