/*
Copyright 2018 The Kubernetes Authors.

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

import "github.com/prometheus/client_golang/prometheus"

// ResettableCounter is a Counter metric, which can
// be reset.
type ResettableCounter interface {
	prometheus.Counter
	Resettable
}

type defaultResettableCounter struct {
	prometheus.Counter
	opts prometheus.CounterOpts
}

// Reset sets the counter to 0.
func (c *defaultResettableCounter) Reset() {
	c.Counter = prometheus.NewCounter(c.opts)
}

// NewResettableCounter creates a new instance of ResettableCounter
// for a specific CounterOpts.
func NewResettableCounter(opts prometheus.CounterOpts) ResettableCounter {
	return &defaultResettableCounter{
		Counter: prometheus.NewCounter(opts),
		opts:    opts,
	}
}
