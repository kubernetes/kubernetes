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

// ResettableHistogram is a Histogram metric, which can
// be reset.
type ResettableHistogram interface {
	prometheus.Histogram
	Resettable
}

type defaultResHistogram struct {
	prometheus.Histogram
	opts prometheus.HistogramOpts
}

// Reset the histogram.
func (c *defaultResHistogram) Reset() {
	c.Histogram = prometheus.NewHistogram(c.opts)
}

// NewResettableHistogram creates a new instance of ResettableHistogram
// for a specific HistogramOpts.
func NewResettableHistogram(opts prometheus.HistogramOpts) ResettableHistogram {
	return &defaultResHistogram{
		Histogram: prometheus.NewHistogram(opts),
		opts:      opts,
	}
}
