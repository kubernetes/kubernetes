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

// ResettableSummary is a Summary metric, which can
// be reset.
type ResettableSummary interface {
	prometheus.Summary
	Resettable
}

type defaultResettableSummary struct {
	prometheus.Summary
	opts prometheus.SummaryOpts
}

// Reset the histogram.
func (c *defaultResettableSummary) Reset() {
	c.Summary = prometheus.NewSummary(c.opts)
}

// NewResettableSummary creates a new instance of ResettableSummary
// for a specific SummaryOpts.
func NewResettableSummary(opts prometheus.SummaryOpts) ResettableSummary {
	return &defaultResettableSummary{
		Summary: prometheus.NewSummary(opts),
		opts:    opts,
	}
}
