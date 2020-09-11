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

import (
	"github.com/blang/semver"

	promext "k8s.io/component-base/metrics/prometheusextension"
)

// SamplingHistogram is our internal representation for our wrapping struct around prometheus
// sampling histograms. SamplingHistogram implements both kubeCollector and SimpleGaugeMetric
type SamplingHistogram struct {
	SimpleGaugeMetric
	*SamplingHistogramOpts
	lazyMetric
	selfCollector
}

var _ kubeCollector = &SamplingHistogram{}
var _ SimpleGaugeMetric = &SamplingHistogram{}

// NewSamplingHistogram returns an object which is SamplingHistogram-like. However, nothing
// will be measured until the histogram is registered somewhere.
func NewSamplingHistogram(opts *SamplingHistogramOpts) *SamplingHistogram {
	opts.StabilityLevel.setDefaults()

	h := &SamplingHistogram{
		SamplingHistogramOpts: opts,
		lazyMetric:            lazyMetric{},
	}
	h.setPrometheusSamplingHistogram(noopMetric{})
	h.lazyInit(h, BuildFQName(opts.Namespace, opts.Subsystem, opts.Name))
	return h
}

// setPrometheusSamplingHistogram sets the underlying KubeGauge object, i.e. the thing that does the measurement.
func (h *SamplingHistogram) setPrometheusSamplingHistogram(histogram promext.SamplingHistogram) {
	h.SimpleGaugeMetric = histogram
	h.initSelfCollection(histogram)
}

// DeprecatedVersion returns a pointer to the Version or nil
func (h *SamplingHistogram) DeprecatedVersion() *semver.Version {
	return parseSemver(h.SamplingHistogramOpts.DeprecatedVersion)
}

// initializeMetric invokes the actual prometheus.SamplingHistogram object instantiation
// and stores a reference to it
func (h *SamplingHistogram) initializeMetric() {
	h.SamplingHistogramOpts.annotateStabilityLevel()
	// this actually creates the underlying prometheus gauge.
	sh, _ := promext.NewSamplingHistogram(h.SamplingHistogramOpts.toPromSamplingHistogramOpts())
	h.setPrometheusSamplingHistogram(sh)
}

// initializeDeprecatedMetric invokes the actual prometheus.SamplingHistogram object instantiation
// but modifies the Help description prior to object instantiation.
func (h *SamplingHistogram) initializeDeprecatedMetric() {
	h.SamplingHistogramOpts.markDeprecated()
	h.initializeMetric()
}
