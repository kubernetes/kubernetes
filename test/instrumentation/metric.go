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

package main

import (
	"k8s.io/component-base/metrics"
)

const (
	counterMetricType   = "Counter"
	gaugeMetricType     = "Gauge"
	histogramMetricType = "Histogram"
)

type metric struct {
	Name              string    `yaml:"name"`
	Subsystem         string    `yaml:"subsystem,omitempty"`
	Namespace         string    `yaml:"namespace,omitempty"`
	Help              string    `yaml:"help,omitempty"`
	Type              string    `yaml:"type,omitempty"`
	DeprecatedVersion string    `yaml:"deprecatedVersion,omitempty"`
	StabilityLevel    string    `yaml:"stabilityLevel,omitempty"`
	Labels            []string  `yaml:"labels,omitempty"`
	Buckets           []float64 `yaml:"buckets,omitempty"`
}

func (m metric) buildFQName() string {
	return metrics.BuildFQName(m.Namespace, m.Subsystem, m.Name)
}

type byFQName []metric

func (ms byFQName) Len() int { return len(ms) }
func (ms byFQName) Less(i, j int) bool {
	return ms[i].buildFQName() < ms[j].buildFQName()
}
func (ms byFQName) Swap(i, j int) {
	ms[i], ms[j] = ms[j], ms[i]
}
