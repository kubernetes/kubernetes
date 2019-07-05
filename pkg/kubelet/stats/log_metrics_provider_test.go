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

package stats

import (
	"fmt"

	"k8s.io/kubernetes/pkg/volume"
)

type fakeLogMetrics struct {
	fakeStats map[string]*volume.Metrics
}

func NewFakeLogMetricsService(stats map[string]*volume.Metrics) LogMetricsService {
	return &fakeLogMetrics{fakeStats: stats}
}

func (l *fakeLogMetrics) createLogMetricsProvider(path string) volume.MetricsProvider {
	return NewFakeMetricsDu(path, l.fakeStats[path])
}

type fakeMetricsDu struct {
	fakeStats *volume.Metrics
}

func NewFakeMetricsDu(path string, stats *volume.Metrics) volume.MetricsProvider {
	return &fakeMetricsDu{fakeStats: stats}
}

func (f *fakeMetricsDu) GetMetrics() (*volume.Metrics, error) {
	if f.fakeStats == nil {
		return nil, fmt.Errorf("no stats provided")
	}
	return f.fakeStats, nil
}
