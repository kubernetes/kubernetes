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
	"k8s.io/kubernetes/pkg/volume"
)

// LogMetricsService defines an interface for providing LogMetrics functionality.
type LogMetricsService interface {
	createLogMetricsProvider(path string) volume.MetricsProvider
}

type logMetrics struct{}

// NewLogMetricsService returns a new LogMetricsService type struct.
func NewLogMetricsService() LogMetricsService {
	return logMetrics{}
}

func (l logMetrics) createLogMetricsProvider(path string) volume.MetricsProvider {
	return volume.NewMetricsDu(path)
}
