/*
Copyright 2015 The Kubernetes Authors.

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

import "k8s.io/component-base/metrics/testutil"

// SchedulerMetrics is metrics for scheduler
type SchedulerMetrics testutil.Metrics

// Equal returns true if all metrics are the same as the arguments.
func (m *SchedulerMetrics) Equal(o SchedulerMetrics) bool {
	return (*testutil.Metrics)(m).Equal(testutil.Metrics(o))
}

func newSchedulerMetrics() SchedulerMetrics {
	result := testutil.NewMetrics()
	return SchedulerMetrics(result)
}

func parseSchedulerMetrics(data string) (SchedulerMetrics, error) {
	result := newSchedulerMetrics()
	if err := testutil.ParseMetrics(data, (*testutil.Metrics)(&result)); err != nil {
		return SchedulerMetrics{}, err
	}
	return result, nil
}
