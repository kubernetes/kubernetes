/*
Copyright 2021 The Kubernetes Authors.

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

// SnapshotControllerMetrics is metrics for controller manager
type SnapshotControllerMetrics testutil.Metrics

// Equal returns true if all metrics are the same as the arguments.
func (m *SnapshotControllerMetrics) Equal(o SnapshotControllerMetrics) bool {
	return (*testutil.Metrics)(m).Equal(testutil.Metrics(o))
}

func newSnapshotControllerMetrics() SnapshotControllerMetrics {
	result := testutil.NewMetrics()
	return SnapshotControllerMetrics(result)
}

func parseSnapshotControllerMetrics(data string) (SnapshotControllerMetrics, error) {
	result := newSnapshotControllerMetrics()
	if err := testutil.ParseMetrics(data, (*testutil.Metrics)(&result)); err != nil {
		return SnapshotControllerMetrics{}, err
	}
	return result, nil
}
