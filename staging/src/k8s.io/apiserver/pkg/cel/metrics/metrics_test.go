/*
Copyright 2022 The Kubernetes Authors.

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
	"math"
	"testing"
	"time"

	"k8s.io/component-base/metrics/legacyregistry"
)

func TestObserveCompilation(t *testing.T) {
	defer legacyregistry.Reset()
	Metrics.ObserveCompilation(2 * time.Second)
	c, s := gatherHistogram(t, "apiserver_cel_compilation_duration_seconds")
	if c != 1 {
		t.Errorf("unexpected count: %v", c)
	}
	if math.Abs(s-2.0) > 1e-7 {
		t.Fatalf("incorrect sum: %v", s)
	}
}

func TestObserveEvaluation(t *testing.T) {
	defer legacyregistry.Reset()
	Metrics.ObserveEvaluation(2 * time.Second)
	c, s := gatherHistogram(t, "apiserver_cel_evaluation_duration_seconds")
	if c != 1 {
		t.Errorf("unexpected count: %v", c)
	}
	if math.Abs(s-2.0) > 1e-7 {
		t.Fatalf("incorrect sum: %v", s)
	}
}

func gatherHistogram(t *testing.T, name string) (count uint64, sum float64) {
	metrics, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("Failed to gather metrics: %s", err)
	}
	for _, mf := range metrics {
		if mf.GetName() == name {
			for _, m := range mf.GetMetric() {
				h := m.GetHistogram()
				count += h.GetSampleCount()
				sum += h.GetSampleSum()
			}
			return
		}
	}
	t.Fatalf("metric not found: %v", name)
	return 0, 0
}
