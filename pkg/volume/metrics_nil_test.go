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

package volume

import (
	"testing"
)

func TestMetricsNilSupportsMetrics(t *testing.T) {
	metrics := &MetricsNil{}
	supported := metrics.SupportsMetrics()
	if supported {
		t.Error("Expected no support for metrics")
	}
}

func TestMetricsNilGetCapacity(t *testing.T) {
	metrics := &MetricsNil{}
	actual, err := metrics.GetMetrics()
	expected := &Metrics{}
	if *actual != *expected {
		t.Errorf("Expected empty Metrics, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics, actual nil")
	}
}
