//go:build !windows
// +build !windows

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

package volume_test

import (
	"testing"

	. "k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestGetMetricsBlockInvalid(t *testing.T) {
	// TODO: Enable this on Windows once support VolumeMode=Block is added.
	metrics := NewMetricsBlock("")
	actual, err := metrics.GetMetrics()
	expected := &Metrics{}
	if !volumetest.MetricsEqualIgnoreTimestamp(actual, expected) {
		t.Errorf("Expected empty Metrics from uninitialized MetricsBlock, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics on uninitialized MetricsBlock, actual nil")
	}

	metrics = NewMetricsBlock("/nonexistent/device/node")
	actual, err = metrics.GetMetrics()
	if !volumetest.MetricsEqualIgnoreTimestamp(actual, expected) {
		t.Errorf("Expected empty Metrics from incorrectly initialized MetricsBlock, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics on incorrectly initialized MetricsBlock, actual nil")
	}
}
