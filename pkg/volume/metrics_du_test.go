/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

// TestMetricsDuGetCapacity tests that MetricsDu can read disk usage
// for path
func TestMetricsDuGetCapacity(t *testing.T) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "metrics_du_test")
	if err != nil {
		t.Fatalf("Can't make a tmp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	metrics := NewMetricsDu(tmpDir)

	actual, err := metrics.GetMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetMetrics %v", err)
	}
	if actual.Used.Value() != 4096 {
		t.Errorf("Expected Used %d for empty directory to be 4096.", actual.Used.Value())
	}

	// TODO(pwittroc): Figure out a way to test these values for correctness, maybe by formatting and mounting a file
	// as a filesystem
	if actual.Capacity.Value() <= 0 {
		t.Errorf("Expected Capacity %d to be greater than 0.", actual.Capacity.Value())
	}
	if actual.Available.Value() <= 0 {
		t.Errorf("Expected Available %d to be greater than 0.", actual.Available.Value())
	}

	// Write a file and expect Used to increase
	ioutil.WriteFile(filepath.Join(tmpDir, "f1"), []byte("Hello World"), os.ModeTemporary)
	actual, err = metrics.GetMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetMetrics %v", err)
	}
	if actual.Used.Value() != 8192 {
		t.Errorf("Unexpected Used for directory with file.  Expected 8192, was %d.", actual.Used.Value())
	}
}

// TestMetricsDuRequireInit tests that if MetricsDu is not initialized with a path, GetMetrics
// returns an error
func TestMetricsDuRequirePath(t *testing.T) {
	metrics := &metricsDu{}
	actual, err := metrics.GetMetrics()
	expected := &Metrics{}
	if *actual != *expected {
		t.Errorf("Expected empty Metrics from uninitialized MetricsDu, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics on uninitialized MetricsDu, actual nil")
	}
}

// TestMetricsDuRealDirectory tests that if MetricsDu is initialized to a non-existent path, GetMetrics
// returns an error
func TestMetricsDuRequireRealDirectory(t *testing.T) {
	metrics := NewMetricsDu("/not/a/real/directory")
	actual, err := metrics.GetMetrics()
	expected := &Metrics{}
	if *actual != *expected {
		t.Errorf("Expected empty Metrics from incorrectly initialized MetricsDu, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics on incorrectly initialized MetricsDu, actual nil")
	}
}
