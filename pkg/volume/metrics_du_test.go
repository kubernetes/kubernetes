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

// TestMetricsDuGetCapacity tests that an initialized MetricsDu can read disk usage
// for path
func TestMetricsDuGetCapacity(t *testing.T) {
	metrics := &MetricsDu{}

	tmpDir, err := ioutil.TempDir(os.TempDir(), "metrics_du_test")
	if err != nil {
		t.Fatalf("Can't make a tmp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	metrics.InitMetricsDu(tmpDir)

	actual, err := metrics.GetCapacityMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetCapacityMetrics %v", err)
	}
	if actual.VolumeBytesUsed != 4096 {
		t.Errorf("Expected VolumeBytesUsed %d for empty directory to be 4096.", actual.VolumeBytesUsed)
	}

	// TODO(pwittroc): Figure out a way to test these values for correctness, maybe by formatting and mounting a file
	// as a filesystem
	if actual.FileSystemBytesUsed <= 0 {
		t.Errorf("Expected FileSystemBytesUsed %d to be greater than 0.", actual.FileSystemBytesUsed)
	}
	if actual.FileSystemBytesAvailable <= 0 {
		t.Errorf("Expected FileSystemBytesAvailable %d to be greater than 0.", actual.FileSystemBytesAvailable)
	}
	if actual.FileSystemBytesAvailable <= actual.FileSystemBytesUsed {
		t.Errorf("Expected FileSystemBytesAvailable %d to be greater than FileSystemBytesUsed %d.",
			actual.FileSystemBytesAvailable, actual.FileSystemBytesUsed)
	}

	// Write a file and expect the VolumeBytesUsed to increase
	ioutil.WriteFile(filepath.Join(tmpDir, "f1"), []byte("Hello World"), os.ModeTemporary)
	actual, err = metrics.GetCapacityMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetCapacityMetrics %v", err)
	}
	if actual.VolumeBytesUsed != 8192 {
		t.Errorf("Unexpected VolumeBytesUsed for directory with file.  Expected 8192, was %d.", actual.VolumeBytesUsed)
	}
}

// TestMetricsDuRequireInit tests that if MetricsDu is not initialized with a path, GetCapacityMetrics
// returns and error
func TestMetricsDuRequireInit(t *testing.T) {
	metrics := &MetricsDu{}
	actual, err := metrics.GetCapacityMetrics()
	expected := &CapacityMetrics{}
	if *actual != *expected {
		t.Errorf("Expected empty CapacityMetrics from uninitialized MetricsDu, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetCapacityMetrics on uninitialized MetricsDu, actual nil")
	}
}
