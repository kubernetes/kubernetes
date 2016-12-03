/*
Copyright 2016 The Kubernetes Authors.

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
	"os"
	"testing"

	utiltesting "k8s.io/kubernetes/pkg/util/testing"
)

func TestGetMetricsStatFS(t *testing.T) {
	metrics := NewMetricsStatFS("")
	actual, err := metrics.GetMetrics()
	expected := &Metrics{}
	if *actual != *expected {
		t.Errorf("Expected empty Metrics from uninitialized MetricsStatFS, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics on uninitialized MetricsStatFS, actual nil")
	}

	metrics = NewMetricsStatFS("/not/a/real/directory")
	actual, err = metrics.GetMetrics()
	if *actual != *expected {
		t.Errorf("Expected empty Metrics from incorrectly initialized MetricsStatFS, actual %v", *actual)
	}
	if err == nil {
		t.Errorf("Expected error when calling GetMetrics on incorrectly initialized MetricsStatFS, actual nil")
	}

	tmpDir, err := utiltesting.MkTmpdir("metric_statfs_test")
	if err != nil {
		t.Fatalf("Can't make a tmp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	metrics = NewMetricsStatFS(tmpDir)
	actual, err = metrics.GetMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetMetrics %v", err)
	}

	if a := actual.Capacity.Value(); a <= 0 {
		t.Errorf("Expected Capacity %d to be greater than 0.", a)
	}
	if a := actual.Available.Value(); a <= 0 {
		t.Errorf("Expected Available %d to be greater than 0.", a)
	}

}
