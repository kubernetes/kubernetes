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
	"io/fs"
	"os"
	"runtime"
	"testing"

	. "k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestGetMetricsBlockInvalid(t *testing.T) {
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

func TestGetMetricsBlock(t *testing.T) {
	// FIXME: this test is Linux specific
	if runtime.GOOS == "windows" {
		t.Skip("Block device detection is Linux specific, no Windows support")
	}

	// find a block device
	// get all available block devices
	//  - ls /sys/block
	devices, err := os.ReadDir("/dev")
	if err != nil {
		t.Skipf("Could not read devices from /dev: %v", err)
	} else if len(devices) == 0 {
		t.Skip("No devices found")
	}

	// for each device, check if it is available in /dev
	devNode := ""
	var stat fs.FileInfo
	for _, device := range devices {
		// if the device exists, use it, return
		devNode = "/dev/" + device.Name()
		stat, err = os.Stat(devNode)
		if err == nil {
			if stat.Mode().Type() == fs.ModeDevice {
				break
			}
		}
		// set to an empty string, so we can do validation of the last
		// device too
		devNode = ""
	}

	// if no devices are found, or none exists in /dev, skip this part
	if devNode == "" {
		t.Skip("Could not find a block device under /dev")
	}

	// when we get here, devNode points to an existing block device
	metrics := NewMetricsBlock(devNode)
	actual, err := metrics.GetMetrics()
	if err != nil {
		t.Errorf("Unexpected error when calling GetMetrics: %v", err)
	}

	if a := actual.Capacity.Value(); a <= 0 {
		t.Errorf("Expected Capacity %d to be greater than 0.", a)
	}
}
