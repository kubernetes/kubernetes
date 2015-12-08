/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

var _ VolumeMetricsProvider = &MetricsDu{}

// MetricsDu represents a VolumeMetricsProvider that calculates the used and available
// Volume capacity by executing the "du" and "df" commands on the Volume path.
type MetricsDu struct {
	// the directory path the volume is mounted to.
	path string
}

// InitMetricsDu initializes MetricsDu with the Volume path.
// This is required before MetricsDu can be used.
func (cm *MetricsDu) InitMetricsDu(path string) {
	cm.path = path
}

// See VolumeMetricsProvider.GetCapacityMetrics
// GetCapacityMetrics calculates the volume usage and device free space by executing "du"
// and "df" on path.
func (cm *MetricsDu) GetCapacityMetrics() (*CapacityMetrics, error) {
	metrics := &CapacityMetrics{}
	if cm.path == "" {
		return metrics, errors.New("no path defined for disk usage capacity metrics.  Must call InitMetricsDu.")
	}

	err := cm.runDu(metrics)
	if err != nil {
		return metrics, err
	}

	err = cm.runDf(metrics)
	if err != nil {
		return metrics, err
	}

	return metrics, nil
}

// runDu executes the "du" command and writes the results to metrics.VolumeKbUsed
func (cm *MetricsDu) runDu(metrics *CapacityMetrics) error {
	// Uses the same niceness level as ccmvisor.fs does when running du
	// Use -k flag so results in are kilobytes
	out, err := exec.Command("nice", "-n", "19", "du", "-s", "-k", cm.path).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed command 'du' on %s with error %v", cm.path, err)
	}
	bUsed, err := strconv.ParseUint(strings.Fields(string(out))[0], 10, 64)
	if err != nil {
		return fmt.Errorf("failed to parse 'du' output %s due to error %v", out, err)
	}
	metrics.VolumeKbUsed = 1024 * bUsed
	return nil
}

// runDf executes the "df" command and writes the results to metrics.DeviceKbUsed
// and metrics.DeviceKbAvailable
// TODO(pwittrock): Consider using a statfs syscall directly to get the data
func (cm *MetricsDu) runDf(metrics *CapacityMetrics) error {
	// Uses the same niceness level as ccmvisor.fs does when running du
	// Use -k flag so results in are kilobytes
	out, err := exec.Command("nice", "-n", "19", "df", "-k", cm.path).CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed command 'df' on %s with error %v", cm.path, err)
	}
	lines := strings.Split(string(out), "\n")
	header := strings.Fields(lines[0])
	content := strings.Fields(lines[1])

	// Find the indexes into content for the fields we are interested in
	usedIndex := -1
	availableIndex := -1
	for i, v := range header {
		if v == "Used" {
			usedIndex = i
		}
		if v == "Available" {
			availableIndex = i
		}
	}

	// Parse "Used"
	if usedIndex >= 0 {
		bUsed, err := strconv.ParseUint(content[usedIndex], 10, 64)
		if err != nil {
			return fmt.Errorf("failed to parse 'df' 'Used' output %s due to error %v", out, err)
		}
		metrics.FileSystemKbUsed = 1024 * bUsed
	} else {
		return fmt.Errorf(`failed to find 'df' field 'Used' in output %s`, out)
	}

	// Parse "Available"
	if availableIndex >= 0 {
		bAvailable, err := strconv.ParseUint(content[availableIndex], 10, 64)
		if err != nil {
			return fmt.Errorf("failed to parse 'df' 'Available' output %s due to error %v", out, err)
		}
		metrics.FileSystemKbAvailable = 1024 * bAvailable
	} else {
		return fmt.Errorf("failed to find 'df' field 'Available' in output %s", out)
	}

	return nil
}
