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

package framework

import (
	"fmt"
	"time"
)

const (
	resizeNodeReadyTimeout    = 2 * time.Minute
	resizeNodeNotReadyTimeout = 2 * time.Minute
)

func ResizeGroup(group string, size int32) error {
	if TestContext.ReportDir != "" {
		CoreDump(TestContext.ReportDir)
		defer CoreDump(TestContext.ReportDir)
	}
	return TestContext.CloudConfig.Provider.ResizeGroup(group, size)
}

func GetGroupNodes(group string) ([]string, error) {
	return TestContext.CloudConfig.Provider.GetGroupNodes(group)
}

func GroupSize(group string) (int, error) {
	return TestContext.CloudConfig.Provider.GroupSize(group)
}

func WaitForGroupSize(group string, size int32) error {
	timeout := 30 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		currentSize, err := GroupSize(group)
		if err != nil {
			Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != int(size) {
			Logf("Waiting for node instance group size %d, current size %d", size, currentSize)
			continue
		}
		Logf("Node instance group has reached the desired size %d", size)
		return nil
	}
	return fmt.Errorf("timeout waiting %v for node instance group size to be %d", timeout, size)
}
