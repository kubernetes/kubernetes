//go:build linux

/*
Copyright 2025 The Kubernetes Authors.

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

package cm

import (
	"testing"

	"k8s.io/klog/v2/ktesting"
)

func TestCgroupV1ManagerVersion(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Create a cgroup v1 manager
	manager := NewCgroupV1Manager(logger, &CgroupSubsystems{}, "systemd")

	// Verify version returns 1
	if manager.Version() != 1 {
		t.Errorf("Expected version 1, got %d", manager.Version())
	}
}

func TestCgroupV2ManagerVersion(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	// Create a cgroup v2 manager
	manager := NewCgroupV2Manager(logger, &CgroupSubsystems{}, "systemd")

	// Verify version returns 2
	if manager.Version() != 2 {
		t.Errorf("Expected version 2, got %d", manager.Version())
	}
}
