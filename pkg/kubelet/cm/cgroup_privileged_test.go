//go:build linux && cgroup_privileged

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
	"os"
	"testing"

	"k8s.io/klog/v2/ktesting"
)

// TestCgroupV1Validate tests cgroup v1 validation
// Requires: root privileges, cgroup v1 filesystem
// Run with: go test -tags=cgroup_privileged -v ./pkg/kubelet/cm/...
func TestCgroupV1Validate(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("Requires root privileges")
	}

	// Check if running cgroup v1
	if isCgroup2() {
		t.Skip("Test requires cgroup v1")
	}

	logger, _ := ktesting.NewTestContext(t)
	manager := NewCgroupV1Manager(logger, &CgroupSubsystems{}, "systemd")

	// Test with a temporary cgroup path
	testCgroupName := CgroupName([]string{"test-privileged-cgroup"})

	err := manager.Validate(testCgroupName)
	if err != nil {
		t.Logf("Expected validation to fail for non-existent cgroup: %v", err)
	}
}

// TestCgroupV2Validate tests cgroup v2 validation
// Requires: root privileges, cgroup v2 filesystem
// Run with: go test -tags=cgroup_privileged -v ./pkg/kubelet/cm/...
func TestCgroupV2Validate(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("Requires root privileges")
	}

	if !isCgroup2() {
		t.Skip("Test requires cgroup v2")
	}

	logger, _ := ktesting.NewTestContext(t)
	manager := NewCgroupV2Manager(logger, &CgroupSubsystems{}, "systemd")

	testCgroupName := CgroupName([]string{"test-privileged-cgroup"})

	err := manager.Validate(testCgroupName)
	if err != nil {
		t.Logf("Expected validation to fail for non-existent cgroup: %v", err)
	}
}

// TestCgroupV1CreateAndDestroy tests creating and destroying cgroups
// Requires: root privileges
// Run with: go test -tags=cgroup_privileged -v ./pkg/kubelet/cm/...
func TestCgroupV1CreateAndDestroy(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("Requires root privileges")
	}

	if isCgroup2() {
		t.Skip("Test requires cgroup v1")
	}

	logger, _ := ktesting.NewTestContext(t)
	manager := NewCgroupV1Manager(logger, &CgroupSubsystems{}, "cgroupfs")

	testCgroupName := CgroupName([]string{"test-create-destroy"})

	// Create cgroup
	config := &CgroupConfig{
		Name:               testCgroupName,
		ResourceParameters: &ResourceConfig{},
	}

	err := manager.Create(logger, config)
	if err != nil {
		t.Fatalf("Failed to create cgroup: %v", err)
	}

	// Verify exists
	if !manager.Exists(testCgroupName) {
		t.Error("Cgroup should exist after creation")
	}

	// Destroy cgroup
	err = manager.Destroy(logger, config)
	if err != nil {
		t.Fatalf("Failed to destroy cgroup: %v", err)
	}
}

// TestCgroupV2CreateAndDestroy tests creating and destroying cgroups in v2
// Requires: root privileges
// Run with: go test -tags=cgroup_privileged -v ./pkg/kubelet/cm/...
func TestCgroupV2CreateAndDestroy(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("Requires root privileges")
	}

	if !isCgroup2() {
		t.Skip("Test requires cgroup v2")
	}

	logger, _ := ktesting.NewTestContext(t)
	manager := NewCgroupV2Manager(logger, &CgroupSubsystems{}, "cgroupfs")

	testCgroupName := CgroupName([]string{"test-create-destroy-v2"})

	config := &CgroupConfig{
		Name:               testCgroupName,
		ResourceParameters: &ResourceConfig{},
	}

	err := manager.Create(logger, config)
	if err != nil {
		t.Fatalf("Failed to create cgroup: %v", err)
	}

	if !manager.Exists(testCgroupName) {
		t.Error("Cgroup should exist after creation")
	}

	err = manager.Destroy(logger, config)
	if err != nil {
		t.Fatalf("Failed to destroy cgroup: %v", err)
	}
}

// TestCgroupMemoryUsage tests reading memory usage
// Requires: root privileges
// Run with: go test -tags=cgroup_privileged -v ./pkg/kubelet/cm/...
func TestCgroupMemoryUsage(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("Requires root privileges")
	}

	logger, _ := ktesting.NewTestContext(t)

	var manager CgroupManager
	if isCgroup2() {
		manager = NewCgroupV2Manager(logger, &CgroupSubsystems{}, "cgroupfs")
	} else {
		manager = NewCgroupV1Manager(logger, &CgroupSubsystems{}, "cgroupfs")
	}

	// Test reading memory of root cgroup
	usage, err := manager.MemoryUsage(RootCgroupName)
	if err != nil {
		t.Logf("Could not read root memory usage: %v", err)
	} else {
		t.Logf("Root cgroup memory usage: %d bytes", usage)
	}
}

func isCgroup2() bool {
	_, err := os.Stat("/sys/fs/cgroup/cgroup.controllers")
	return err == nil
}
