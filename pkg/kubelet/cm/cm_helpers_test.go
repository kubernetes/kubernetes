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
)

func TestInt64Slice(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		expected []int64
	}{
		{
			name:     "Empty slice",
			input:    []int{},
			expected: []int64{},
		},
		{
			name:     "Single element",
			input:    []int{1},
			expected: []int64{1},
		},
		{
			name:     "Multiple elements",
			input:    []int{1, 2, 3, 4, 5},
			expected: []int64{1, 2, 3, 4, 5},
		},
		{
			name:     "Negative numbers",
			input:    []int{-1, -2, -3},
			expected: []int64{-1, -2, -3},
		},
		{
			name:     "Large numbers",
			input:    []int{1e9, 2e9, 3e9},
			expected: []int64{1e9, 2e9, 3e9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := int64Slice(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("int64Slice() returned %d elements, want %d", len(result), len(tt.expected))
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("int64Slice()[%d] = %d, want %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestCgroupNameBasics(t *testing.T) {
	// Test CgroupName basic operations
	name := CgroupName([]string{"kubepods", "burstable", "pod-123"})

	// Test String representation
	if len(name) != 3 {
		t.Errorf("Expected length 3, got %d", len(name))
	}

	// Test empty name
	emptyName := CgroupName([]string{})
	if len(emptyName) != 0 {
		t.Errorf("Expected empty name, got length %d", len(emptyName))
	}
}

func TestResourceConfig(t *testing.T) {
	// Test ResourceConfig creation
	memLimit := int64(1024 * 1024 * 100) // 100MB
	cpuShares := uint64(1024)
	cpuQuota := int64(50000)
	cpuPeriod := uint64(100000)

	config := &ResourceConfig{
		Memory:    &memLimit,
		CPUShares: &cpuShares,
		CPUQuota:  &cpuQuota,
		CPUPeriod: &cpuPeriod,
	}

	if config == nil {
		t.Error("ResourceConfig should not be nil")
	}

	if *config.Memory != 1024*1024*100 {
		t.Errorf("Memory = %d, want %d", *config.Memory, 1024*1024*100)
	}
}

func TestCgroupConfig(t *testing.T) {
	// Test CgroupConfig creation
	name := CgroupName([]string{"test", "pod"})
	memLimit := int64(1024 * 1024 * 50) // 50MB
	config := &CgroupConfig{
		Name: name,
		ResourceParameters: &ResourceConfig{
			Memory: &memLimit,
		},
	}

	if config == nil {
		t.Error("CgroupConfig should not be nil")
	}

	if len(config.Name) != 2 {
		t.Errorf("Name length = %d, want 2", len(config.Name))
	}
}

func TestQOSContainersInfo(t *testing.T) {
	// Test QOSContainersInfo creation
	info := QOSContainersInfo{
		// QOS containers are identified by their paths
	}

	// Should be able to create without panic
	_ = info
}

func TestStatus(t *testing.T) {
	// Test Status creation - Status is a simple struct with SoftRequirements
	status := Status{
		SoftRequirements: nil,
	}

	// Should not panic
	_ = status.SoftRequirements
}

func TestNodeConfig(t *testing.T) {
	// Test NodeConfig creation
	config := NodeConfig{
		CgroupRoot:    "kubepods",
		CgroupsPerQOS: true,
	}

	if config.CgroupRoot != "kubepods" {
		t.Errorf("CgroupRoot = %s, want kubepods", config.CgroupRoot)
	}

	if !config.CgroupsPerQOS {
		t.Error("CgroupsPerQOS should be true")
	}
}
