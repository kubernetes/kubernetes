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

	"k8s.io/utils/ptr"
)

func TestResourceConfig(t *testing.T) {
	// Test ResourceConfig creation
	memLimit := int64(1024 * 1024 * 100) // 100MB
	cpuShares := uint64(1024)
	cpuQuota := int64(50000)
	cpuPeriod := uint64(100000)

	config := &ResourceConfig{
		Memory:    ptr.To(memLimit),
		CPUShares: ptr.To(cpuShares),
		CPUQuota:  ptr.To(cpuQuota),
		CPUPeriod: ptr.To(cpuPeriod),
	}

	if config == nil {
		t.Error("ResourceConfig should not be nil")
	}
	if *config.Memory != 1024*1024*100 {
		t.Errorf("Memory = %d, want %d", *config.Memory, 1024*1024*100)
	}
	if *config.CPUShares != 1024 {
		t.Errorf("CPUShares = %d, want %d", *config.CPUShares, 1024)
	}
	if *config.CPUQuota != 50000 {
		t.Errorf("CPUQuota = %d, want %d", *config.CPUQuota, 50000)
	}
	if *config.CPUPeriod != 100000 {
		t.Errorf("CPUPeriod = %d, want %d", *config.CPUPeriod, 100000)
	}
}

func TestCgroupConfig(t *testing.T) {
	name := CgroupName([]string{"test", "pod"})
	memLimit := int64(1024 * 1024 * 50)
	config := &CgroupConfig{
		Name: name,
		ResourceParameters: &ResourceConfig{
			Memory: ptr.To(memLimit),
		},
	}

	if config == nil {
		t.Error("CgroupConfig should not be nil")
	}

	if len(config.Name) != 2 {
		t.Errorf("Name length = %d, want 2", len(config.Name))
	}
}
