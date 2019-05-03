// +build !linux

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

package cm

import "fmt"

type unsupportedCgroupManager struct{}

// Make sure that unsupportedCgroupManager implements the CgroupManager interface
var _ CgroupManager = &unsupportedCgroupManager{}

// CgroupSubsystems holds information about the mounted cgroup subsystems.
type CgroupSubsystems struct {
	Mounts      []interface{}
	MountPoints map[string]string
}

// NewCgroupManager is a factory method that returns a CgroupManager for unsupported systems.
func NewCgroupManager(_ interface{}) CgroupManager {
	return &unsupportedCgroupManager{}
}

// Name implements CgroupManager.Name.
func (m *unsupportedCgroupManager) Name(_ CgroupName) string {
	return ""
}

// Exists implements CgroupManager.Exists.
func (m *unsupportedCgroupManager) Exists(_ CgroupName) bool {
	return false
}

// Destroy implements CgroupManager.Destroy.
func (m *unsupportedCgroupManager) Destroy(_ *CgroupConfig) error {
	return nil
}

// Update implements CgroupManager.Update.
func (m *unsupportedCgroupManager) Update(_ *CgroupConfig) error {
	return nil
}

// Create implements CgroupManager.Create.
func (m *unsupportedCgroupManager) Create(_ *CgroupConfig) error {
	return fmt.Errorf("Cgroup Manager is not supported in this build")
}

// GetResourceStats implements CgroupManager.GetResourceStats.
func (m *unsupportedCgroupManager) GetResourceStats(name CgroupName) (*ResourceStats, error) {
	return nil, fmt.Errorf("Cgroup Manager is not supported in this build")
}

// PIDs implements CgroupManager.PIDs.
func (m *unsupportedCgroupManager) PIDs(_ CgroupName) []int {
	return nil
}

// CgroupName implements CgroupManager.CgroupName.
func (m *unsupportedCgroupManager) CgroupName(name string) CgroupName {
	return CgroupName([]string{})
}

// ReduceCPULimits implements CgroupManager.ReduceCPULimits.
func (m *unsupportedCgroupManager) ReduceCPULimits(cgroupName CgroupName) error {
	return nil
}

// NewCgroupName composes a new cgroup name.
// This implementation doesn't perform any validation.
func NewCgroupName(base CgroupName, components ...string) CgroupName {
	return CgroupName(append(base, components...))
}

// ToSystemd is a stub implementation.
func (cgroupName CgroupName) ToSystemd() string {
	return ""
}

// ParseSystemdToCgroupName is a stub implementation.
func ParseSystemdToCgroupName(name string) CgroupName {
	return nil
}

// ToCgroupfs is a stub implementation.
func (cgroupName CgroupName) ToCgroupfs() string {
	return ""
}

// ParseCgroupfsToCgroupName is a stub implementation.
func ParseCgroupfsToCgroupName(name string) CgroupName {
	return nil
}

// IsSystemdStyleName is a stub implementation.
func IsSystemdStyleName(name string) bool {
	return false
}
