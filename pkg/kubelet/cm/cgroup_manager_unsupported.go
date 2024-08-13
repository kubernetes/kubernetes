//go:build !linux
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

import (
	"errors"

	v1 "k8s.io/api/core/v1"
)

type unsupportedCgroupManager struct{}

var errNotSupported = errors.New("Cgroup Manager is not supported in this build")

// Make sure that unsupportedCgroupManager implements the CgroupManager interface
var _ CgroupManager = &unsupportedCgroupManager{}

type CgroupSubsystems struct {
	Mounts      []interface{}
	MountPoints map[string]string
}

func NewCgroupManager(_ interface{}) CgroupManager {
	return &unsupportedCgroupManager{}
}

func (m *unsupportedCgroupManager) Version() int {
	return 0
}

func (m *unsupportedCgroupManager) Name(_ CgroupName) string {
	return ""
}

func (m *unsupportedCgroupManager) Validate(_ CgroupName) error {
	return errNotSupported
}

func (m *unsupportedCgroupManager) Exists(_ CgroupName) bool {
	return false
}

func (m *unsupportedCgroupManager) Destroy(_ *CgroupConfig) error {
	return nil
}

func (m *unsupportedCgroupManager) Update(_ *CgroupConfig) error {
	return nil
}

func (m *unsupportedCgroupManager) Create(_ *CgroupConfig) error {
	return errNotSupported
}

func (m *unsupportedCgroupManager) MemoryUsage(_ CgroupName) (int64, error) {
	return -1, errNotSupported
}

func (m *unsupportedCgroupManager) Pids(_ CgroupName) []int {
	return nil
}

func (m *unsupportedCgroupManager) CgroupName(name string) CgroupName {
	return CgroupName([]string{})
}

func (m *unsupportedCgroupManager) ReduceCPULimits(cgroupName CgroupName) error {
	return nil
}

func (m *unsupportedCgroupManager) GetCgroupConfig(name CgroupName, resource v1.ResourceName) (*ResourceConfig, error) {
	return nil, errNotSupported
}

func (m *unsupportedCgroupManager) SetCgroupConfig(name CgroupName, resource v1.ResourceName, resourceConfig *ResourceConfig) error {
	return errNotSupported
}

var RootCgroupName = CgroupName([]string{})

func NewCgroupName(base CgroupName, components ...string) CgroupName {
	return append(append([]string{}, base...), components...)
}

func (cgroupName CgroupName) ToSystemd() string {
	return ""
}

func ParseSystemdToCgroupName(name string) CgroupName {
	return nil
}

func (cgroupName CgroupName) ToCgroupfs() string {
	return ""
}

func ParseCgroupfsToCgroupName(name string) CgroupName {
	return nil
}

func IsSystemdStyleName(name string) bool {
	return false
}
