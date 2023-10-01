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
	"k8s.io/kubernetes/pkg/util/libcontainer"

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

func (m *unsupportedCgroupManager) Name(_ libcontainer.CgroupName) string {
	return ""
}

func (m *unsupportedCgroupManager) Validate(_ libcontainer.CgroupName) error {
	return errNotSupported
}

func (m *unsupportedCgroupManager) Exists(_ libcontainer.CgroupName) bool {
	return false
}

func (m *unsupportedCgroupManager) Destroy(_ *libcontainer.CgroupConfig) error {
	return nil
}

func (m *unsupportedCgroupManager) Update(_ *libcontainer.CgroupConfig) error {
	return nil
}

func (m *unsupportedCgroupManager) Create(_ *libcontainer.CgroupConfig) error {
	return errNotSupported
}

func (m *unsupportedCgroupManager) MemoryUsage(_ libcontainer.CgroupName) (int64, error) {
	return -1, errNotSupported
}

func (m *unsupportedCgroupManager) Pids(_ libcontainer.CgroupName) []int {
	return nil
}

func (m *unsupportedCgroupManager) CgroupName(name string) libcontainer.CgroupName {
	return libcontainer.CgroupName([]string{})
}

func (m *unsupportedCgroupManager) ReduceCPULimits(cgroupName libcontainer.CgroupName) error {
	return nil
}

func (m *unsupportedCgroupManager) GetCgroupConfig(name libcontainer.CgroupName, resource v1.ResourceName) (*libcontainer.ResourceConfig, error) {
	return nil, errNotSupported
}

func (m *unsupportedCgroupManager) SetCgroupConfig(name libcontainer.CgroupName, resource v1.ResourceName, resourceConfig *libcontainer.ResourceConfig) error {
	return errNotSupported
}

var RootCgroupName = libcontainer.CgroupName([]string{})

func NewCgroupName(base libcontainer.CgroupName, components ...string) libcontainer.CgroupName {
	return append(append([]string{}, base...), components...)
}

func (cgroupName libcontainer.CgroupName) ToSystemd() string {
	return ""
}

func ParseSystemdToCgroupName(name string) libcontainer.CgroupName {
	return nil
}

func (cgroupName libcontainer.CgroupName) ToCgroupfs() string {
	return ""
}

func ParseCgroupfsToCgroupName(name string) libcontainer.CgroupName {
	return nil
}

func IsSystemdStyleName(name string) bool {
	return false
}
