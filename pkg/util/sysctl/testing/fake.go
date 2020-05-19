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

package testing

import (
	"os"

	"k8s.io/kubernetes/pkg/util/sysctl"
)

// Fake is a map-backed implementation of sysctl.Interface, for testing/mocking.
type Fake struct {
	Settings map[string]int
}

// NewFake creates a fake sysctl implementation.
func NewFake() *Fake {
	return &Fake{
		Settings: make(map[string]int),
	}
}

// GetSysctl returns the value for the specified sysctl setting.
func (m *Fake) GetSysctl(sysctl string) (int, error) {
	v, found := m.Settings[sysctl]
	if !found {
		return -1, os.ErrNotExist
	}
	return v, nil
}

// SetSysctl modifies the specified sysctl flag to the new value.
func (m *Fake) SetSysctl(sysctl string, newVal int) error {
	m.Settings[sysctl] = newVal
	return nil
}

var _ = sysctl.Interface(&Fake{})
