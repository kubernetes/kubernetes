//go:build linux
// +build linux

// Copyright 2021 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Manager of resctrl for containers.
package resctrl

import (
	"errors"
	"time"

	"k8s.io/klog/v2"

	"github.com/google/cadvisor/container/raw"
	"github.com/google/cadvisor/stats"
)

type Manager interface {
	Destroy()
	GetCollector(containerName string, getContainerPids func() ([]string, error), numberOfNUMANodes int) (stats.Collector, error)
}

type manager struct {
	stats.NoopDestroy
	interval        time.Duration
	vendorID        string
	inHostNamespace bool
}

func (m *manager) GetCollector(containerName string, getContainerPids func() ([]string, error), numberOfNUMANodes int) (stats.Collector, error) {
	collector := newCollector(containerName, getContainerPids, m.interval, numberOfNUMANodes, m.vendorID, m.inHostNamespace)
	err := collector.setup()
	if err != nil {
		return &stats.NoopCollector{}, err
	}

	return collector, nil
}

func NewManager(interval time.Duration, setup func() error, vendorID string, inHostNamespace bool) (Manager, error) {
	err := setup()
	if err != nil {
		return &NoopManager{}, err
	}

	if !isResctrlInitialized {
		return &NoopManager{}, errors.New("the resctrl isn't initialized")
	}
	if !(enabledCMT || enabledMBM) {
		return &NoopManager{}, errors.New("there are no monitoring features available")
	}

	if !*raw.DockerOnly {
		klog.Warning("--docker_only should be set when collecting Resctrl metrics! See the runtime docs.")
	}

	return &manager{interval: interval, vendorID: vendorID, inHostNamespace: inHostNamespace}, nil
}

type NoopManager struct {
	stats.NoopDestroy
}

func (np *NoopManager) GetCollector(_ string, _ func() ([]string, error), _ int) (stats.Collector, error) {
	return &stats.NoopCollector{}, nil
}
