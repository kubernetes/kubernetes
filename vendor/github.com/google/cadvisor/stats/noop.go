// Copyright 2020 Google Inc. All Rights Reserved.
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

// Noop perf Manager and Collector.
package stats

import (
	"k8s.io/klog/v2"

	v1 "github.com/google/cadvisor/info/v1"
)

type NoopManager struct {
	NoopDestroy
}

type NoopDestroy struct{}

func (nsd NoopDestroy) Destroy() {
	klog.V(5).Info("No-op Destroy function called")
}

func (m *NoopManager) GetCollector(cgroup string) (Collector, error) {
	return &NoopCollector{}, nil
}

type NoopCollector struct {
	NoopDestroy
}

func (c *NoopCollector) UpdateStats(stats *v1.ContainerStats) error {
	return nil
}
