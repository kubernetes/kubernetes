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

// Handling statistics that are fully controlled in cAdvisor
package stats

import info "github.com/google/cadvisor/info/v1"

// This is supposed to store global state about an cAdvisor metrics collector.
// cAdvisor manager will call Destroy() when it stops.
// For each container detected by the cAdvisor manager, it will call
// GetCollector() with the devices cgroup path for that container.
// GetCollector() is supposed to return an object that can update
// external stats for that container.
type Manager interface {
	Destroy()
	GetCollector(deviceCgroup string) (Collector, error)
}

// Collector can update ContainerStats by adding more metrics.
type Collector interface {
	Destroy()
	UpdateStats(*info.ContainerStats) error
}
