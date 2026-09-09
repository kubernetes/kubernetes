// Copyright 2024 Google Inc. All Rights Reserved.
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

package stats

// ResctrlManager is the resctrl analogue of Manager: it produces per-container
// resctrl Collectors. The full cAdvisor binary injects an implementation; the
// kubelet uses NoopResctrlManager (this library carries no resctrl code).
type ResctrlManager interface {
	Destroy()
	GetCollector(containerName string, getContainerPids func() ([]string, error), numberOfNUMANodes int) (Collector, error)
}

// NoopResctrlManager is the lean default: it collects nothing.
type NoopResctrlManager struct {
	NoopDestroy
}

func (n *NoopResctrlManager) GetCollector(containerName string, getContainerPids func() ([]string, error), numberOfNUMANodes int) (Collector, error) {
	return &NoopCollector{}, nil
}
