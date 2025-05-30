// Copyright 2025 Google Inc. All Rights Reserved.
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

package resctrl

import (
	"fmt"
	"sync"
	"time"

	"github.com/google/cadvisor/stats"

	"k8s.io/klog/v2"
)

type ResControlManager interface {
	Destroy()
	GetCollector(containerName string, getContainerPids func() ([]string, error), numberOfNUMANodes int) (stats.Collector, error)
}

// All registered auth provider plugins.
var pluginsLock sync.Mutex
var plugins = make(map[string]ResControlManagerPlugin)

type ResControlManagerPlugin interface {
	NewManager(interval time.Duration, vendorID string, inHostNamespace bool) (ResControlManager, error)
}

func RegisterPlugin(name string, plugin ResControlManagerPlugin) error {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	if _, found := plugins[name]; found {
		return fmt.Errorf("ResControlManagerPlugin %q was registered twice", name)
	}
	klog.V(4).Infof("Registered ResControlManagerPlugin %q", name)
	plugins[name] = plugin
	return nil
}

func NewManager(interval time.Duration, vendorID string, inHostNamespace bool) (ResControlManager, error) {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	for _, plugin := range plugins {
		return plugin.NewManager(interval, vendorID, inHostNamespace)
	}
	return nil, fmt.Errorf("unable to find plugins for resctrl manager")
}
