// Copyright 2014 Google Inc. All Rights Reserved.
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

package container

import (
	"fmt"
	"sync"

	"github.com/google/cadvisor/fs"
	info "github.com/google/cadvisor/info/v1"
	"github.com/google/cadvisor/watcher"

	"k8s.io/klog"
)

type ContainerHandlerFactory interface {
	// Create a new ContainerHandler using this factory. CanHandleAndAccept() must have returned true.
	NewContainerHandler(name string, inHostNamespace bool) (c ContainerHandler, err error)

	// Returns whether this factory can handle and accept the specified container.
	CanHandleAndAccept(name string) (handle bool, accept bool, err error)

	// Name of the factory.
	String() string

	// Returns debugging information. Map of lines per category.
	DebugInfo() map[string][]string
}

// MetricKind represents the kind of metrics that cAdvisor exposes.
type MetricKind string

const (
	CpuUsageMetrics         MetricKind = "cpu"
	ProcessSchedulerMetrics MetricKind = "sched"
	PerCpuUsageMetrics      MetricKind = "percpu"
	MemoryUsageMetrics      MetricKind = "memory"
	CpuLoadMetrics          MetricKind = "cpuLoad"
	DiskIOMetrics           MetricKind = "diskIO"
	DiskUsageMetrics        MetricKind = "disk"
	NetworkUsageMetrics     MetricKind = "network"
	NetworkTcpUsageMetrics  MetricKind = "tcp"
	NetworkUdpUsageMetrics  MetricKind = "udp"
	AcceleratorUsageMetrics MetricKind = "accelerator"
	AppMetrics              MetricKind = "app"
	ProcessMetrics          MetricKind = "process"
)

func (mk MetricKind) String() string {
	return string(mk)
}

type MetricSet map[MetricKind]struct{}

func (ms MetricSet) Has(mk MetricKind) bool {
	_, exists := ms[mk]
	return exists
}

func (ms MetricSet) Add(mk MetricKind) {
	ms[mk] = struct{}{}
}

// All registered auth provider plugins.
var pluginsLock sync.Mutex
var plugins = make(map[string]Plugin)

type Plugin interface {
	// InitializeFSContext is invoked when populating an fs.Context object for a new manager.
	// A returned error here is fatal.
	InitializeFSContext(context *fs.Context) error

	// Register is invoked when starting a manager. It can optionally return a container watcher.
	// A returned error is logged, but is not fatal.
	Register(factory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics MetricSet) (watcher.ContainerWatcher, error)
}

func RegisterPlugin(name string, plugin Plugin) error {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	if _, found := plugins[name]; found {
		return fmt.Errorf("Plugin %q was registered twice", name)
	}
	klog.V(4).Infof("Registered Plugin %q", name)
	plugins[name] = plugin
	return nil
}

func InitializeFSContext(context *fs.Context) error {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()
	for name, plugin := range plugins {
		err := plugin.InitializeFSContext(context)
		if err != nil {
			klog.V(5).Infof("Initialization of the %s context failed: %v", name, err)
			return err
		}
	}
	return nil
}

func InitializePlugins(factory info.MachineInfoFactory, fsInfo fs.FsInfo, includedMetrics MetricSet) []watcher.ContainerWatcher {
	pluginsLock.Lock()
	defer pluginsLock.Unlock()

	containerWatchers := []watcher.ContainerWatcher{}
	for name, plugin := range plugins {
		watcher, err := plugin.Register(factory, fsInfo, includedMetrics)
		if err != nil {
			klog.V(5).Infof("Registration of the %s container factory failed: %v", name, err)
		}
		if watcher != nil {
			containerWatchers = append(containerWatchers, watcher)
		}
	}
	return containerWatchers
}

// TODO(vmarmol): Consider not making this global.
// Global list of factories.
var (
	factories     = map[watcher.ContainerWatchSource][]ContainerHandlerFactory{}
	factoriesLock sync.RWMutex
)

// Register a ContainerHandlerFactory. These should be registered from least general to most general
// as they will be asked in order whether they can handle a particular container.
func RegisterContainerHandlerFactory(factory ContainerHandlerFactory, watchTypes []watcher.ContainerWatchSource) {
	factoriesLock.Lock()
	defer factoriesLock.Unlock()

	for _, watchType := range watchTypes {
		factories[watchType] = append(factories[watchType], factory)
	}
}

// Returns whether there are any container handler factories registered.
func HasFactories() bool {
	factoriesLock.Lock()
	defer factoriesLock.Unlock()

	return len(factories) != 0
}

// Create a new ContainerHandler for the specified container.
func NewContainerHandler(name string, watchType watcher.ContainerWatchSource, inHostNamespace bool) (ContainerHandler, bool, error) {
	factoriesLock.RLock()
	defer factoriesLock.RUnlock()

	// Create the ContainerHandler with the first factory that supports it.
	for _, factory := range factories[watchType] {
		canHandle, canAccept, err := factory.CanHandleAndAccept(name)
		if err != nil {
			klog.V(4).Infof("Error trying to work out if we can handle %s: %v", name, err)
		}
		if canHandle {
			if !canAccept {
				klog.V(3).Infof("Factory %q can handle container %q, but ignoring.", factory, name)
				return nil, false, nil
			}
			klog.V(3).Infof("Using factory %q for container %q", factory, name)
			handle, err := factory.NewContainerHandler(name, inHostNamespace)
			return handle, canAccept, err
		} else {
			klog.V(4).Infof("Factory %q was unable to handle container %q", factory, name)
		}
	}

	return nil, false, fmt.Errorf("no known factory can handle creation of container")
}

// Clear the known factories.
func ClearContainerHandlerFactories() {
	factoriesLock.Lock()
	defer factoriesLock.Unlock()

	factories = map[watcher.ContainerWatchSource][]ContainerHandlerFactory{}
}

func DebugInfo() map[string][]string {
	factoriesLock.RLock()
	defer factoriesLock.RUnlock()

	// Get debug information for all factories.
	out := make(map[string][]string)
	for _, factoriesSlice := range factories {
		for _, factory := range factoriesSlice {
			for k, v := range factory.DebugInfo() {
				out[k] = v
			}
		}
	}
	return out
}
