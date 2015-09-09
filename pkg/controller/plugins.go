/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package controller

import (
	"fmt"
	"sync"
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"

	"github.com/golang/glog"
)

// Plugin is an interface to controller plugins that can be run
// in kube-controller-manager.  A PluginMgr handles instantiating
// and managing the lifecycle of controllers.
type Plugin interface {
	// Init initializes the plugin.  This will be called once before Run.
	Init(host Host)

	// Name returns the plugin's name.  Plugins should use namespaced names
	// such as "example.com/replication-controller".  The "kubernetes.io" namespace is
	// reserved for plugins which are bundled with kubernetes.
	Name() string

	// NewController returns an instance of this plugin's controller.
	NewController() (Controller, error)
}

// Host is an interface that plugins can use to access the kube-controller-manager
type Host interface {
	// GetKubeClient returns a client interface
	GetKubeClient() client.Interface
}

// PluginMgr is an interface to a component that manages the lifecycle of controller plugins.
type PluginMgr interface {
	InitPlugins(plugins []Plugin, host Host) error
	RunAll()
	StopAll()
	Status() []PluginStatus
	FindPluginByName(name string) (Plugin, error)
}

// pluginMgr is a private implementation of PluginMgr creatable through its factory func NewPluginMgr
type pluginMgr struct {
	mutex   sync.Mutex
	plugins map[string]*managedPlugin
}

// NewPluginMgr returns a new instance of pluginMgr
func NewPluginMgr() *pluginMgr {
	return &pluginMgr{
		plugins: make(map[string]*managedPlugin),
	}
}

// managedPlugin is a wrapper around Plugin that adds runtime information
type managedPlugin struct {
	// plugin is the actual plugin being run by the plugin manager
	plugin Plugin
	// controller is the instance of a controller started by the plugin manager
	controller Controller
	// running is the runtime status of the plugin.
	// True means the plugin is running correctly.
	// False means either the controller has not yet been run or an error has occurred
	running bool
	// err is any error caught during InitPlugins or RunAll.
	// Only plugins without errors after Init will be run
	err error
}

// PluginStatus is a value object that describes a plugin's current status
type PluginStatus struct {
	// Name is the name of the plugin
	Name string
	// Running is the runtime status of the plugin.  See managedPlugin.running
	Running bool
	// Err is any error caught for this plugin.  See managedPlugin.err
	Err error
}

// ControllerConfig is how controller plugins receive configuration.  An instance of ControllerConfig specific to
// the plugin will be passed to the plugin's ProbeControllerPlugins(config) func.  Only the binary hosting the
// plugins knows what plugins are included.  This requires specific configuration to happen in the binary which
// is passed downward generically via ControllerConfig.
//
// Volumes in ControllerConfig are intended to be relevant to several plugins, but not necessarily all plugins.
// The preference is to leverage strong typing in this struct.  All config items must have a descriptive but
// non-specific name (e.g, SyncPeriod is OK but SyncPeriodForNodeController is !OK).
//
// OtherAttributes is a map of string values intended for one-off configuration relevant to a single plugin. All
// values are passed by string and require interpretation by the plugin.  Passing config strings is the least
// desirable option.  The preference is for common attributes and strong typing for config.
type ControllerConfig struct {
	SyncPeriod      time.Duration
	OtherAttributes map[string]string
}

// InitPlugins initializes each plugin.
// This must be called exactly once before any New* methods are called on any plugins.
func (mgr *pluginMgr) InitPlugins(plugins []Plugin, host Host) error {
	mgr.mutex.Lock()
	defer mgr.mutex.Unlock()

	allErrs := []error{}
	for _, plugin := range plugins {
		name := plugin.Name()
		if !util.IsQualifiedName(name) {
			allErrs = append(allErrs, fmt.Errorf("controller plugin has invalid name: %#v", plugin))
			continue
		}

		if _, found := mgr.plugins[name]; found {
			allErrs = append(allErrs, fmt.Errorf("controller plugin %q was registered more than once", name))
			continue
		}

		plugin.Init(host)
		mgr.plugins[name] = &managedPlugin{plugin: plugin}
		glog.V(1).Infof("Loaded controller plugin %q", name)
	}
	return errors.NewAggregate(allErrs)
}

// FindPluginByName fetches a plugin by name.  If no plugin is found, returns error.
func (mgr *pluginMgr) FindPluginByName(name string) (Plugin, error) {
	mgr.mutex.Lock()
	defer mgr.mutex.Unlock()
	if managedPlugin, ok := mgr.plugins[name]; ok {
		return managedPlugin.plugin, nil
	}
	return nil, fmt.Errorf("no controller plugin found matching %s", name)
}

// RunAll attempts to create Controller instances for each plugin and run them.  This method will not fail
// for any particular plugin.  Runtime Status is kept for all plugins.  Callers should use pluginMgr.Status
func (mgr *pluginMgr) RunAll() {
	mgr.mutex.Lock()
	defer mgr.mutex.Unlock()
	for name, managedPlugin := range mgr.plugins {
		// do not attempt to run a plugin that had previous errors or is already running.
		// this check makes RunAll idempotent.
		if managedPlugin.err != nil || managedPlugin.running {
			continue
		}
		managedPlugin.controller, managedPlugin.err = managedPlugin.plugin.NewController()
		if managedPlugin.err != nil {
			glog.V(1).Infof("Unexpected error creating controller %s: %v", name, managedPlugin.err)
			continue
		}

		managedPlugin.err = managedPlugin.controller.Run()
		if managedPlugin.err != nil {
			glog.V(1).Infof("Unexpected error running controller %s: %v", name, managedPlugin.err)
			continue
		}

		glog.V(3).Infof("Controller plugin %s is running", name)
		managedPlugin.running = true
	}
}

// StopAll stops all Controllers by closing their stop channels
func (mgr *pluginMgr) StopAll() {
	mgr.mutex.Lock()
	defer mgr.mutex.Unlock()

	for name, managedPlugin := range mgr.plugins {
		// do not attempt to stop a plugin that isn't running or that has previous errors
		// this checks makes StopAll idempotent
		if managedPlugin.err != nil || !managedPlugin.running {
			continue
		}
		managedPlugin.err = managedPlugin.controller.Stop()
		if managedPlugin.err != nil {
			glog.V(1).Infof("Unexpected error stopping controller %s: %v", name, managedPlugin.err)
			continue
		}

		glog.V(3).Infof("Controller plugin %s is stopped", name)
		managedPlugin.running = false
	}
}

// Status returns the runtime status of all plugins under management
func (mgr *pluginMgr) Status() []PluginStatus {
	statuses := []PluginStatus{}
	for name, managedPlugin := range mgr.plugins {
		status := PluginStatus{
			Name:    name,
			Running: managedPlugin.running,
			Err:     managedPlugin.err,
		}
		statuses = append(statuses, status)
	}
	return statuses
}

// controllerHost is a basic implementation of Host
type ControllerHost struct {
	client client.Interface
}

func (host *ControllerHost) GetKubeClient() client.Interface {
	return host.client
}

func NewHost(client client.Interface) Host {
	return &ControllerHost{client}
}
