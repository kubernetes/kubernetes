/*
Copyright 2019 The Kubernetes Authors.

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

// Package reconciler implements interfaces that attempt to reconcile the
// desired state of the world with the actual state of the world by triggering
// relevant actions (register/deregister plugins).
package reconciler

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/operationexecutor"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
)

// Reconciler runs a periodic loop to reconcile the desired state of the world
// with the actual state of the world by triggering register and unregister
// operations. Also provides a means to add a handler for a plugin type.
type Reconciler interface {
	// Run starts running the reconciliation loop which executes periodically,
	// checks if plugins are correctly registered or unregistered.
	// If not, it will trigger register/unregister operations to rectify.
	Run(stopCh <-chan struct{})

	// AddHandler adds the given plugin handler for a specific plugin type,
	// which will be added to the actual state of world cache.
	AddHandler(pluginType string, pluginHandler cache.PluginHandler)
}

// NewReconciler returns a new instance of Reconciler.
//
// operationExecutor - used to trigger register/unregister operations safely
// (prevents more than one operation from being triggered on the same
// socket path)
//
// loopSleepDuration - the amount of time the reconciler loop sleeps between
// successive executions
//
// desiredStateOfWorld - cache containing the desired state of the world
//
// actualStateOfWorld - cache containing the actual state of the world
func NewReconciler(
	operationExecutor operationexecutor.OperationExecutor,
	loopSleepDuration time.Duration,
	desiredStateOfWorld cache.DesiredStateOfWorld,
	actualStateOfWorld cache.ActualStateOfWorld) Reconciler {
	return &reconciler{
		operationExecutor:   operationExecutor,
		loopSleepDuration:   loopSleepDuration,
		desiredStateOfWorld: desiredStateOfWorld,
		actualStateOfWorld:  actualStateOfWorld,
		handlers:            make(map[string]cache.PluginHandler),
	}
}

type reconciler struct {
	operationExecutor   operationexecutor.OperationExecutor
	loopSleepDuration   time.Duration
	desiredStateOfWorld cache.DesiredStateOfWorld
	actualStateOfWorld  cache.ActualStateOfWorld
	handlers            map[string]cache.PluginHandler
	sync.RWMutex
}

var _ Reconciler = &reconciler{}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(func() {
		rc.reconcile()
	},
		rc.loopSleepDuration,
		stopCh)
}

func (rc *reconciler) AddHandler(pluginType string, pluginHandler cache.PluginHandler) {
	rc.Lock()
	defer rc.Unlock()

	rc.handlers[pluginType] = pluginHandler
}

func (rc *reconciler) getHandlers() map[string]cache.PluginHandler {
	rc.RLock()
	defer rc.RUnlock()

	var copyHandlers = make(map[string]cache.PluginHandler)
	for pluginType, handler := range rc.handlers {
		copyHandlers[pluginType] = handler
	}
	return copyHandlers
}

func (rc *reconciler) reconcile() {
	// Unregisterations are triggered before registrations

	// Ensure plugins that should be unregistered are unregistered.
	for _, registeredPlugin := range rc.actualStateOfWorld.GetRegisteredPlugins() {
		unregisterPlugin := false
		if !rc.desiredStateOfWorld.PluginExists(registeredPlugin.SocketPath) {
			unregisterPlugin = true
		} else {
			// We also need to unregister the plugins that exist in both actual state of world
			// and desired state of world cache, but the timestamps don't match.
			// Iterate through desired state of world plugins and see if there's any plugin
			// with the same socket path but different timestamp.
			for _, dswPlugin := range rc.desiredStateOfWorld.GetPluginsToRegister() {
				if dswPlugin.SocketPath == registeredPlugin.SocketPath && dswPlugin.UUID != registeredPlugin.UUID {
					klog.V(5).InfoS("An updated version of plugin has been found, unregistering the plugin first before reregistering", "plugin", registeredPlugin)
					unregisterPlugin = true
					break
				}
			}
		}

		if unregisterPlugin {
			klog.V(5).InfoS("Starting operationExecutor.UnregisterPlugin", "plugin", registeredPlugin)
			err := rc.operationExecutor.UnregisterPlugin(registeredPlugin, rc.actualStateOfWorld)
			if err != nil &&
				!goroutinemap.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore goroutinemap.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				klog.ErrorS(err, "OperationExecutor.UnregisterPlugin failed", "plugin", registeredPlugin)
			}
			if err == nil {
				klog.V(1).InfoS("OperationExecutor.UnregisterPlugin started", "plugin", registeredPlugin)
			}
		}
	}

	// Ensure plugins that should be registered are registered
	for _, pluginToRegister := range rc.desiredStateOfWorld.GetPluginsToRegister() {
		if !rc.actualStateOfWorld.PluginExistsWithCorrectUUID(pluginToRegister) {
			klog.V(5).InfoS("Starting operationExecutor.RegisterPlugin", "plugin", pluginToRegister)
			err := rc.operationExecutor.RegisterPlugin(pluginToRegister.SocketPath, pluginToRegister.UUID, rc.getHandlers(), rc.actualStateOfWorld)
			if err != nil &&
				!goroutinemap.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore goroutinemap.IsAlreadyExists and exponentialbackoff.IsExponentialBackoff errors, they are expected.
				klog.ErrorS(err, "OperationExecutor.RegisterPlugin failed", "plugin", pluginToRegister)
			}
			if err == nil {
				klog.V(1).InfoS("OperationExecutor.RegisterPlugin started", "plugin", pluginToRegister)
			}
		}
	}
}
