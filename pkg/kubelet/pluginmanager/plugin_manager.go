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

package pluginmanager

import (
	"context"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/metrics"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/operationexecutor"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/pluginwatcher"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/reconciler"
)

// PluginManager runs a set of asynchronous loops that figure out which plugins
// need to be registered/deregistered and makes it so.
type PluginManager interface {
	// Starts the plugin manager and all the asynchronous loops that it controls
	Run(ctx context.Context, sourcesReady config.SourcesReady, stopCh <-chan struct{})

	// AddHandler adds the given plugin handler for a specific plugin type, which
	// will be added to the actual state of world cache so that it can be passed to
	// the desired state of world cache in order to be used during plugin
	// registration/deregistration
	AddHandler(pluginType string, pluginHandler cache.PluginHandler)

	// Stopped returns a closed channel once Run() has fully exited.
	Stopped() <-chan struct{}
}

const (
	// loopSleepDuration is the amount of time the reconciler loop waits
	// between successive executions
	loopSleepDuration = 1 * time.Second
)

// NewPluginManager returns a new concrete instance implementing the
// PluginManager interface.
func NewPluginManager(
	sockDir string,
	recorder record.EventRecorder) PluginManager {
	asw := cache.NewActualStateOfWorld()
	dsw := cache.NewDesiredStateOfWorld()
	reconciler := reconciler.NewReconciler(
		operationexecutor.NewOperationExecutor(
			operationexecutor.NewOperationGenerator(
				recorder,
			),
		),
		loopSleepDuration,
		dsw,
		asw,
	)

	pm := &pluginManager{
		desiredStateOfWorldPopulator: pluginwatcher.NewWatcher(
			sockDir,
			dsw,
		),
		reconciler:          reconciler,
		desiredStateOfWorld: dsw,
		actualStateOfWorld:  asw,
		stopped:             make(chan struct{}),
	}
	// Close "stopped" channel immediately so Stopped() returns a closed channel
	// if Run() is never called.
	close(pm.stopped)
	return pm
}

// pluginManager implements the PluginManager interface
type pluginManager struct {
	// desiredStateOfWorldPopulator (the plugin watcher) runs an asynchronous
	// periodic loop to populate the desiredStateOfWorld.
	desiredStateOfWorldPopulator *pluginwatcher.Watcher

	// reconciler runs an asynchronous periodic loop to reconcile the
	// desiredStateOfWorld with the actualStateOfWorld by triggering register
	// and unregister operations using the operationExecutor.
	reconciler reconciler.Reconciler

	// actualStateOfWorld is a data structure containing the actual state of
	// the world according to the manager: i.e. which plugins are registered.
	// The data structure is populated upon successful completion of register
	// and unregister actions triggered by the reconciler.
	actualStateOfWorld cache.ActualStateOfWorld

	// desiredStateOfWorld is a data structure containing the desired state of
	// the world according to the plugin manager: i.e. what plugins are registered.
	// The data structure is populated by the desired state of the world
	// populator (plugin watcher).
	desiredStateOfWorld cache.DesiredStateOfWorld

	// stopped is closed when Run() has fully exited.
	// It's created closed and is reopened when Run() is called.
	stopped chan struct{}

	// stoppedMu protects access to stopped channel.
	stoppedMu sync.RWMutex

	// runOnce ensures Run() logic executes only once.
	runOnce sync.Once
}

var _ PluginManager = &pluginManager{}

func (pm *pluginManager) Run(ctx context.Context, sourcesReady config.SourcesReady, stopCh <-chan struct{}) {
	pm.runOnce.Do(func() {
		// Reopen the channel since it was created closed.
		pm.stoppedMu.Lock()
		pm.stopped = make(chan struct{})
		pm.stoppedMu.Unlock()

		defer close(pm.stopped)
		defer runtime.HandleCrashWithContext(ctx)

		// Check if a shutdown was requested before manager initialization.
		// This prevents the filesystem/watcher setup from immediate Kubelet
		// shutdowns either production or in the test scope.
		select {
		case <-stopCh:
			return
		default:
		}

		logger := klog.FromContext(ctx)

		if err := pm.desiredStateOfWorldPopulator.Start(ctx, stopCh); err != nil {
			logger.Error(err, "The desired_state_of_world populator (plugin watcher) starts failed!")
			return
		}

		logger.V(2).Info("The desired_state_of_world populator (plugin watcher) starts")

		logger.Info("Starting Kubelet Plugin Manager")
		go pm.reconciler.Run(stopCh)

		metrics.Register(pm.actualStateOfWorld, pm.desiredStateOfWorld)
		<-stopCh
		logger.Info("Shutting down Kubelet Plugin Manager")

		// Wait for both reconciler and plugin watcher to stop
		<-pm.reconciler.Stopped()
		<-pm.desiredStateOfWorldPopulator.Stopped()
	})
}

func (pm *pluginManager) AddHandler(pluginType string, handler cache.PluginHandler) {
	pm.reconciler.AddHandler(pluginType, handler)
}

// Stopped returns a channel that is closed once Run() has fully exited.
// If Run() was never called, the returned channel is already closed.
func (pm *pluginManager) Stopped() <-chan struct{} {
	pm.stoppedMu.RLock()
	defer pm.stoppedMu.RUnlock()
	return pm.stopped
}
