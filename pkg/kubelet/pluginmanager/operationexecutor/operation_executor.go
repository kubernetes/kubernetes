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

// Package operationexecutor implements interfaces that enable execution of
// register and unregister operations with a
// goroutinemap so that more than one operation is never triggered
// on the same plugin.
package operationexecutor

import (
	"time"

	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/pkg/util/goroutinemap"
)

// OperationExecutor defines a set of operations for registering and unregistering
// a plugin that are executed with a NewGoRoutineMap which
// prevents more than one operation from being triggered on the same socket path.
//
// These operations should be idempotent (for example, RegisterPlugin should
// still succeed if the plugin is already registered, etc.). However,
// they depend on the plugin handlers (for each plugin type) to implement this
// behavior.
//
// Once an operation completes successfully, the actualStateOfWorld is updated
// to indicate the plugin is registered/unregistered.
//
// Once the operation is started, since it is executed asynchronously,
// errors are simply logged and the goroutine is terminated without updating
// actualStateOfWorld.
type OperationExecutor interface {
	// RegisterPlugin registers the given plugin using the a handler in the plugin handler map.
	// It then updates the actual state of the world to reflect that.
	RegisterPlugin(socketPath string, timestamp time.Time, pluginHandlers map[string]cache.PluginHandler, actualStateOfWorld ActualStateOfWorldUpdater) error

	// UnregisterPlugin deregisters the given plugin using a handler in the given plugin handler map.
	// It then updates the actual state of the world to reflect that.
	UnregisterPlugin(pluginInfo cache.PluginInfo, actualStateOfWorld ActualStateOfWorldUpdater) error
}

// NewOperationExecutor returns a new instance of OperationExecutor.
func NewOperationExecutor(
	operationGenerator OperationGenerator) OperationExecutor {

	return &operationExecutor{
		pendingOperations:  goroutinemap.NewGoRoutineMap(true /* exponentialBackOffOnError */),
		operationGenerator: operationGenerator,
	}
}

// ActualStateOfWorldUpdater defines a set of operations updating the actual
// state of the world cache after successful registration/deregistration.
type ActualStateOfWorldUpdater interface {
	// AddPlugin add the given plugin in the cache if no existing plugin
	// in the cache has the same socket path.
	// An error will be returned if socketPath is empty.
	AddPlugin(pluginInfo cache.PluginInfo) error

	// RemovePlugin deletes the plugin with the given socket path from the actual
	// state of world.
	// If a plugin does not exist with the given socket path, this is a no-op.
	RemovePlugin(socketPath string)
}

type operationExecutor struct {
	// pendingOperations keeps track of pending attach and detach operations so
	// multiple operations are not started on the same volume
	pendingOperations goroutinemap.GoRoutineMap

	// operationGenerator is an interface that provides implementations for
	// generating volume function
	operationGenerator OperationGenerator
}

var _ OperationExecutor = &operationExecutor{}

func (oe *operationExecutor) IsOperationPending(socketPath string) bool {
	return oe.pendingOperations.IsOperationPending(socketPath)
}

func (oe *operationExecutor) RegisterPlugin(
	socketPath string,
	timestamp time.Time,
	pluginHandlers map[string]cache.PluginHandler,
	actualStateOfWorld ActualStateOfWorldUpdater) error {
	generatedOperation :=
		oe.operationGenerator.GenerateRegisterPluginFunc(socketPath, timestamp, pluginHandlers, actualStateOfWorld)

	return oe.pendingOperations.Run(
		socketPath, generatedOperation)
}

func (oe *operationExecutor) UnregisterPlugin(
	pluginInfo cache.PluginInfo,
	actualStateOfWorld ActualStateOfWorldUpdater) error {
	generatedOperation :=
		oe.operationGenerator.GenerateUnregisterPluginFunc(pluginInfo, actualStateOfWorld)

	return oe.pendingOperations.Run(
		pluginInfo.SocketPath, generatedOperation)
}
