/*
Copyright 2017 The Kubernetes Authors.

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

package deviceplugin

import (
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

// MonitorCallback is the function called when a device's health state changes,
// or new devices are reported, or old devices are deleted.
// Updated contains the most recent state of the Device.
type MonitorCallback func(resourceName string, added, updated, deleted []*pluginapi.Device)

// Manager manages all the Device Plugins running on a node.
type Manager interface {
	// Start starts the gRPC Registration service.
	Start() error

	// Devices is the map of devices that have registered themselves
	// against the manager.
	// The map key is the ResourceName of the device plugins.
	Devices() map[string][]*pluginapi.Device

	// Allocate takes resourceName and list of device Ids, and calls the
	// gRPC Allocate on the device plugin matching the resourceName.
	Allocate(string, []string) (*pluginapi.AllocateResponse, error)

	// Stop stops the manager.
	Stop() error

	// Returns checkpoint file path.
	CheckpointFile() string
}

// TODO: evaluate whether we need these error definitions.
const (
	// errFailedToDialDevicePlugin is the error raised when the device plugin could not be
	// reached on the registered socket
	errFailedToDialDevicePlugin = "failed to dial device plugin:"
	// errUnsuportedVersion is the error raised when the device plugin uses an API version not
	// supported by the Kubelet registry
	errUnsuportedVersion = "unsupported API version by the Kubelet registry"
	// errDevicePluginAlreadyExists is the error raised when a device plugin with the
	// same Resource Name tries to register itself
	errDevicePluginAlreadyExists = "another device plugin already registered this Resource Name"
	// errInvalidResourceName is the error raised when a device plugin is registering
	// itself with an invalid ResourceName
	errInvalidResourceName = "the ResourceName is invalid"
	// errEmptyResourceName is the error raised when the resource name field is empty
	errEmptyResourceName = "invalid Empty ResourceName"

	// errBadSocket is the error raised when the registry socket path is not absolute
	errBadSocket = "bad socketPath, must be an absolute path:"
	// errRemoveSocket is the error raised when the registry could not remove the existing socket
	errRemoveSocket = "failed to remove socket while starting device plugin registry, with error"
	// errListenSocket is the error raised when the registry could not listen on the socket
	errListenSocket = "failed to listen to socket while starting device plugin registry, with error"
	// errListAndWatch is the error raised when ListAndWatch ended unsuccessfully
	errListAndWatch = "listAndWatch ended unexpectedly for device plugin %s with error %v"
)
