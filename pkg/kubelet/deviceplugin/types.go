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

// TODO: evaluate whether we need these error definitions.
const (
	// ErrFailedToDialDevicePlugin is the error raised when the device plugin could not be
	// reached on the registered socket
	ErrFailedToDialDevicePlugin = "failed to dial device plugin:"
	// ErrUnsuportedVersion is the error raised when the device plugin uses an API version not
	// supported by the Kubelet registry
	ErrUnsuportedVersion = "unsupported API version by the Kubelet registry"
	// ErrDevicePluginAlreadyExists is the error raised when a device plugin with the
	// same Resource Name tries to register itself
	ErrDevicePluginAlreadyExists = "another device plugin already registered this Resource Name"
	// ErrInvalidResourceName is the error raised when a device plugin is registering
	// itself with an invalid ResourceName
	ErrInvalidResourceName = "the ResourceName is invalid"
	// ErrEmptyResourceName is the error raised when the resource name field is empty
	ErrEmptyResourceName = "invalid Empty ResourceName"

	// ErrBadSocket is the error raised when the registry socket path is not absolute
	ErrBadSocket = "bad socketPath, must be an absolute path:"
	// ErrRemoveSocket is the error raised when the registry could not remove the existing socket
	ErrRemoveSocket = "failed to remove socket while starting device plugin registry, with error"
	// ErrListenSocket is the error raised when the registry could not listen on the socket
	ErrListenSocket = "failed to listen to socket while starting device plugin registry, with error"
	// ErrListAndWatch is the error raised when ListAndWatch ended unsuccessfully
	ErrListAndWatch = "listAndWatch ended unexpectedly for device plugin %s with error %v"
)
