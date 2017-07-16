/*
Copyright 2016 The Kubernetes Authors.

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
	"sync"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

const (
	// ErrDevicePluginUnknown is the error raised when the device Plugin returned by
	// Monitor is not know by the Device Plugin manager
	ErrDevicePluginUnknown = "Manager does not have device plugin for device:"
	// ErrDeviceUnknown is the error raised when the device returned by Monitor
	// is not know by the Device Plugin manager
	ErrDeviceUnknown = "Could not find device in it's Device Plugin's Device List:"
	// ErrBadSocket is the error raised when the registry socket path is not absolute
	ErrBadSocket = "Bad socketPath, must be an absolute path:"
	// ErrRemoveSocket is the error raised when the registry could not remove the existing
	// socket
	ErrRemoveSocket = "Failed to remove socket while starting device plugin registry," +
		" with error"
	// ErrListenSocket is the error raised when the registry could not listen on the socket
	ErrListenSocket = "Failed to listen to socket while starting device plugin registry," +
		" with error"
)

// MonitorCallback is the is the function called when a device becomes
// unhealthy (or healthy again)
type MonitorCallback func(*pluginapi.Device)

// Manager is the structure in charge of managing Device Plugins
type Manager struct {
	registry *registry

	// Key is Kind
	devices   map[string][]*pluginapi.Device
	available map[string][]*pluginapi.Device

	// Key is vendor
	vendors map[string][]*pluginapi.Device

	mutex sync.Mutex

	callback MonitorCallback
}
