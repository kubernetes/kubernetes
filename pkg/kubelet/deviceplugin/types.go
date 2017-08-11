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
	"sync"

	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

// MonitorCallback is the function called when a device becomes
// unhealthy (or healthy again)
// Updated contains the most recent state of the Device
type MonitorCallback func(resourceName string, added, updated, deleted []*pluginapi.Device)

// Manager manages the Device Plugins running on a machine
type Manager interface {
	// Start starts the gRPC service
	Start() error
	// Devices is the map of devices that have registered themselves
	// against the manager.
	// The map key is the ResourceName of the device plugins
	Devices() map[string][]*pluginapi.Device

	// Allocate is calls the gRPC Allocate on the device plugin
	Allocate(string, []*pluginapi.Device) (*pluginapi.AllocateResponse, error)

	// Stop stops the manager
	Stop() error
}

// ManagerImpl is the structure in charge of managing Device Plugins
type ManagerImpl struct {
	socketname string
	socketdir  string

	Endpoints map[string]*endpoint // Key is ResourceName
	mutex     sync.Mutex

	callback MonitorCallback

	server *grpc.Server
}

const (
	// ErrDevicePluginUnknown is the error raised when the device Plugin returned by Monitor is not know by the Device Plugin manager
	ErrDevicePluginUnknown = "Manager does not have device plugin for device:"
	// ErrDeviceUnknown is the error raised when the device returned by Monitor is not know by the Device Plugin manager
	ErrDeviceUnknown = "Could not find device in it's Device Plugin's Device List:"
	// ErrBadSocket is the error raised when the registry socket path is not absolute
	ErrBadSocket = "Bad socketPath, must be an absolute path:"
	// ErrRemoveSocket is the error raised when the registry could not remove the existing socket
	ErrRemoveSocket = "Failed to remove socket while starting device plugin registry, with error"
	// ErrListenSocket is the error raised when the registry could not listen on the socket
	ErrListenSocket = "Failed to listen to socket while starting device plugin registry, with error"
	// ErrListAndWatch is the error raised when ListAndWatch ended unsuccessfully
	ErrListAndWatch = "ListAndWatch ended unexpectedly for device plugin %s with error %v"
)
