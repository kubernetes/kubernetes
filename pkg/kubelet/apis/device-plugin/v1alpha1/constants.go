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

const (
	// Healthy means that the device is healty
	Healty = "Healthy"
	// UnHealthy means that the device is unhealty
	Unhealthy = "Unhealthy"

	// HeartbeatOk means that the heartbeat succeeded
	HeartbeatOk = "Heartbeat ok"
	// HeartbeatFailure means that the heartbeat did not succeed
	HeartbeatFailure = "Heartbeat Failure"

	// Version is the API version
	Version = "alpha"
	// DevicePluginPath is the folder the Device Plugin is expecting sockets to be on
	DevicePluginPath = "/var/run/kubernetes/"
	// KubeletSocket is the path the Kubelet registry socket
	KubeletSocket = DevicePluginPath + "kubelet.sock"

	// InvalidChars are the characters that may not appear in a Vendor or Kind field
	InvalidChars = "/-"

	// ErrFailedToDialDevicePlugin is the error raised when the device plugin could not be
	// reached on the registered socket
	ErrFailedToDialDevicePlugin = "Failed to dial device plugin:"
	// ErrUnsuportedVersion is the error raised when the device plugin uses an API version not
	// supported by the Kubelet registry
	ErrUnsuportedVersion = "Unsupported version"
	// ErrDevicePluginAlreadyExists is the error raised when a device plugin already
	// registered itself on a socket
	ErrDevicePluginAlreadyExists = "Another device plugin is already on this socket:"
	// ErrVendorMismatch is the error raised when a Vendor is not consistent accross a
	// Device Plugin fields
	ErrVendorMismatch = "Vendor mismatch for device and Vendor:"
	// ErrEmptyDevice is the error raised when the Kind or name field of a device is empty
	ErrEmptyDevice = "Invalid Empty Kind or name for device:"
	// ErrEmptyDevice is the error raised when the vendor field is empty
	ErrEmptyVendor = "Invalid Empty vendor"
	// ErrInvalidDeviceKind is the error raised when the Kind or name field of a
	// device is invalid
	ErrInvalidDeviceKind = "Kind should not contain any of '" + InvalidChars +
		"' for device:"
	// ErrInvalidDeviceKind is the error raised when the Vendor field is invalid
	ErrInvalidVendor = "Vendor should not contain any of '" + InvalidChars +
		"' for device:"
)
