/*
Copyright 2024 The Kubernetes Authors.

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

package state

import (
	"time"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

type ClaimInfoStateList []ClaimInfoState

// +k8s:deepcopy-gen=true
type ClaimInfoState struct {
	// ClaimUID is the UID of a resource claim
	ClaimUID types.UID

	// ClaimName is the name of a resource claim
	ClaimName string

	// Namespace is a claim namespace
	Namespace string

	// PodUIDs is a set of pod UIDs that reference a resource
	PodUIDs sets.Set[string]

	// DriverState contains information about all drivers which have allocation
	// results in the claim, even if they don't provide devices for their results.
	DriverState map[string]DriverState
}

// DriverState is used to store per-device claim info state in a checkpoint
// +k8s:deepcopy-gen=true
type DriverState struct {
	Devices []Device
}

// Device is how a DRA driver described an allocated device in a claim
// to kubelet. RequestName, ShareID, and CDI device IDs are optional.
// +k8s:deepcopy-gen=true
type Device struct {
	PoolName     string
	DeviceName   string
	ShareID      *types.UID
	RequestNames []string
	CDIDeviceIDs []string
}

// DevicesHealthMap is a map between driver names and the list of the device's health.
type DevicesHealthMap map[string]DriverHealthState

// DriverHealthState is used to store health information of all devices of a driver.
type DriverHealthState struct {
	// Devices maps a device's unique key ("<pool>/<device>") to its health state.
	Devices map[string]DeviceHealth
}

type DeviceHealthStatus string

const (
	// DeviceHealthStatusHealthy represents a healthy device.
	DeviceHealthStatusHealthy DeviceHealthStatus = "Healthy"
	// DeviceHealthStatusUnhealthy represents an unhealthy device.
	DeviceHealthStatusUnhealthy DeviceHealthStatus = "Unhealthy"
	// DeviceHealthStatusUnknown represents a device with unknown health status.
	DeviceHealthStatusUnknown DeviceHealthStatus = "Unknown"
)

// DeviceHealth is used to store health information of a device.
type DeviceHealth struct {
	// PoolName is the name of the pool where the device is allocated.
	PoolName string

	// DeviceName is the name of the device.
	// The full identifier is '<driver name>/<pool name>/<device name>' across the system.
	DeviceName string

	// Health is the health status of the device.
	// Statuses: "Healthy", "Unhealthy", "Unknown".
	Health DeviceHealthStatus

	// LastUpdated keeps track of the last health status update of this device.
	LastUpdated time.Time

	// HealthCheckTimeout is the timeout for the health check of the device.
	// Zero value means use the default timeout (DefaultHealthTimeout).
	// This ensures backward compatibility with existing data.
	HealthCheckTimeout time.Duration
}
