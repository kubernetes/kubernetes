package configs

import "github.com/opencontainers/runc/libcontainer/devices"

type (
	// Deprecated: use libcontainer/devices.Device
	Device = devices.Device

	// Deprecated: use libcontainer/devices.Rule
	DeviceRule = devices.Rule

	// Deprecated: use libcontainer/devices.Type
	DeviceType = devices.Type

	// Deprecated: use libcontainer/devices.Permissions
	DevicePermissions = devices.Permissions
)
