// +build linux

package water

type DevicePermissions struct {
	// ID of the user which will be granted ownership of the device.
	// If set to a negative value, the owner value will not be changed.
	// By default, Linux sets the owner to -1, which allows any user.
	Owner uint

	// ID of the group which will be granted access to the device.
	// If set to a negative value, the group value will not be changed.
	// By default, Linux sets the group to -1, which allows any group.
	Group uint
}

// PlatformSpecificParams defines parameters in Config that are specific to
// Linux. A zero-value of such type is valid, yielding an interface
// with OS defined name.
type PlatformSpecificParams struct {
	// Name is the name to be set for the interface to be created. This overrides
	// the default name assigned by OS such as tap0 or tun0. A zero-value of this
	// field, i.e. an empty string, indicates that the default name should be
	// used.
	Name string

	// Enable or disable persistence mode for the interface device.
	Persist bool

	// Owner and Group permissions for the device.
	// A zero-value of this field, i.e. nil, indicates that no changes to owner
	// or group will be made.
	Permissions *DevicePermissions
}

func defaultPlatformSpecificParams() PlatformSpecificParams {
	return PlatformSpecificParams{}
}
