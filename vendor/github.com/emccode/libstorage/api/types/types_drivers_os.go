package types

// NewOSDriver is a function that constructs a new OSDriver.
type NewOSDriver func() OSDriver

// DeviceMountOpts are options when mounting a device.
type DeviceMountOpts struct {
	MountOptions string
	MountLabel   string
	Opts         Store
}

// DeviceFormatOpts are options when formatting a device.
type DeviceFormatOpts struct {
	NewFSType   string
	OverwriteFS bool
	Opts        Store
}

// OSDriverManager is the management wrapper for an OSDriver.
type OSDriverManager interface {
	OSDriver

	// Driver returns the underlying driver.
	Driver() OSDriver
}

// OSDriver is the interface implemented by types that provide OS introspection
// and management.
type OSDriver interface {
	Driver

	// Mounts get a list of mount points for a local device.
	Mounts(
		ctx Context,
		deviceName, mountPoint string,
		opts Store) ([]*MountInfo, error)

	// Mount mounts a device to a specified path.
	Mount(
		ctx Context,
		deviceName, mountPoint string,
		opts *DeviceMountOpts) error

	// Unmount unmounts the underlying device from the specified path.
	Unmount(
		ctx Context,
		mountPoint string,
		opts Store) error

	// IsMounted checks whether a path is mounted or not
	IsMounted(
		ctx Context,
		mountPoint string,
		opts Store) (bool, error)

	// Format formats a device.
	Format(
		ctx Context,
		deviceName string,
		opts *DeviceFormatOpts) error
}
