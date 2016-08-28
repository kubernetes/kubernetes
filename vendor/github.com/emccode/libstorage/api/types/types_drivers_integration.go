package types

// NewIntegrationDriver is a function that constructs a new IntegrationDriver.
type NewIntegrationDriver func() IntegrationDriver

// VolumeMountOpts are options for mounting a volume.
type VolumeMountOpts struct {
	OverwriteFS bool
	NewFSType   string
	Preempt     bool
	Opts        Store
}

// VolumeMapping is a volume's name and the path to which it is mounted.
type VolumeMapping interface {
	// VolumeName returns the volume's name.
	VolumeName() string

	// MountPoint returns the volume's mount point.
	MountPoint() string

	// Status returns the volume's details for an inspect.
	Status() map[string]interface{}
}

// IntegrationDriverManager is the management wrapper for an IntegrationDriver.
type IntegrationDriverManager interface {
	IntegrationDriver

	// Driver returns the underlying driver.
	Driver() IntegrationDriver
}

// IntegrationDriver is the interface implemented to integrate external
// storage consumers, such as Docker, with libStorage.
type IntegrationDriver interface {
	Driver

	// List a map that relates volume names to their mount points.
	List(
		ctx Context,
		opts Store) ([]VolumeMapping, error)

	// Inspect returns a specific volume as identified by the provided
	// volume name.
	Inspect(
		ctx Context,
		volumeName string,
		opts Store) (VolumeMapping, error)

	// Mount will return a mount point path when specifying either a volumeName
	// or volumeID.  If a overwriteFs boolean is specified it will overwrite
	// the FS based on newFsType if it is detected that there is no FS present.
	Mount(
		ctx Context,
		volumeID, volumeName string,
		opts *VolumeMountOpts) (string, *Volume, error)

	// Unmount will unmount the specified volume by volumeName or volumeID.
	Unmount(
		ctx Context,
		volumeID, volumeName string,
		opts Store) error

	// Path will return the mounted path of the volumeName or volumeID.
	Path(
		ctx Context,
		volumeID, volumeName string,
		opts Store) (string, error)

	// Create will create a new volume with the volumeName and opts.
	Create(
		ctx Context,
		volumeName string,
		opts *VolumeCreateOpts) (*Volume, error)

	// Remove will remove a volume of volumeName.
	Remove(
		ctx Context,
		volumeName string,
		opts Store) error

	// Attach will attach a volume based on volumeName to the instance of
	// instanceID.
	Attach(
		ctx Context,
		volumeName string,
		opts *VolumeAttachOpts) (string, error)

	// Detach will detach a volume based on volumeName to the instance of
	// instanceID.
	Detach(
		ctx Context,
		volumeName string,
		opts *VolumeDetachOpts) error
}
