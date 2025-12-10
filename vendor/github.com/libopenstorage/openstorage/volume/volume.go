package volume

import (
	"errors"

	"github.com/libopenstorage/openstorage/api"
)

var (
	// ErrAlreadyShutdown returned when driver is shutdown
	ErrAlreadyShutdown = errors.New("VolumeDriverProvider already shutdown")
	// ErrExit returned when driver already registered
	ErrExist = errors.New("Already exists")
	// ErrDriverNotFound returned when a driver is not registered
	ErrDriverNotFound = errors.New("Driver implementation not found")
	// ErrDriverInitializing returned when a driver is initializing
	ErrDriverInitializing = errors.New("Driver is initializing")
	// ErrEnoEnt returned when volume does not exist
	ErrEnoEnt = errors.New("Volume does not exist.")
	// ErrEnomem returned when we are out of memory
	ErrEnomem = errors.New("Out of memory.")
	// ErrEinval returned when an invalid input is provided
	ErrEinval = errors.New("Invalid argument")
	// ErrVolDetached returned when volume is in detached state
	ErrVolDetached = errors.New("Volume is detached")
	// ErrVolAttached returned when volume is in attached state
	ErrVolAttached = errors.New("Volume is attached")
	// ErrVolAttachedOnRemoteNode returned when volume is in attached on different node
	ErrVolAttachedOnRemoteNode = errors.New("Volume is attached on another node")
	// ErrVolAttachedScale returned when volume is attached and can be scaled
	ErrVolAttachedScale = errors.New("Volume is attached on another node." +
		" Increase scale factor to create more instances")
	// ErrVolHasSnaps returned when volume has previous snapshots
	ErrVolHasSnaps = errors.New("Volume has snapshots associated")
	// ErrNotSupported returned when the operation is not supported
	ErrNotSupported = errors.New("Operation not supported")
	// ErrVolBusy returned when volume is in busy state
	ErrVolBusy = errors.New("Volume is busy")
)

// Constants used by the VolumeDriver
const (
	// APIVersion for the volume management apis
	APIVersion = "v1"
	// PluginAPIBase where the docker unix socket resides
	PluginAPIBase = "/run/docker/plugins/"
	// DriverAPIBase where the osd unix socket resides
	DriverAPIBase = "/var/lib/osd/driver/"
	// MountBase for osd mountpoints
	MountBase = "/var/lib/osd/mounts/"
	// VolumeBase for osd volumes
	VolumeBase = "/var/lib/osd/"
)

const (
	// LocationConstaint is a label that specifies data location constraint.
	LocationConstraint = "LocationConstraint"
	// LocalNode is an alias for this node - similar to localhost.
	LocalNode = "LocalNode"
)

// AttachOptionsKey specifies a key type from a key-value pair
// that will be passed in to the Attach api
type AttachOptionsKey string

const (
	AttachOptionsSecret = AttachOptionsKey("SECRET_KEY")
)

// Store defines the interface for basic volume store operations
type Store interface {
	// Lock volume specified by volumeID.
	Lock(volumeID string) (interface{}, error)
	// Lock volume with token obtained from call to Lock.
	Unlock(token interface{}) error
	// CreateVol returns error if volume with the same ID already existe.
	CreateVol(vol *api.Volume) error
	// GetVol from volumeID.
	GetVol(volumeID string) (*api.Volume, error)
	// UpdateVol with vol
	UpdateVol(vol *api.Volume) error
	// DeleteVol. Returns error if volume does not exist.
	DeleteVol(volumeID string) error
}

// VolumeDriver is the main interface to be implemented by any storage driver.
// Every driver must at minimum implement the ProtoDriver sub interface.
type VolumeDriver interface {
	IODriver
	ProtoDriver
	BlockDriver
	Enumerator
}

// IODriver interfaces applicable to object store interfaces.
type IODriver interface {
	// Read sz bytes from specified volume at specified offset.
	// Return number of bytes read and error.
	Read(volumeID string, buf []byte, sz uint64, offset int64) (int64, error)
	// Write sz bytes from specified volume at specified offset.
	// Return number of bytes written and error.
	Write(volumeID string, buf []byte, sz uint64, offset int64) (int64, error)
	// Flush writes to stable storage.
	// Return error.
	Flush(volumeID string) error
}

// SnapshotDriver interfaces provides snapshot capability
type SnapshotDriver interface {
	// Snapshot create volume snapshot.
	// Errors ErrEnoEnt may be returned
	Snapshot(volumeID string, readonly bool, locator *api.VolumeLocator) (string, error)
	// Restore restores volume to specified snapshot.
	Restore(volumeID string, snapshotID string) error
}

// StatsDriver interface provides stats features
type StatsDriver interface {
	// Stats for specified volume.
	// cumulative stats are /proc/diskstats style stats.
	// nonCumulative stats are stats for specific duration.
	// Errors ErrEnoEnt may be returned
	Stats(volumeID string, cumulative bool) (*api.Stats, error)
	// UsedSize returns currently used volume size.
	// Errors ErrEnoEnt may be returned.
	UsedSize(volumeID string) (uint64, error)
	// GetActiveRequests get active requests
	GetActiveRequests() (*api.ActiveRequests, error)
}

// ProtoDriver must be implemented by all volume drivers.  It specifies the
// most basic functionality, such as creating and deleting volumes.
type ProtoDriver interface {
	SnapshotDriver
	StatsDriver
	// Name returns the name of the driver.
	Name() string
	// Type of this driver
	Type() api.DriverType
	// Create a new Vol for the specific volume spec.
	// It returns a system generated VolumeID that uniquely identifies the volume
	Create(locator *api.VolumeLocator, Source *api.Source, spec *api.VolumeSpec) (string, error)
	// Delete volume.
	// Errors ErrEnoEnt, ErrVolHasSnaps may be returned.
	Delete(volumeID string) error
	// Mount volume at specified path
	// Errors ErrEnoEnt, ErrVolDetached may be returned.
	Mount(volumeID string, mountPath string) error
	// MountedAt return volume mounted at specified mountpath.
	MountedAt(mountPath string) string
	// Unmount volume at specified path
	// Errors ErrEnoEnt, ErrVolDetached may be returned.
	Unmount(volumeID string, mountPath string) error
	// Update not all fields of the spec are supported, ErrNotSupported will be thrown for unsupported
	// updates.
	Set(volumeID string, locator *api.VolumeLocator, spec *api.VolumeSpec) error
	// Status returns a set of key-value pairs which give low
	// level diagnostic status about this driver.
	Status() [][2]string
	// Shutdown and cleanup.
	Shutdown()
}

// Enumerator provides a set of interfaces to get details on a set of volumes.
type Enumerator interface {
	// Inspect specified volumes.
	// Returns slice of volumes that were found.
	Inspect(volumeIDs []string) ([]*api.Volume, error)
	// Enumerate volumes that map to the volumeLocator. Locator fields may be regexp.
	// If locator fields are left blank, this will return all volumes.
	Enumerate(locator *api.VolumeLocator, labels map[string]string) ([]*api.Volume, error)
	// Enumerate snaps for specified volumes
	SnapEnumerate(volID []string, snapLabels map[string]string) ([]*api.Volume, error)
}

// StoreEnumerator combines Store and Enumerator capabilities
type StoreEnumerator interface {
	Store
	Enumerator
}

// BlockDriver needs to be implemented by block volume drivers.  Filesystem volume
// drivers can ignore this interface and include the builtin DefaultBlockDriver.
type BlockDriver interface {
	// Attach map device to the host.
	// On success the devicePath specifies location where the device is exported
	// Errors ErrEnoEnt, ErrVolAttached may be returned.
	Attach(volumeID string, attachOptions map[string]string) (string, error)
	// Detach device from the host.
	// Errors ErrEnoEnt, ErrVolDetached may be returned.
	Detach(volumeID string, unmountBeforeDetach bool) error
}

// VolumeDriverProvider provides VolumeDrivers.
type VolumeDriverProvider interface {
	// Get gets the VolumeDriver for the given name.
	// If a VolumeDriver was not created for the given name, the error ErrDriverNotFound is returned.
	Get(name string) (VolumeDriver, error)
	// Shutdown shuts down all volume drivers.
	Shutdown() error
}

// VolumeDriverRegistry registers VolumeDrivers.
type VolumeDriverRegistry interface {
	VolumeDriverProvider
	// New creates the VolumeDriver for the given name.
	// If a VolumeDriver was already created for the given name, the error ErrExist is returned.
	Register(name string, params map[string]string) error

	// Add inserts a new VolumeDriver provider with a well known name.
	Add(name string, init func(map[string]string) (VolumeDriver, error)) error
}

// NewVolumeDriverRegistry constructs a new VolumeDriverRegistry.
func NewVolumeDriverRegistry(nameToInitFunc map[string]func(map[string]string) (VolumeDriver, error)) VolumeDriverRegistry {
	return newVolumeDriverRegistry(nameToInitFunc)
}
