package volume

import (
	"context"
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
	// ErrAttachedHostSpecNotFound returned when the attached host's spec is not found
	ErrAttachedHostSpecNotFound = errors.New("Spec of the attached host is not found")
	// ErrVolAttached returned when volume is in attached state
	ErrVolAttached = errors.New("Volume is attached")
        // ErrVolAttachedOnRemoteNode returned when volume is attached on different node
        ErrVolAttachedOnRemoteNode = errors.New("Volume is attached on another node")
        // ErrNonSharedVolAttachedOnRemoteNode returned when a non-shared volume is attached on different node
        ErrNonSharedVolAttachedOnRemoteNode = errors.New("Non-shared volume is already attached on another node." +
                " Non-shared volumes can only be attached on one node at a time.")
	// ErrVolAttachedScale returned when volume is attached and can be scaled
	ErrVolAttachedScale = errors.New("Volume is attached on another node." +
		" Increase scale factor to create more instances")
	// ErrVolHasSnaps returned when volume has previous snapshots
	ErrVolHasSnaps = errors.New("Volume has snapshots associated")
	// ErrNotSupported returned when the operation is not supported
	ErrNotSupported = errors.New("Operation not supported")
	// ErrVolBusy returned when volume is in busy state
	ErrVolBusy = errors.New("Volume is busy")
	// ErrAborted returned when capacityUsageInfo cannot be returned
	ErrAborted = errors.New("Aborted CapacityUsage request")
	// ErrInvalidName returned when Cloudbackup Name/request is invalid
	ErrInvalidName = errors.New("Invalid name for cloud backup/restore request")
	// ErrFsResizeFailed returned when Filesystem resize failed because of filesystem
	// errors
	ErrFsResizeFailed = errors.New("Filesystem Resize failed due to filesystem errors")
	// ErrNoVolumeUpdate is returned when a volume update has no changes requested
	ErrNoVolumeUpdate = errors.New("No change requested")
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
	// FromTrashCan is a label that specified a volume being in the TrashCan
	FromTrashCan = "FromTrashCan"
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
	Snapshot(volumeID string, readonly bool, locator *api.VolumeLocator, noRetry bool) (string, error)
	// Restore restores volume to specified snapshot.
	Restore(volumeID string, snapshotID string) error
	// SnapshotGroup takes a snapshot of a group of volumes that can be specified with either of the following
	//	1. group ID
	//	2. labels
	//	3. volumeIDs
	//	deleteOnFailure indicates whether to delete the successful snaps if some of the snapshots failed
	SnapshotGroup(groupID string, labels map[string]string, volumeIDs []string, deleteOnFailure bool) (*api.GroupSnapCreateResponse, error)
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
	// CapacityUsage returns both exclusive and shared usage
	// of a snap/volume
	CapacityUsage(ID string) (*api.CapacityUsageResponse, error)
	// VolumeUsageByNode returns capacity usage of all volumes and snaps for a
	// given node
	VolumeUsageByNode(nodeID string) (*api.VolumeUsageByNode, error)
}

type QuiesceDriver interface {
	// Freezes mounted filesystem resulting in a quiesced volume state.
	// Only one freeze operation may be active at any given time per volume.
	// Unfreezes after timeout seconds if it is non-zero.
	// An optional quiesceID can be passed for driver-specific use.
	Quiesce(volumeID string, timeoutSeconds uint64, quiesceID string) error
	// Unfreezes mounted filesystem if it was frozen.
	Unquiesce(volumeID string) error
}

// CloudBackupDriver interface provides Cloud backup features
type CloudBackupDriver interface {
	// CloudBackupCreate uploads snapshot of a volume to the cloud
	CloudBackupCreate(input *api.CloudBackupCreateRequest) (*api.CloudBackupCreateResponse, error)
	// CloudBackupGroupCreate creates and then uploads volumegroup snapshots
	CloudBackupGroupCreate(input *api.CloudBackupGroupCreateRequest) (*api.CloudBackupGroupCreateResponse, error)
	// CloudBackupRestore downloads a cloud backup and restores it to a volume
	CloudBackupRestore(input *api.CloudBackupRestoreRequest) (*api.CloudBackupRestoreResponse, error)
	// CloudBackupEnumerate enumerates the backups for a given cluster/credential/volumeID
	CloudBackupEnumerate(input *api.CloudBackupEnumerateRequest) (*api.CloudBackupEnumerateResponse, error)
	// CloudBackupDelete deletes the specified backup in cloud
	CloudBackupDelete(input *api.CloudBackupDeleteRequest) error
	// CloudBackupDelete deletes all the backups for a given volume in cloud
	CloudBackupDeleteAll(input *api.CloudBackupDeleteAllRequest) error
	// CloudBackupStatus indicates the most recent status of backup/restores
	CloudBackupStatus(input *api.CloudBackupStatusRequest) (*api.CloudBackupStatusResponse, error)
	// CloudBackupCatalog displays listing of backup content
	CloudBackupCatalog(input *api.CloudBackupCatalogRequest) (*api.CloudBackupCatalogResponse, error)
	// CloudBackupHistory displays past backup/restore operations on a volume
	CloudBackupHistory(input *api.CloudBackupHistoryRequest) (*api.CloudBackupHistoryResponse, error)
	// CloudBackupStateChange allows a current backup state transisions(pause/resume/stop)
	CloudBackupStateChange(input *api.CloudBackupStateChangeRequest) error
	// CloudBackupSchedCreate creates a schedule to backup volume to cloud
	CloudBackupSchedCreate(input *api.CloudBackupSchedCreateRequest) (*api.CloudBackupSchedCreateResponse, error)
	// CloudBackupGroupSchedCreate creates a schedule to backup a volumegroup to cloud
	CloudBackupGroupSchedCreate(input *api.CloudBackupGroupSchedCreateRequest) (*api.CloudBackupSchedCreateResponse, error)
	// CloudBackupSchedCreate creates a schedule to backup volume to cloud
	CloudBackupSchedUpdate(input *api.CloudBackupSchedUpdateRequest) error
	// CloudBackupGroupSchedCreate creates a schedule to backup a volumegroup to cloud
	CloudBackupGroupSchedUpdate(input *api.CloudBackupGroupSchedUpdateRequest) error
	// CloudBackupSchedDelete delete a backup schedule
	CloudBackupSchedDelete(input *api.CloudBackupSchedDeleteRequest) error
	// CloudBackupSchedEnumerate enumerates the configured backup schedules in the cluster
	CloudBackupSchedEnumerate() (*api.CloudBackupSchedEnumerateResponse, error)
	// CloudBackupSize fetches the size of a cloud backup
	CloudBackupSize(input *api.SdkCloudBackupSizeRequest) (*api.SdkCloudBackupSizeResponse, error)
}

// CloudMigrateDriver interface provides Cloud migration features
type CloudMigrateDriver interface {
	// CloudMigrateStart starts a migrate operation
	CloudMigrateStart(request *api.CloudMigrateStartRequest) (*api.CloudMigrateStartResponse, error)
	// CloudMigrateCancel cancels a migrate operation
	CloudMigrateCancel(request *api.CloudMigrateCancelRequest) error
	// CloudMigrateStatus returns status for the migration operations
	CloudMigrateStatus(request *api.CloudMigrateStatusRequest) (*api.CloudMigrateStatusResponse, error)
}

// FilesystemTrimDriver interface exposes APIs to manage filesystem trim
// operation on a volume
type FilesystemTrimDriver interface {
	// FilesystemTrimStart starts a filesystem trim background operation on a
	// specified volume
	FilesystemTrimStart(request *api.SdkFilesystemTrimStartRequest) (*api.SdkFilesystemTrimStartResponse, error)
	// FilesystemTrimStatus returns the status of a filesystem trim
	// background operation on a specified volume, if any
	FilesystemTrimStatus(request *api.SdkFilesystemTrimStatusRequest) (*api.SdkFilesystemTrimStatusResponse, error)
	// AutoFilesystemTrimStatus returns the status of auto fs trim
	// operations on volumes
	AutoFilesystemTrimStatus(request *api.SdkAutoFSTrimStatusRequest) (*api.SdkAutoFSTrimStatusResponse, error)
	// AutoFilesystemTrimUsage returns the volume usage and trimmable
	// space of locally mounted pxd volumes
	AutoFilesystemTrimUsage(request *api.SdkAutoFSTrimUsageRequest) (*api.SdkAutoFSTrimUsageResponse, error)
	// FilesystemTrimStop stops a filesystem trim background operation on
	// a specified volume, if any
	FilesystemTrimStop(request *api.SdkFilesystemTrimStopRequest) (*api.SdkFilesystemTrimStopResponse, error)
	// AutoFilesystemTrimPush pushes an autofstrim job to queue
	AutoFilesystemTrimPush(request *api.SdkAutoFSTrimPushRequest) (*api.SdkAutoFSTrimPushResponse, error)
	// AutoFilesystemTrimPop pops an autofstrim job from queue
	AutoFilesystemTrimPop(request *api.SdkAutoFSTrimPopRequest) (*api.SdkAutoFSTrimPopResponse, error)
}

// FilesystemCheckDriver interface exposes APIs to manage filesystem check
// operation on a volume
type FilesystemCheckDriver interface {
	// FilesystemCheckStart starts a filesystem check background operation
	// on a specified volume
	FilesystemCheckStart(request *api.SdkFilesystemCheckStartRequest) (*api.SdkFilesystemCheckStartResponse, error)
	// FilesystemCheckStatus returns the status of a filesystem check
	// background operation on the filesystem of a specified volume, if any.
	FilesystemCheckStatus(request *api.SdkFilesystemCheckStatusRequest) (*api.SdkFilesystemCheckStatusResponse, error)
	// FilesystemCheckStop stops the filesystem check background operation on
	// the filesystem of a specified volume, if any.
	FilesystemCheckStop(request *api.SdkFilesystemCheckStopRequest) (*api.SdkFilesystemCheckStopResponse, error)
}

// ProtoDriver must be implemented by all volume drivers.  It specifies the
// most basic functionality, such as creating and deleting volumes.
type ProtoDriver interface {
	SnapshotDriver
	StatsDriver
	QuiesceDriver
	CredsDriver
	CloudBackupDriver
	CloudMigrateDriver
	FilesystemTrimDriver
	FilesystemCheckDriver
	// Name returns the name of the driver.
	Name() string
	// Type of this driver
	Type() api.DriverType
	// Version information of the driver
	Version() (*api.StorageVersion, error)
	// Create a new Vol for the specific volume spec.
	// It returns a system generated VolumeID that uniquely identifies the volume
	Create(ctx context.Context, locator *api.VolumeLocator, Source *api.Source, spec *api.VolumeSpec) (string, error)
	// Delete volume.
	// Errors ErrEnoEnt, ErrVolHasSnaps may be returned.
	Delete(ctx context.Context, volumeID string) error
	// Mount volume at specified path
	// Errors ErrEnoEnt, ErrVolDetached may be returned.
	Mount(ctx context.Context, volumeID string, mountPath string, options map[string]string) error
	// MountedAt return volume mounted at specified mountpath.
	MountedAt(ctx context.Context, mountPath string) string
	// Unmount volume at specified path
	// Errors ErrEnoEnt, ErrVolDetached may be returned.
	Unmount(ctx context.Context, volumeID string, mountPath string, options map[string]string) error
	// Update not all fields of the spec are supported, ErrNotSupported will be thrown for unsupported
	// updates.
	Set(volumeID string, locator *api.VolumeLocator, spec *api.VolumeSpec) error
	// Status returns a set of key-value pairs which give low
	// level diagnostic status about this driver.
	Status() [][2]string
	// Shutdown and cleanup.
	Shutdown()
	// DU specified volume and potentially the subfolder if provided.
	Catalog(volumeid, subfolder string, depth string) (api.CatalogResponse, error)
	// Does a Filesystem Trim operation to free unused space to block device(block discard)
	VolService(volumeID string, vsreq *api.VolumeServiceRequest) (*api.VolumeServiceResponse, error)
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
	Attach(ctx context.Context, volumeID string, attachOptions map[string]string) (string, error)
	// Detach device from the host.
	// Errors ErrEnoEnt, ErrVolDetached may be returned.
	Detach(ctx context.Context, volumeID string, options map[string]string) error
}

// CredsDriver provides methods to handle credentials
type CredsDriver interface {
	// CredsCreate creates credential for a given cloud provider
	CredsCreate(params map[string]string) (string, error)
	// CredsUpdate updates credential for an already configured credential
	CredsUpdate(name string, params map[string]string) error
	// CredsEnumerate lists the configured credentials in the cluster
	CredsEnumerate() (map[string]interface{}, error)
	// CredsDelete deletes the credential associated credUUID
	CredsDelete(credUUID string) error
	// CredsValidate validates the credential associated credUUID
	CredsValidate(credUUID string) error
	// CredsDeleteReferences delets any  with the creds
	CredsDeleteReferences(credUUID string) error
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

	// Removes driver from registry. Does nothing if driver name does not exist.
	Remove(name string)
}

// NewVolumeDriverRegistry constructs a new VolumeDriverRegistry.
func NewVolumeDriverRegistry(nameToInitFunc map[string]func(map[string]string) (VolumeDriver, error)) VolumeDriverRegistry {
	return newVolumeDriverRegistry(nameToInitFunc)
}
