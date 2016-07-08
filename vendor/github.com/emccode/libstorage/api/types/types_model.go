package types

// StorageType is the type of storage a driver provides.
type StorageType string

const (
	// Block is block storage.
	Block StorageType = "block"

	// NAS is network attached storage.
	NAS StorageType = "nas"

	// Object is object-backed storage.
	Object StorageType = "object"
)

// VolumeMap is the response for listing volumes for a single service.
type VolumeMap map[string]*Volume

// SnapshotMap is the response for listing snapshots for a single service.
type SnapshotMap map[string]*Snapshot

// ServiceVolumeMap is the response for listing volumes for multiple services.
type ServiceVolumeMap map[string]VolumeMap

// ServiceSnapshotMap is the response for listing snapshots for multiple
// services.
type ServiceSnapshotMap map[string]SnapshotMap

// ServicesMap is the response when getting one to many ServiceInfos.
type ServicesMap map[string]*ServiceInfo

// ExecutorsMap is the response when getting one to many ExecutorInfos.
type ExecutorsMap map[string]*ExecutorInfo

// Instance provides information about a storage object.
type Instance struct {
	// The ID of the instance to which the object is connected.
	InstanceID *InstanceID `json:"instanceID" yaml:"instanceID,omitempty"`

	// The name of the instance.
	Name string `json:"name,omitempty" yaml:",omitempty"`

	// The name of the provider that owns the object.
	ProviderName string `json:"providerName" yaml:"providerName,omitempty"`

	// The region from which the object originates.
	Region string `json:"region,omitempty" yaml:",omitempty"`

	// Fields are additional properties that can be defined for this type.
	Fields map[string]string `json:"fields,omitempty" yaml:",omitempty"`
}

// MountInfo reveals information about a particular mounted filesystem. This
// struct is populated from the content in the /proc/<pid>/mountinfo file.
type MountInfo struct {
	// ID is a unique identifier of the mount (may be reused after umount).
	ID int `json:"id" yaml:"id"`

	// Parent indicates the ID of the mount parent (or of self for the top of
	// the mount tree).
	Parent int `json:"parent"`

	// Major indicates one half of the device ID which identifies the device
	// class.
	Major int `json:"major"`

	// Minor indicates one half of the device ID which identifies a specific
	// instance of device.
	Minor int `json:"minor"`

	// Root of the mount within the filesystem.
	Root string `json:"root"`

	// MountPoint indicates the mount point relative to the process's root.
	MountPoint string `json:"mountPoint" yaml:"mountPoint"`

	// Opts represents mount-specific options.
	Opts string `json:"opts"`

	// Optional represents optional fields.
	Optional string `json:"optional"`

	// FSType indicates the type of filesystem, such as EXT3.
	FSType string `json:"fsType" yaml:"fsType"`

	// Source indicates filesystem specific information or "none".
	Source string `json:"source"`

	// VFSOpts represents per super block options.
	VFSOpts string `json:"vfsOpts" yaml:"vfsOpts"`

	// Fields are additional properties that can be defined for this type.
	Fields map[string]string `json:"fields,omitempty" yaml:",omitempty"`
}

// Snapshot provides information about a storage-layer snapshot.
type Snapshot struct {
	// A description of the snapshot.
	Description string `json:"description,omitempty" yaml:",omitempty"`

	// The name of the snapshot.
	Name string `json:"name,omitempty" yaml:",omitempty"`

	// The snapshot's ID.
	ID string `json:"id" yaml:"id"`

	// The time (epoch) at which the request to create the snapshot was submitted.
	StartTime int64 `json:"startTime,omitempty" yaml:"startTime,omitempty"`

	// The status of the snapshot.
	Status string `json:"status,omitempty" yaml:",omitempty"`

	// The ID of the volume to which the snapshot belongs.
	VolumeID string `json:"volumeID,omitempty" yaml:"volumeID,omitempty"`

	// The size of the volume to which the snapshot belongs.
	VolumeSize int64 `json:"volumeSize,omitempty" yaml:"volumeSize,omitempty"`

	// Fields are additional properties that can be defined for this type.
	Fields map[string]string `json:"fields,omitempty" yaml:",omitempty"`
}

// Volume provides information about a storage volume.
type Volume struct {
	// The volume's attachments.
	Attachments []*VolumeAttachment `json:"attachments,omitempty" yaml:",omitempty"`

	// The availability zone for which the volume is available.
	AvailabilityZone string `json:"availabilityZone,omitempty" yaml:"availabilityZone,omitempty"`

	// The volume IOPs.
	IOPS int64 `json:"iops,omitempty" yaml:"iops,omitempty"`

	// The name of the volume.
	Name string `json:"name"`

	// NetworkName is the name the device is known by in order to discover
	// locally.
	NetworkName string `json:"networkName,omitempty" yaml:"networkName,omitempty"`

	// The size of the volume.
	Size int64 `json:"size,omitempty" yaml:",omitempty"`

	// The volume status.
	Status string `json:"status,omitempty" yaml:",omitempty"`

	// ID is a piece of information that uniquely identifies the volume on
	// the storage platform to which the volume belongs. A volume ID is not
	// guaranteed to be unique across multiple, configured services.
	ID string `json:"id" yaml:"id"`

	// The volume type.
	Type string `json:"type"`

	// Fields are additional properties that can be defined for this type.
	Fields map[string]string `json:"fields,omitempty" yaml:",omitempty"`
}

// VolumeName returns the volume's name.
func (v *Volume) VolumeName() string {
	return v.Name
}

// MountPoint returns the volume's mount point, if one is present.
func (v *Volume) MountPoint() string {
	if len(v.Attachments) == 0 {
		return ""
	}
	return v.Attachments[0].MountPoint
}

// VolumeAttachment provides information about an object attached to a
// storage volume.
type VolumeAttachment struct {
	// The name of the device on which the volume to which the object is
	// attached is mounted.
	DeviceName string `json:"deviceName" yaml:"deviceName,omitempty"`

	// MountPoint is the mount point for the volume. This field is set when a
	// volume is retrieved via an integration driver.
	MountPoint string `json:"mountPoint,omitempty" yaml:"mountPoint,omitempty"`

	// The ID of the instance on which the volume to which the attachment
	// belongs is mounted.
	InstanceID *InstanceID `json:"instanceID" yaml:"instanceID,omitempty"`

	// The status of the attachment.
	Status string `json:"status" yaml:",omitempty"`

	// The ID of the volume to which the attachment belongs.
	VolumeID string `json:"volumeID" yaml:"volumeID,omitempty"`

	// Fields are additional properties that can be defined for this type.
	Fields map[string]string `json:"fields,omitempty" yaml:",omitempty"`
}

// VolumeDevice provides information about a volume's backing storage
// device. This might be a block device, NAS device, object device, etc.
type VolumeDevice struct {
	// The name of the device.
	Name string `json:"name"`

	// The ID of the instance to which the device is connected.
	InstanceID *InstanceID `json:"instanceID,omitempty" yaml:"instanceID,omitempty"`

	// The name of the network on which the device resides.
	NetworkName string `json:"networkName,omitempty" yaml:"networkName,omitempty"`

	// The name of the provider that owns the block device.
	ProviderName string `json:"providerName,omitempty" yaml:"providerName,omitempty"`

	// The region from which the device originates.
	Region string `json:"region,omitempty" yaml:",omitempty"`

	// The device status.
	Status string `json:"status"`

	// The ID of the volume for which the device is mounted.
	VolumeID string `json:"volumeID" yaml:"volumeID"`

	// Fields are additional properties that can be defined for this type.
	Fields map[string]string `json:"fields,omitempty" yaml:",omitempty"`
}

// ExecutorInfo contains information about a client-side executor, such as
// its name and MD5 checksum.
type ExecutorInfo struct {

	// Name is the name of the executor.
	Name string `json:"name"`

	// MD5Checksum is the MD5 checksum of the executor. This can be used to
	// determine if a local copy of the executor needs to be updated.
	MD5Checksum string `json:"md5checksum" yaml:"md5checksum"`

	// Size is the size of the executor in bytes.
	Size int64 `json:"size"`

	// LastModified is the time the executor was last modified as an epoch.
	LastModified int64 `json:"lastModified" yaml:"lastModified"`
}

// ServiceInfo is information about a service.
type ServiceInfo struct {
	// Name is the service's name.
	Name string `json:"name"`

	// Instance is the service's instance.
	Instance *Instance `json:"instance,omitempty" yaml:",omitempty"`

	// Driver is the name of the driver registered for the service.
	Driver *DriverInfo `json:"driver"`
}

// DriverInfo is information about a driver.
type DriverInfo struct {
	// Name is the driver's name.
	Name string `json:"name"`

	// Type is the type of storage the driver provides: block, nas, object.
	Type StorageType `json:"type"`

	// NextDevice is the next available device information for the service.
	NextDevice *NextDeviceInfo `json:"nextDevice,omitempty" yaml:"nextDevice,omitempty"`
}

// NextDeviceInfo assists the libStorage client in determining the
// next available device name by providing the driver's device prefix and
// optional pattern.
//
// For example, the Amazon Web Services (AWS) device prefix is "xvd" and its
// pattern is "[a-z]". These two values would be used to determine on an EC2
// instance where "/dev/xvda" and "/dev/xvdb" are in use that the next
// available device name is "/dev/xvdc".
//
// If the Ignore field is set to true then the client logic does not invoke the
// GetNextAvailableDeviceName function prior to submitting an AttachVolume
// request to the server.
type NextDeviceInfo struct {
	// Ignore is a flag that indicates whether the client logic should invoke
	// the GetNextAvailableDeviceName function prior to submitting an
	// AttachVolume request to the server.
	Ignore bool `json:"ignore"`

	// Prefix is the first part of a device path's value after the "/dev/"
	// porition. For example, the prefix in "/dev/xvda" is "xvd".
	Prefix string `json:"prefix"`

	// Pattern is the regex to match the part of a device path after the prefix.
	Pattern string `json:"pattern"`
}

// TaskState is the possible state of a task.
type TaskState string

const (
	// TaskStateQueued is the state for a task that has been enqueued but not
	// yet started.
	TaskStateQueued TaskState = "queued"

	// TaskStateRunning is the state for a task that is running.
	TaskStateRunning = "running"

	// TaskStateSuccess is the state for a task that has completed successfully.
	TaskStateSuccess = "success"

	// TaskStateError is the state for a task that has completed with an error.
	TaskStateError = "error"
)

// Task is a representation of an asynchronous, long-running task.
type Task struct {
	// ID is the task's ID.
	ID int `json:"id" yaml:"id"`

	// User is the name of the user that created the task.
	User string `json:"user,omitempty" yaml:",omitempty"`

	// CompleteTime is the time stamp when the task was completed
	// (whether success or failure).
	CompleteTime int64 `json:"completeTime,omitempty" yaml:"completeTime,omitempty"`

	// QueueTime is the time stamp when the task was created.
	QueueTime int64 `json:"queueTime" yaml:"queueTime"`

	// StartTime is the time stamp when the task started running.
	StartTime int64 `json:"startTime,omitempty" yaml:"startTime,omitempty"`

	// State is the current state of the task.
	State TaskState `json:"state"`

	// Result holds the result of the task.
	Result interface{} `json:"result,omitempty" yaml:",omitempty"`

	// Error contains the error if the task was unsuccessful.
	Error error `json:"error,omitempty" yaml:",omitempty"`
}
