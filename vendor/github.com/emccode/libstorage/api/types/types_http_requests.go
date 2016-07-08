package types

// NewRequestObjFunc is a function that creates a new instance of the type to
// which the request body is serialized.
type NewRequestObjFunc func() interface{}

// VolumeCreateRequest is the JSON body for creating a new volume.
type VolumeCreateRequest struct {
	Name             string                 `json:"name"`
	AvailabilityZone *string                `json:"availabilityZone,omitempty"`
	IOPS             *int64                 `json:"iops,omitempty"`
	Size             *int64                 `json:"size,omitempty"`
	Type             *string                `json:"type,omitempty"`
	Opts             map[string]interface{} `json:"opts,omitempty"`
}

// VolumeCopyRequest is the JSON body for copying a volume.
type VolumeCopyRequest struct {
	VolumeName string                 `json:"volumeName"`
	Opts       map[string]interface{} `json:"opts,omitempty"`
}

// VolumeSnapshotRequest is the JSON body for snapshotting a volume.
type VolumeSnapshotRequest struct {
	SnapshotName string                 `json:"snapshotName"`
	Opts         map[string]interface{} `json:"opts,omitempty"`
}

// VolumeAttachRequest is the JSON body for attaching a volume to an instance.
type VolumeAttachRequest struct {
	Force          bool                   `json:"force,omitempty"`
	NextDeviceName *string                `json:"nextDeviceName,omitempty"`
	Opts           map[string]interface{} `json:"opts,omitempty"`
}

// VolumeDetachRequest is the JSON body for detaching a volume from an instance.
type VolumeDetachRequest struct {
	Force bool                   `json:"force,omitempty"`
	Opts  map[string]interface{} `json:"opts,omitempty"`
}

// SnapshotCopyRequest is the JSON body for copying a snapshot.
type SnapshotCopyRequest struct {
	SnapshotName  string                 `json:"snapshotName"`
	DestinationID string                 `json:"destinationID"`
	Opts          map[string]interface{} `json:"opts,omitempty"`
}

// SnapshotRemoveRequest is the JSON body for removing a snapshot.
type SnapshotRemoveRequest struct {
	Opts map[string]interface{} `json:"opts,omitempty"`
}
